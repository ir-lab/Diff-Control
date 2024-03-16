import torch

# from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import json
import bisect
import clip
import random
from PIL import Image
from einops import rearrange, repeat
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pdb


class Duck(Dataset):
    def __init__(self, data_dirs, random=False, length_total=24, image_size=(224, 224)):
        # |--datadir
        #     |--trial0
        #         |--0_left.jpg
        #         |--0_right.jpg
        #         |--x_left.jpg
        #         |--states_ee.json
        #     |--trial1
        #     |--...
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_inst_to_verb = {
            "pick": ["pick", "pick up", "raise", "hold"],
        }
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

        all_dirs = []
        for data_dir in data_dirs:
            all_dirs = all_dirs + [f.path for f in os.scandir(data_dir) if f.is_dir()]

        self.length_total = length_total
        self.image_size = image_size
        self.trials = []
        self.lengths_index = []

        length = 0
        for trial in all_dirs:
            trial_dict = {}
            states_json = os.path.join(trial, "states_ee.json")
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            trial_dict["len"] = len(states_dict)
            trial_dict["img_paths"] = [
                os.path.join(trial, str(i) + "_left.jpg")
                for i in range(trial_dict["len"])
            ]
            trial_dict["joint_angles"] = np.asarray(
                [states_dict[i]["joints"] for i in range(trial_dict["len"])]
            )
            trial_dict["gripper_position"] = np.asarray(
                [
                    [
                        states_dict[i]["gripper_position_echo"] / 255.0
                        for i in range(trial_dict["len"])
                    ]
                ]
            ).T
            trial_dict["joint_angles"] = np.concatenate(
                (trial_dict["joint_angles"], trial_dict["gripper_position"]), axis=1
            )
            trial_dict["EE_xyzrpy"] = np.asarray(
                [
                    states_dict[i]["objects_to_track"]["EE"]["xyz"]
                    + self.rpy2rrppyy(states_dict[i]["objects_to_track"]["EE"]["rpy"])
                    for i in range(trial_dict["len"])
                ]
            )
            # There are (trial_dict['len']) steps in the trial, which means (trial_dict['len'] + 1) states
            trial_dict["len"] -= 1
            self.trials.append(trial_dict)
            length = length + trial_dict["len"]
            self.lengths_index.append(length)
        # ee = []
        # for i in range (len(self.trials)):
        #     if i ==0:
        #         ee = self.trials[i]["EE_xyzrpy"]
        #     else:
        #         ee = np.concatenate((ee, self.trials[i]["EE_xyzrpy"]), axis=0)

        # state = [x, y, z, gripper]
        self.max_state = np.array(
            [
                0.03971246,
                0.72129052,
                0.38620684,
                1.0,
                0.38974198,
                0.25328839,
                1.0,
                0.71690577,
                -0.69717008,
                1.01,
            ]
        )
        self.min_state = np.array(
            [
                -0.18053129,
                0.56153237,
                0.18126588,
                0.84830063,
                -0.52951491,
                -0.29562168,
                0.95530509,
                -0.06018639,
                -0.99999919,
                0.99,
            ]
        )

    def rpy2rrppyy(self, rpy):
        rrppyy = [0] * 6
        for i in range(3):
            rrppyy[i * 2] = np.sin(rpy[i])
            rrppyy[i * 2 + 1] = np.cos(rpy[i])
        return rrppyy

    def noun_phrase_template(self, target_id):
        self.noun_phrase = {
            0: {
                "name": ["yellow"],
                "object": ["duck"],
            },
        }
        id_name = np.random.randint(len(self.noun_phrase[target_id]["name"]))
        id_object = np.random.randint(len(self.noun_phrase[target_id]["object"]))
        name = self.noun_phrase[target_id]["name"][id_name]
        obj = self.noun_phrase[target_id]["object"][id_object]
        return (name + " " + obj).strip()

    def verb_phrase_template(self, action_inst):
        if action_inst is None:
            action_inst = random.choice(list(self.action_inst_to_verb.keys()))
        action_id = np.random.randint(len(self.action_inst_to_verb[action_inst]))
        verb = self.action_inst_to_verb[action_inst][action_id]
        return verb.strip()

    def sentence_template(self, action_inst=None):
        sentence = ""
        verb = self.verb_phrase_template(action_inst)
        sentence = sentence + verb
        sentence = sentence + " " + self.noun_phrase_template(0)
        return sentence.strip()

    def xyz_to_xy(self, xyz):
        xy = np.dot(xyz, self.weight) + self.bias
        xy[1] = 224 - xy[1]
        return xy

    def get_actions_only(self, step_idx, trial_idx):
        ee_traj = torch.tensor(
            (self.trials[trial_idx]["EE_xyzrpy"][step_idx:]), dtype=torch.float32
        )
        joint_angles_traj = torch.tensor(
            self.trials[trial_idx]["joint_angles"][step_idx:], dtype=torch.float32
        )
        length_total = self.length_total
        length_left = max(length_total - ee_traj.shape[0], 0)

        if length_left > 0:
            ee_traj_appendix = ee_traj[-1:].repeat(length_left, 1)
            ee_traj = torch.cat((ee_traj, ee_traj_appendix), axis=0)

            joint_angles_traj_appendix = joint_angles_traj[-1:].repeat(length_left, 1)
            joint_angles_traj = torch.cat(
                (joint_angles_traj, joint_angles_traj_appendix), axis=0
            )
        else:
            ee_traj = ee_traj[:length_total]
            joint_angles_traj = joint_angles_traj[:length_total]

        gripper = joint_angles_traj[:, -1]
        gripper = rearrange(gripper, "(t k) -> t k", k=1)
        action = torch.cat((ee_traj, gripper), axis=1)
        action = 2 * (action - self.min_state) / (self.max_state - self.min_state) - 1
        action = rearrange(action, "time dim -> dim time")
        action = action.clone().detach()
        action = action.to(torch.float32)
        return action

    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        transition_history = 12
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        if trial_idx == 0:
            step_idx = index
        else:
            step_idx = index - self.lengths_index[trial_idx - 1]

        # for controlnet
        if step_idx < transition_history:
            step_idx_pre = 0
        else:
            step_idx_pre = step_idx - transition_history

        img = Image.open(self.trials[trial_idx]["img_paths"][step_idx])
        shape = img.size
        img = np.array(img.resize(self.image_size))[:, :, :3] / 255.0
        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        img = torch.tensor(img, dtype=torch.float32)

        sentence = self.sentence_template()
        sentence = clip.tokenize([sentence])
        action = self.get_actions_only(step_idx, trial_idx)
        prior_action = self.get_actions_only(step_idx_pre, trial_idx)

        img = rearrange(img, "h w ch -> ch h w")
        return img, prior_action, action, sentence[0]


def pad_collate_xy_lang(batch):
    (img, prior_action, action, lang) = zip(*batch)
    img = torch.stack(img)
    prior_action = torch.stack(prior_action)
    action = torch.stack(action)
    lang = torch.stack(lang)
    return img, prior_action, action, lang


# if __name__ == "__main__":
#     data_dirs = [
#         "/tf/datasets/pick_duck"
#     ]
#     dataset = Duck(data_dirs)
#     for item in dataset:
#         img, prior_action, ee, sentence = item
#         input()
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=1,
#         collate_fn=batch_data,
#     )
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # clip_model, preprocess = clip.load("ViT-B/32", device=device)

#     for item in dataset:
#         (images, prior_state, action, pre_action, sentence, target_pos) = item
#         # print(
#         #     images.shape,
#         #     prior_state.shape,
#         #     action.shape,
#         #     pre_action.shape,
#         #     sentence.shape,
#         #     target_pos.shape,
#         # )
#         print("=======", action.shape)
#         # with torch.no_grad():
#         #     text_features = clip_model.encode_text(sentence)
#         input()
