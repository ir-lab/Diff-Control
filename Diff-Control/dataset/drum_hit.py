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


class Drum(Dataset):
    def __init__(self, data_dirs, random=False, length_total=80, image_size=(224, 224)):
        # |--datadir
        #     |--trial0
        #         |--0_left.jpg
        #         |--0_right.jpg
        #         |--x_left.jpg
        #         |--states_ee.json
        #     |--trial1
        #     |--...
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        self.design_length = 120
        pad_length = True

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

            if pad_length == True:
                actual_length = len(states_dict)
                num_to_pad = 12
                # grab the last ones
                img_path = trial_dict["img_paths"][-1]
                joints = np.array([trial_dict["joint_angles"][-1, :]])
                ee = np.asarray([trial_dict["EE_xyzrpy"][-1, :]])

                for n in range(num_to_pad):
                    trial_dict["img_paths"].append(img_path)
                    trial_dict["joint_angles"] = np.concatenate(
                        (trial_dict["joint_angles"], joints), axis=0
                    )
                    trial_dict["EE_xyzrpy"] = np.concatenate(
                        (trial_dict["EE_xyzrpy"], ee), axis=0
                    )

            # There are (trial_dict['len']) steps in the trial, which means (trial_dict['len'] + 1) states
            trial_dict["len"] = trial_dict["len"] + num_to_pad
            self.trials.append(trial_dict)
            length = length + trial_dict["len"]
            self.lengths_index.append(length)

        # state = [x, y, z, gripper]
        self.max_state = np.array(
            [
                -0.08253406,
                0.6651301,
                0.12477053,
                1.0,
                0.11919251,
                0.63730323,
                0.90369423,
                0.91832716,
                -0.39582221,
                1.0001,
            ]
        )
        self.min_state = np.array(
            [
                -0.13554766,
                0.59381776,
                0.00575311,
                0.96633573,
                -0.2572844,
                0.42817839,
                0.77061313,
                0.76308,
                -0.64630404,
                0.999,
            ]
        )

    def rpy2rrppyy(self, rpy):
        rrppyy = [0] * 6
        for i in range(3):
            rrppyy[i * 2] = np.sin(rpy[i])
            rrppyy[i * 2 + 1] = np.cos(rpy[i])
        return rrppyy

    def xyz_to_xy(self, xyz):
        xy = np.dot(xyz, self.weight) + self.bias
        xy[1] = 224 - xy[1]
        return xy

    def get_actions_only(self, step_idx, trial_idx, prior):
        drum = False
        # for controlnet pad to the left and right
        if prior == True:
            if step_idx == 0:
                drum = True
                ee_traj = torch.tensor(
                    (self.trials[trial_idx]["EE_xyzrpy"][step_idx:]),
                    dtype=torch.float32,
                )
                joint_angles_traj = torch.tensor(
                    self.trials[trial_idx]["joint_angles"][step_idx:],
                    dtype=torch.float32,
                )
                to_pad = ee_traj[0]
                to_pad = rearrange(to_pad, "(n dim) -> n dim", n=1)
                ee_traj_appendix = to_pad.repeat(self.length_total, 1)
                ee_traj = ee_traj_appendix

                to_pad = joint_angles_traj[0]
                to_pad = rearrange(to_pad, "(n dim) -> n dim", n=1)
                joint_angles_traj_appendix = to_pad.repeat(self.length_total, 1)
                joint_angles_traj = joint_angles_traj_appendix
            else:
                if step_idx < self.transition_history:
                    pad_left = self.transition_history - step_idx
                    ee_traj = torch.tensor(
                        (self.trials[trial_idx]["EE_xyzrpy"][step_idx:]),
                        dtype=torch.float32,
                    )
                    joint_angles_traj = torch.tensor(
                        self.trials[trial_idx]["joint_angles"][step_idx:],
                        dtype=torch.float32,
                    )
                    length_total = self.length_total - pad_left
                    length_left = max(length_total - ee_traj.shape[0], 0)

                else:
                    step_idx = step_idx - self.transition_history
                    pad_left = 0
                    ee_traj = torch.tensor(
                        (self.trials[trial_idx]["EE_xyzrpy"][step_idx:]),
                        dtype=torch.float32,
                    )
                    joint_angles_traj = torch.tensor(
                        self.trials[trial_idx]["joint_angles"][step_idx:],
                        dtype=torch.float32,
                    )
                    length_total = self.length_total
                    length_left = max(length_total - ee_traj.shape[0], 0)

        else:
            # just diffusion
            ee_traj = torch.tensor(
                (self.trials[trial_idx]["EE_xyzrpy"][step_idx:]), dtype=torch.float32
            )
            joint_angles_traj = torch.tensor(
                self.trials[trial_idx]["joint_angles"][step_idx:], dtype=torch.float32
            )
            pad_left = 0
            length_total = self.length_total
            length_left = max(length_total - ee_traj.shape[0], 0)

        if drum == False:
            if length_left > 0 and pad_left == 0:
                ee_traj_appendix = ee_traj[-1:].repeat(length_left, 1)
                ee_traj = torch.cat((ee_traj, ee_traj_appendix), axis=0)

                joint_angles_traj_appendix = joint_angles_traj[-1:].repeat(
                    length_left, 1
                )
                joint_angles_traj = torch.cat(
                    (joint_angles_traj, joint_angles_traj_appendix), axis=0
                )
            elif length_left == 0 and pad_left == 0:
                ee_traj = ee_traj[:length_total]
                joint_angles_traj = joint_angles_traj[:length_total]

            elif length_left > 0 and pad_left > 0:
                to_pad = ee_traj[0]
                to_pad = rearrange(to_pad, "(n dim) -> n dim", n=1)
                ee_traj_appendix = to_pad.repeat(pad_left, 1)
                ee_traj = torch.cat((ee_traj_appendix, ee_traj), axis=0)

                to_pad = joint_angles_traj[0]
                to_pad = rearrange(to_pad, "(n dim) -> n dim", n=1)
                joint_angles_traj_appendix = to_pad.repeat(pad_left, 1)
                joint_angles_traj = torch.cat(
                    (joint_angles_traj_appendix, joint_angles_traj), axis=0
                )

                # pad to right
                ee_traj_appendix = ee_traj[-1:].repeat(length_left, 1)
                ee_traj = torch.cat((ee_traj, ee_traj_appendix), axis=0)

                joint_angles_traj_appendix = joint_angles_traj[-1:].repeat(
                    length_left, 1
                )
                joint_angles_traj = torch.cat(
                    (joint_angles_traj, joint_angles_traj_appendix), axis=0
                )
            elif length_left == 0 and pad_left > 0:
                ee_traj = ee_traj[:length_total]
                joint_angles_traj = joint_angles_traj[:length_total]

                to_pad = ee_traj[0]
                to_pad = rearrange(to_pad, "(n dim) -> n dim", n=1)
                ee_traj_appendix = to_pad.repeat(pad_left, 1)
                ee_traj = torch.cat((ee_traj_appendix, ee_traj), axis=0)

                to_pad = joint_angles_traj[0]
                to_pad = rearrange(to_pad, "(n dim) -> n dim", n=1)
                joint_angles_traj_appendix = to_pad.repeat(pad_left, 1)
                joint_angles_traj = torch.cat(
                    (joint_angles_traj_appendix, joint_angles_traj), axis=0
                )

        if prior == True:
            if step_idx == 0:
                gripper = joint_angles_traj[:, -1]
                gripper = rearrange(gripper, "(t k) -> t k", k=1)
                action = torch.cat((ee_traj, gripper), axis=1)
                action = (
                    2 * (action - self.min_state) / (self.max_state - self.min_state)
                    - 1
                )
                action = rearrange(action, "time dim -> dim time")
                action = action.clone().detach()
                action = action.to(torch.float32)
                action = torch.zeros_like(action)
            else:
                gripper = joint_angles_traj[:, -1]
                gripper = rearrange(gripper, "(t k) -> t k", k=1)
                action = torch.cat((ee_traj, gripper), axis=1)
                action = (
                    2 * (action - self.min_state) / (self.max_state - self.min_state)
                    - 1
                )
                action = rearrange(action, "time dim -> dim time")
                action = action.clone().detach()
                action = action.to(torch.float32)
        else:
            gripper = joint_angles_traj[:, -1]
            gripper = rearrange(gripper, "(t k) -> t k", k=1)
            action = torch.cat((ee_traj, gripper), axis=1)
            action = (
                2 * (action - self.min_state) / (self.max_state - self.min_state) - 1
            )
            action = rearrange(action, "time dim -> dim time")
            action = action.clone().detach()
            action = action.to(torch.float32)
        return action

    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        self.transition_history = 12
        check = True

        while check:
            trial_idx = bisect.bisect_right(self.lengths_index, index)
            p = random.uniform(0, 1)
            if trial_idx == 0:
                if p < 0.1:
                    step_idx = 0
                else:
                    step_idx = index
            else:
                if p < 0.1:
                    step_idx = 0
                else:
                    step_idx = index - self.lengths_index[trial_idx - 1]

            if 1 <= step_idx <= 11:
                index = random.randint(0, self.lengths_index[-1] - 1)
            else:
                check = False

        img = Image.open(self.trials[trial_idx]["img_paths"][step_idx])
        shape = img.size
        img = np.array(img.resize(self.image_size))[:, :, :3] / 255.0
        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        img = torch.tensor(img, dtype=torch.float32)

        sentence = clip.tokenize(["hit the drum for 5 times"])
        action = self.get_actions_only(step_idx, trial_idx, prior=False)
        prior_action = self.get_actions_only(step_idx, trial_idx, prior=True)

        img = rearrange(img, "h w ch -> ch h w")
        # -------------------------------------------------
        return img, prior_action, action, sentence[0]
        # return img, prior_action, action, sentence


def pad_collate_xy_lang(batch):
    (img, prior_action, action, lang) = zip(*batch)
    img = torch.stack(img)
    prior_action = torch.stack(prior_action)
    action = torch.stack(action)
    lang = torch.stack(lang)
    return img, prior_action, action, lang


# if __name__ == "__main__":
#     data_dirs = [
#         "/Users/xiaoliu/project/RSS/drum2",
#     ]
#     dataset = Drum(data_dirs)
#     for item in dataset:
#         img, prior_action, ee, sentence = item

#         gt = ee.cpu().detach().numpy()
#         prior = prior_action.cpu().detach().numpy()
#         print(gt.shape)
#         print(prior_action.shape)
