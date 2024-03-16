import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import clip
from model import (
    UNetwithControl,
    SensorModel,
    ControlNet,
    StatefulControlNet,
    StatefulUNet,
)
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import copy
import time
import random
import pickle
import json
import pdb


class Engine:
    def __init__(self, model_path_1, model_path_2):
        self.batch_size = 1
        self.dim_x = 10
        self.dim_gt = 10
        self.channel_img_1 = 2
        self.win_size = 24
        self.global_step = 0
        self.mode = "Test"
        self.checkpoint_path_1 = model_path_1
        self.checkpoint_path_2 = model_path_2

        dataset = "duck"  # "duck"

        if dataset == "Drum":
            self.base_model = StatefulUNet(dim_x=self.dim_x, window_size=self.win_size)
            self.model = StatefulControlNet(dim_x=self.dim_x, window_size=self.win_size)
        else:
            self.base_model = UNetwithControl(
                dim_x=self.dim_x, window_size=self.win_size
            )
            self.model = ControlNet(dim_x=self.dim_x, window_size=self.win_size)
        self.sensor_model = SensorModel(
            state_est=1,
            dim_x=self.dim_x,
            emd_size=256,
            input_channel=self.channel_img_1,
        )

        # -----------------------------------------------------------------------------#
        # ----------------------------    diffusion API     ---------------------------#
        # -----------------------------------------------------------------------------#
        num_diffusion_iters = 50
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # additional parameters
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

        # -----------------------------------------------------------------------------#
        # ---------------------------    get model ready     --------------------------#
        # -----------------------------------------------------------------------------#
        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda()
            self.sensor_model.cuda()
            self.base_model.cuda()
        self.clip_model, preprocess = clip.load("ViT-B/32", device=self.device)

        # -----------------------------------------------------------------------------#
        # ---------------------------------    setup     ------------------------------#
        # -----------------------------------------------------------------------------#
        if torch.cuda.is_available():
            checkpoint_1 = torch.load(self.checkpoint_path_1)
            self.model.load_state_dict(checkpoint_1["model"])
            checkpoint_2 = torch.load(self.checkpoint_path_2)
            self.sensor_model.load_state_dict(checkpoint_2["model"])
        else:
            checkpoint_1 = torch.load(
                self.checkpoint_path_1, map_location=torch.device("cpu")
            )
            checkpoint_2 = torch.load(
                self.checkpoint_path_2, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint_1["model"])
            self.sensor_model.load_state_dict(checkpoint_2["model"])

        # -----------------------------------------------------------------------------#
        # ---------------------------   create base model   ---------------------------#
        # -----------------------------------------------------------------------------#
        self.base_model.time_mlp.load_state_dict(self.model.time_mlp.state_dict())
        self.base_model.lang_model.load_state_dict(self.model.lang_model.state_dict())
        self.base_model.fusion_layer.load_state_dict(
            self.model.fusion_layer.state_dict()
        )
        self.base_model.downs.load_state_dict(self.model.downs.state_dict())
        self.base_model.mid_block1.load_state_dict(self.model.mid_block1.state_dict())
        self.base_model.mid_block2.load_state_dict(self.model.mid_block2.state_dict())
        self.base_model.ups.load_state_dict(self.model.ups.state_dict())
        self.base_model.final_conv.load_state_dict(self.model.final_conv.state_dict())

        if dataset == "Drum":
            self.base_model.addition_module.load_state_dict(
                self.model.addition_module.state_dict()
            )

        self.base_model.eval()
        self.model.eval()
        self.sensor_model.eval()

    def inv_transform(self, state):
        state = rearrange(state, "dim time -> time dim")

        # # tomato pick and place
        # max_state = np.array([ 0.24146586,  0.66639685,  0.2270379 ,
        #                            1.        ,  0.39946742, 0.8255449 ,
        #                             0.99999999,  0.21055005, -0.92910059, 1])
        # min_state = np.array([-0.18271977,  0.39480549, -0.01296423,
        #                            0.91674739, -0.2569546 , -0.17026   ,
        #                            0.56433644, -0.36982711, -1.        , 0])

        # # push fanta
        # max_state = np.array([0.31521671, 0.87183916, 0.09679487,
        #                     1.        ,0.7106418 , 0.9999994 ,
        #                     1.        , 0.70819508, 0.97595528, 1])
        # min_state = np.array([-0.46621527,  0.37062453, -0.01181129,
        #                         -0.97543304, -0.99999865,-0.02631372,
        #                         0.00109455, -0.99999563, -1., 0])

        # openlid
        # max_state = np.array([0.04071683,  0.62149828,  0.23503569,
        #                         1.        ,  0.30002407,0.73809373,
        #                         1.        ,  0.17969302, -0.95326051, 1])
        # min_state = np.array([-2.16162411e-01,  3.99501858e-01, -3.31081122e-04,
        #                         9.43961789e-01,-3.30054755e-01, -2.34764239e-01,
        #                         6.74698193e-01, -3.02149644e-01,-1.00000000e+00, 0])

        # duck
        max_state = np.array(
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
        min_state = np.array(
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

        # # drum
        # max_state = np.array(
        #     [
        #         -0.08253406,
        #         0.6651301,
        #         0.12477053,
        #         1.0,
        #         0.11919251,
        #         0.63730323,
        #         0.90369423,
        #         0.91832716,
        #         -0.39582221,
        #         1.0001,
        #     ]
        # )
        # min_state = np.array(
        #     [
        #         -0.13554766,
        #         0.59381776,
        #         0.00575311,
        #         0.96633573,
        #         -0.2572844,
        #         0.42817839,
        #         0.77061313,
        #         0.76308,
        #         -0.64630404,
        #         0.999,
        #     ]
        # )

        state = (state + 1) / 2.0 * (max_state - min_state) + min_state
        state = rearrange(state, "time dim -> dim time")
        return state

    def base_policy(self, img, sentence):
        # -----------------------------------------------------------------------------#
        # ---------------------------------    test      ------------------------------#
        # -----------------------------------------------------------------------------#
        # get data ready - sentence
        sentence = clip.tokenize([sentence])
        with torch.no_grad():
            text_features = self.clip_model.encode_text(sentence)
            text_features = text_features.clone().detach()
            text_features = text_features.to(torch.float32)

        # get data ready - image
        image_size = (224, 224)
        img = np.array(img.resize(image_size))[:, :, :3] / 255.0
        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        img = torch.tensor(img, dtype=torch.float32)
        img = rearrange(img, "h w ch -> ch h w")
        img = rearrange(img, "(n ch) h w -> n ch h w", n=1)

        # -----------------------------------------------------------------------------#
        # ------------------------------ diffusion sampling ---------------------------#
        # -----------------------------------------------------------------------------#
        self.noise_scheduler.set_timesteps(50)
        # initialize action from Guassian noise
        noisy_action = torch.randn((1, self.dim_x, self.win_size)).to(self.device)
        with torch.no_grad():
            img_emb = self.sensor_model(img)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                t = torch.stack([k]).to(self.device)
                predicted_noise = self.base_model(
                    noisy_action, img_emb, text_features, t
                )

                # inverse diffusion step (remove noise)
                noisy_action = self.noise_scheduler.step(
                    model_output=predicted_noise, timestep=k, sample=noisy_action
                ).prev_sample
            pred = noisy_action.cpu().detach().numpy()
            raw_pred = noisy_action
            pred = self.inv_transform(pred[0])
        return pred, raw_pred

    def controlnet(self, img, sentence, raw_pred):
        # -----------------------------------------------------------------------------#
        # ---------------------------------    test      ------------------------------#
        # -----------------------------------------------------------------------------#
        # get data ready - sentence
        sentence = clip.tokenize([sentence])
        with torch.no_grad():
            text_features = self.clip_model.encode_text(sentence)
            text_features = text_features.clone().detach()
            text_features = text_features.to(torch.float32)

        # get data ready - image
        image_size = (224, 224)
        img = np.array(img.resize(image_size))[:, :, :3] / 255.0
        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        img = torch.tensor(img, dtype=torch.float32)
        img = rearrange(img, "h w ch -> ch h w")
        img = rearrange(img, "(n ch) h w -> n ch h w", n=1)

        # -----------------------------------------------------------------------------#
        # ------------------------------ diffusion sampling ---------------------------#
        # -----------------------------------------------------------------------------#
        self.noise_scheduler.set_timesteps(50)
        # initialize action from Guassian noise
        noisy_action = torch.randn((1, self.dim_x, self.win_size)).to(self.device)
        with torch.no_grad():
            img_emb = self.sensor_model(img)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                t = torch.stack([k]).to(self.device)
                predicted_noise = self.model(
                    noisy_action, img_emb, text_features, raw_pred, t
                )

                # inverse diffusion step (remove noise)
                noisy_action = self.noise_scheduler.step(
                    model_output=predicted_noise, timestep=k, sample=noisy_action
                ).prev_sample
            pred = noisy_action.cpu().detach().numpy()
            raw_pred = noisy_action
            pred = self.inv_transform(pred[0])
        return pred, raw_pred


def transform(state):
    state = rearrange(state, "dim time -> time dim")
    # drum
    max_state = np.array(
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
    min_state = np.array(
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
    state = 2 * (state - min_state) / (max_state - min_state) - 1
    state = rearrange(state, "time dim -> dim time")
    return state


if __name__ == "__main__":
    # gat path to the model
    path_1 = "path_to_controlnet/controlnet-model-153000"
    path_2 = "path_to_sensor_model/sensor_model-225600"
    inference = Engine(path_1, path_2)

    # run inference
    img = Image.open("0.jpg")
    sentence = "pick up the "
    # sentence = 0

    # run diffusion
    save = []
    for i in range(1):
        pred, raw_pred = inference.base_policy(img, sentence)
        save.append(pred)
    with open("diffusion_out", "wb") as f:
        pickle.dump(save, f)

    # run controlnet
    with open(os.path.join("outputs.json"), "r") as fh:
        traj = json.load(fh)["traj"]
        traj = np.array(traj)
    traj = transform(traj)
    traj = torch.tensor(traj, dtype=torch.float32)
    traj = rearrange(traj, "(n dim) time -> n dim time", n=1)
    raw_pred = traj
    save = []
    for i in range(1):
        pred, raw_pred = inference.controlnet(img, sentence, raw_pred)
        save.append(pred)
    with open("controlnet_out", "wb") as f:
        pickle.dump(save, f)
