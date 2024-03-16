import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import clip
from model import BCZ_LSTM
from dataset.lid_pick_and_place import *
from dataset.tomato_pick_and_place import *
from dataset.pick_duck import *
from dataset.drum_hit import *
from optimizer import build_optimizer
from optimizer import build_lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import copy
import time
import random
import pickle
import torch.optim as optim


class Engine:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.train.batch_size
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.dim_gt = self.args.train.dim_gt
        self.sensor_len = self.args.train.sensor_len
        self.channel_img_1 = self.args.train.channel_img_1
        self.channel_img_2 = self.args.train.channel_img_2
        self.input_size_1 = self.args.train.input_size_1
        self.input_size_2 = self.args.train.input_size_2
        self.input_size_3 = self.args.train.input_size_3
        self.num_ensemble = self.args.train.num_ensemble
        self.win_size = self.args.train.win_size
        self.global_step = 0
        self.mode = self.args.mode.mode

        if self.args.train.dataset == "OpenLid":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            self.dataset = OpenLid(self.data_path)
        elif self.args.train.dataset == "Tomato":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            self.dataset = Tomato(self.data_path)
        elif self.args.train.dataset == "Duck":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            self.dataset = Duck(self.data_path)
        elif self.args.train.dataset == "Drum":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            self.dataset = Drum(self.data_path)

        self.model = BCZ_LSTM()

        # -----------------------------------------------------------------------------#
        # ---------------------------    get model ready     --------------------------#
        # -----------------------------------------------------------------------------#
        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda()

        # -----------------------------------------------------------------------------#
        # ------------------------------- use pre-trained?  ---------------------------#
        # -----------------------------------------------------------------------------#
        # if torch.cuda.is_available():
        #     checkpoint_1 = torch.load(self.args.test.checkpoint_path_1)
        #     self.model.load_state_dict(checkpoint_1["model"])
        # else:
        #     checkpoint_1 = torch.load(
        #         self.args.test.checkpoint_path_1, map_location=torch.device("cpu")
        #     )
        #     self.model.load_state_dict(checkpoint_1["model"])

        self.clip_model, preprocess = clip.load("ViT-B/32", device=self.device)

    def train(self):
        # -----------------------------------------------------------------------------#
        # ---------------------------------    setup     ------------------------------#
        # -----------------------------------------------------------------------------#
        # self.criterion = nn.MSELoss()  # nn.MSELoss() or nn.L1Loss()
        self.criterion = nn.HuberLoss()
        # self.criterion = nn.functional.huber_loss()

        if self.args.train.dataset == "Tomato":
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
                collate_fn=tomato_pad_collate_xy_lang,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
                collate_fn=pad_collate_xy_lang,
            )
        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Total number of parameters: ", pytorch_total_params)

        optimizer_ = optim.Adam(self.model.parameters(), lr=1e-4)
        # Create LR scheduler
        if self.args.mode.mode == "train":
            num_total_steps = self.args.train.num_epochs * len(dataloader)
            scheduler = build_lr_scheduler(
                optimizer_,
                self.args.optim.lr_scheduler,
                self.args.train.learning_rate,
                num_total_steps,
                self.args.train.end_learning_rate,
            )

        # Epoch calculations
        steps_per_epoch = len(dataloader)
        num_total_steps = self.args.train.num_epochs * steps_per_epoch
        epoch = self.global_step // steps_per_epoch
        duration = 0

        # tensorboard writer
        self.writer = SummaryWriter(
            f"./experiments/{self.args.train.model_name}/summaries"
        )

        # -----------------------------------------------------------------------------#
        # ---------------------------------    train     ------------------------------#
        # -----------------------------------------------------------------------------#
        while epoch < self.args.train.num_epochs:
            step = 0
            for data in dataloader:
                data = [item.to(self.device) for item in data]
                if (
                    self.args.train.dataset == "Tomato"
                    or self.args.train.dataset == "UR5_real"
                    or self.args.train.dataset == "UR5_sim"
                ):
                    (images, prior_action, action, sentence, target) = data
                else:
                    (images, prior_action, action, sentence) = data
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(sentence)
                    text_features = text_features.clone().detach()
                    text_features = text_features.to(torch.float32)

                optimizer_.zero_grad()
                before_op_time = time.time()

                # forward
                images = rearrange(images, "bs ch h w -> bs h w ch")
                pred = self.model(images, text_features, prior_action)
                loss_1 = self.criterion(pred, action)
                loss = loss_1

                # backprop
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer_.step()
                current_lr = optimizer_.param_groups[0]["lr"]

                # verbose
                if self.global_step % self.args.train.log_freq == 0:
                    string = "[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], loss1: {:.12f}"
                    self.logger.info(
                        string.format(
                            epoch,
                            step,
                            steps_per_epoch,
                            self.global_step,
                            # current_lr,
                            loss_1,
                        )
                    )

                    if np.isnan(loss.cpu().item()):
                        self.logger.warning("NaN in loss occurred. Aborting training.")
                        return -1

                # tensorboard
                duration += time.time() - before_op_time
                if (
                    self.global_step
                    and self.global_step % self.args.train.log_freq == 0
                ):
                    self.writer.add_scalar(
                        "end_to_end_loss", loss.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "action_loss", loss_1.cpu().item(), self.global_step
                    )
                    # self.writer.add_scalar(
                    #     "state_loss", loss_2.cpu().item(), self.global_step
                    # )

                step += 1
                self.global_step += 1
                if scheduler is not None:
                    scheduler.step(self.global_step)

            # Save a model based of a chosen save frequency
            if self.global_step != 0 and (epoch + 1) % self.args.train.save_freq == 0:
                checkpoint = {
                    "global_step": self.global_step,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer_.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.args.train.log_directory,
                        self.args.train.model_name,
                        "v1.0-BCZ-LSTM-{}".format(self.global_step),
                    ),
                )

            # online evaluation
            if (
                self.args.mode.do_online_eval
                and self.global_step != 0
                and (epoch + 1) % self.args.train.eval_freq == 0
            ):
                time.sleep(0.1)
                self.model.eval()
                self.online_test()
                self.model.train()

            # Update epoch
            epoch += 1

    # -----------------------------------------------------------------------------#
    # ----------------------------------    test     ------------------------------#
    # -----------------------------------------------------------------------------#
    def online_test(self):
        if self.args.train.dataset == "OpenLid":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            test_dataset = OpenLid(self.data_path)
        elif self.args.train.dataset == "Tomato":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            test_dataset = Tomato(self.data_path)
        elif self.args.train.dataset == "Duck":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            test_dataset = Duck(self.data_path)
        elif self.args.train.dataset == "Drum":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            test_dataset = Drum(self.data_path)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8
        )

        # save test data
        save_data = {}
        traj_save = []
        gt_save = []

        step = 0
        for data in test_dataloader:
            data = [item.to(self.device) for item in data]
            if self.args.train.dataset == "Tomato":
                (images, prior_action, action, sentence, target) = data
            else:
                (images, prior_action, action, sentence) = data
            with torch.no_grad():
                text_features = self.clip_model.encode_text(sentence)
                text_features = text_features.clone().detach()
                text_features = text_features.to(torch.float32)

                # forward
                images = rearrange(images, "bs ch h w -> bs h w ch")

                traj = self.model(images, text_features, prior_action)

                traj = traj.cpu().detach().numpy()
                gt = action.cpu().detach().numpy()
                traj_save.append(traj)
                gt_save.append(gt)
                step = step + 1
                if step == 10:
                    break

        save_data["traj"] = traj_save
        save_data["gt"] = gt_save

        save_path = os.path.join(
            self.args.train.eval_summary_directory,
            self.args.train.model_name,
            "v1.0-BCZ-LSTM-{}.pkl".format(self.global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

    # def test(self):
    #     # Load the pretrained model
    #     if torch.cuda.is_available():
    #         checkpoint_1 = torch.load(self.args.test.checkpoint_path_1)
    #         self.model.load_state_dict(checkpoint_1["model"])
    #     else:
    #         checkpoint_1 = torch.load(
    #             self.args.test.checkpoint_path_1, map_location=torch.device("cpu")
    #         )
    #         self.model.load_state_dict(checkpoint_1["model"])

    #     self.model.eval()

    #     test_dataset = UR5Dataloader(
    #         self.args.test.data_path, "test", normalize="separate"
    #     )
    #     test_dataloader = torch.utils.data.DataLoader(
    #         test_dataset, batch_size=1, shuffle=False, num_workers=8
    #     )

    #     # save test data
    #     save_data = {}
    #     traj_save = []
    #     gt_save = []

    #     step = 0
    #     for data in test_dataloader:
    #         data = [item.to(self.device) for item in data]
    #         (images, prior_state, action, pre_action, sentence, _) = data
    #         with torch.no_grad():
    #             text_features = self.clip_model.encode_text(sentence)
    #             text_features = torch.tensor(text_features, dtype=torch.float32)

    #             # forward
    #             images = rearrange(images, "bs t ch h w -> bs (t ch) h w")
    #             images = rearrange(images, "bs ch h w -> bs h w ch")
    #             traj = self.model(images, text_features)

    #             pred = traj.cpu().detach().numpy()
    #             gt = action.cpu().detach().numpy()

    #             traj_save.append(pred)
    #             gt_save.append(gt)
    #             step = step + 1
    #             if step == 10:
    #                 break

    #     save_data["traj"] = traj_save
    #     save_data["gt"] = gt_save

    #     save_path = os.path.join(
    #         self.args.train.eval_summary_directory,
    #         self.args.train.model_name,
    #         "v2.0-BCZ-{}.pkl".format(1),
    #     )

    #     with open(save_path, "wb") as f:
    #         pickle.dump(save_data, f)
