import cv2
import wandb

from dataset.square_ph import SquarePhDataset
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from model import UNetwithControl, SensorModel, StatefulUNet
from dataset.lid_pick_and_place import *
#from config.tomato_pick_and_place import *
from dataset.pick_duck import *
from dataset.drum_hit import *
from optimizer import build_optimizer
from optimizer import build_lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import pickle

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel


class Engine(BaseImagePolicy):
    def __init__(self, args, logger, diff_pol_dataset=None, env_runner=None):
        super().__init__()
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
        self.env_runner = env_runner

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
        elif self.args.train.dataset == "SquarePhDataset":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path

            self.dataset = SquarePhDataset(diff_pol_dataset)
            self.test_dataset = self.dataset.get_validation_dataset()

        if self.args.train.dataset == "Drum":
            self.model = StatefulUNet(dim_x=self.dim_x, window_size=self.win_size)
        else:
            self.model = UNetwithControl(dim_x=self.dim_x, window_size=self.win_size)
        self.sensor_model = SensorModel(
            state_est=1,
            dim_x=self.dim_x,
            emd_size=256,
            input_channel=self.channel_img_1,
        )
        # -----------------------------------------------------------------------------#
        # ---------------------------    get model ready     --------------------------#
        # -----------------------------------------------------------------------------#
        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda(self.device)
            self.sensor_model.cuda(self.device)

        # -----------------------------------------------------------------------------#
        # ------------------------------- use pre-trained?  ---------------------------#
        # -----------------------------------------------------------------------------#
        # if torch.cuda.is_available():
        #     checkpoint_1 = torch.load(self.args.test.checkpoint_path_1)
        #     self.model.load_state_dict(checkpoint_1["model"])
        #     checkpoint_2 = torch.load(self.args.test.checkpoint_path_2)
        #     self.sensor_model.load_state_dict(checkpoint_2["model"])
        # else:
        #     checkpoint_1 = torch.load(
        #         self.args.test.checkpoint_path_1, map_location=torch.device("cpu")
        #     )
        #     self.model.load_state_dict(checkpoint_1["model"])
        #     checkpoint_2 = torch.load(
        #         self.args.test.checkpoint_path_2, map_location=torch.device("cpu")
        #     )
        #     self.sensor_model.load_state_dict(checkpoint_2["model"])

        # -----------------------------------------------------------------------------#
        # ------------------------------- ema model  ----------------------------------#
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
        self.ema = EMAModel(parameters=self.model.parameters(), power=0.75)
        self.clip_model, preprocess = clip.load("ViT-B/32", device=self.device)

    def train(self):
        # -----------------------------------------------------------------------------#
        # ---------------------------------    setup     ------------------------------#
        # -----------------------------------------------------------------------------#
        self.criterion = nn.MSELoss()  # nn.MSELoss() or nn.L1Loss()
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

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

        # Create optimizer
        optimizer_ = build_optimizer(
            [self.model, self.sensor_model],
            # self.model,
            self.args.network.name,
            self.args.optim.optim,
            self.args.train.learning_rate,
            self.args.train.weight_decay,
            self.args.train.adam_eps,
        )
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
        previous_val_loss = float("inf")
        all_val_losses = []
        while epoch < self.args.train.num_epochs:
            step = 0
            for data in dataloader:
                data = [item.to(self.device) for item in data]
                (images, prior_action, action, sentence) = data
                optimizer_.zero_grad()
                before_op_time = time.time()

                with torch.no_grad():
                    text_features = self.clip_model.encode_text(sentence)
                    text_features = text_features.clone().detach()
                    text_features = text_features.to(torch.float32)

                # sample noise to add to actions
                noise = torch.randn(action.shape, device=self.device)
                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (action.shape[0],),
                    device=self.device,
                ).long()
                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

                # forward
                img_emb = self.sensor_model(images)
                predicted_noise = self.model(
                    noisy_actions, img_emb, text_features, timesteps
                )
                loss_1 = self.criterion(noise, predicted_noise)
                loss = loss_1

                # backprop
                loss.backward()
                optimizer_.step()
                self.ema.step(self.model.parameters())
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
                            # loss_2,
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
                        "noise_loss", loss_1.cpu().item(), self.global_step
                    )

                step += 1
                self.global_step += 1
                if scheduler is not None:
                    scheduler.step(self.global_step)

            # Save a model based of a chosen save frequency
            #if self.global_step != 0 and (epoch + 1) % self.args.train.save_freq == 0:
            if self.global_step != 0:
                self.ema_nets = self.model
                checkpoint = {
                    "global_step": self.global_step,
                    "model": self.ema_nets.state_dict(),
                    "optimizer": optimizer_.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.args.train.log_directory,
                        self.args.train.model_name,
                        "v1.0-ema-model-{}".format(self.global_step),
                    ),
                )
                checkpoint = {
                    "global_step": self.global_step,
                    "model": self.sensor_model.state_dict(),
                    "optimizer": optimizer_.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.args.train.log_directory,
                        self.args.train.model_name,
                        "v1.0-sensor_model-{}".format(self.global_step),
                    ),
                )

            if self.global_step != 0:
                runner_log = self.env_runner.run(self)
                videos = {k: v for k, v in runner_log.items() if isinstance(v, wandb.Video)}
                for k, v in videos:
                    cap = cv2.VideoCapture(v._path)
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from OpenCV BGR to RGB
                        frames.append(frame)
                    cap.release()
                    video_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
                    # Add batch dimension (1, T, C, H, W)
                    video_tensor = video_tensor.unsqueeze(0)
                    self.writer.add_video(k, video_tensor, self.global_step, fps=10)
            # online evaluation
            if (
                self.args.mode.do_online_eval
                and self.global_step != 0
                #and (epoch + 1) % self.args.train.eval_freq == 0
            ):
                time.sleep(0.1)
                self.ema_nets = self.model
                self.ema_nets.eval()
                self.sensor_model.eval()
                self.online_test()
                ###############################################################
                val_loss = None
                with torch.no_grad():
                    val_losses = list()
                    for data in test_dataloader:
                        data = [item.to(self.device) for item in data]
                        (images, prior_action, action, sentence) = data
                        optimizer_.zero_grad()

                        text_features = self.clip_model.encode_text(sentence)
                        text_features = text_features.clone().detach()
                        text_features = text_features.to(torch.float32)

                        # sample noise to add to actions
                        noise = torch.randn(action.shape, device=self.device)
                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (action.shape[0],),
                            device=self.device,
                        ).long()
                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

                        # forward
                        img_emb = self.sensor_model(images)
                        predicted_noise = self.model(
                            noisy_actions, img_emb, text_features, timesteps
                        )
                        loss = self.criterion(noise, predicted_noise)
                        val_losses.append(loss.cpu().item())
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        all_val_losses.append((epoch, val_loss))

                        ###############################################################
                        self.ema_nets.train()
                        self.sensor_model.train()
                        self.ema.copy_to(self.ema_nets.parameters())

                if val_loss > previous_val_loss:
                    return

                previous_val_loss = val_loss

            # Update epoch
            epoch += 1

    def set_normalizer(self, normalizer):
        pass

    def predict_action(self, obs_dict):
        images = obs_dict["image"]
        sentence = obs_dict["sentence"].to(self.device).unsqueeze(0)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(sentence)
            text_features = text_features.clone().detach()
            text_features = text_features.to(torch.float32)

            img_emb = self.sensor_model(images)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, self.dim_x, self.win_size)).to(
                self.device
            )

            # init scheduler
            self.noise_scheduler.set_timesteps(50)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                t = torch.stack([k]).to(self.device)
                predicted_noise = self.ema_nets(
                    noisy_action, img_emb, text_features, t
                )

                # inverse diffusion step (remove noise)
                noisy_action = self.noise_scheduler.step(
                    model_output=predicted_noise, timestep=k, sample=noisy_action
                ).prev_sample
        return noisy_action

    # -----------------------------------------------------------------------------#
    # ---------------------------------     test     ------------------------------#
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
        elif self.args.train.dataset == "SquarePhDataset":
            if self.mode == "train":
                self.data_path = self.args.train.data_path
            else:
                self.data_path = self.args.test.data_path
            test_dataset = self.dataset.get_validation_dataset()

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=True, num_workers=8
        )

        # save test data
        save_data = {}
        traj_save = []
        gt_save = []

        step = 0
        print("------------ testing ------------")
        for data in test_dataloader:
            data = [item.to(self.device) for item in data]
            (images, prior_action, action, sentence) = data
            with torch.no_grad():
                text_features = self.clip_model.encode_text(sentence)
                text_features = text_features.clone().detach()
                text_features = text_features.to(torch.float32)
                # -------------------------------------------------
                # text_features = sentence
                img_emb = self.sensor_model(images)

                # initialize action from Guassian noise
                noisy_action = torch.randn((1, self.dim_x, self.win_size)).to(
                    self.device
                )

                # init scheduler
                self.noise_scheduler.set_timesteps(50)

                traj_stack = []
                for k in self.noise_scheduler.timesteps:
                    # predict noise
                    t = torch.stack([k]).to(self.device)
                    predicted_noise = self.ema_nets(
                        noisy_action, img_emb, text_features, t
                    )

                    # inverse diffusion step (remove noise)
                    noisy_action = self.noise_scheduler.step(
                        model_output=predicted_noise, timestep=k, sample=noisy_action
                    ).prev_sample

                    tmp_traj = noisy_action.cpu().detach().numpy()
                    traj_stack.append(tmp_traj)

                traj = noisy_action
                traj = traj.cpu().detach().numpy()
                gt = action.cpu().detach().numpy()
                traj_save.append(traj)
                gt_save.append(gt)
                step = step + 1
                if step == 10:
                    break

        save_data["traj"] = traj_save
        save_data["gt"] = gt_save
        save_data["traj_stack"] = traj_stack

        save_path = os.path.join(
            self.args.train.eval_summary_directory,
            self.args.train.model_name,
            "v1.0-result-{}.pkl".format(self.global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

    def test(self):
        # Load the pretrained model
        if torch.cuda.is_available():
            checkpoint_1 = torch.load(self.args.test.checkpoint_path_1)
            self.model.load_state_dict(checkpoint_1["model"])
            checkpoint_2 = torch.load(self.args.test.checkpoint_path_2)
            self.sensor_model.load_state_dict(checkpoint_2["model"])
        else:
            checkpoint_1 = torch.load(
                self.args.test.checkpoint_path_1, map_location=torch.device("cpu")
            )
            checkpoint_2 = torch.load(
                self.args.test.checkpoint_path_2, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint_1["model"])
            self.sensor_model.load_state_dict(checkpoint_2["model"])

        self.model.eval()
        self.sensor_model.eval()
