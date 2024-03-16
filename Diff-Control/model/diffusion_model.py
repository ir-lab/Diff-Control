import os
from collections import OrderedDict
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import random
import logging
from tqdm import tqdm
import pdb


class Diffusion:
    def __init__(
        self, noise_steps=50, beta_start=1e-4, beta_end=0.02, dim_x=3, window_size=24
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # self.beta = self.betas_for_alpha_bar(num_diffusion_timesteps=noise_steps).to(
        #     self.device
        # )
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.dim_x = dim_x
        self.time_dim = window_size

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_traj(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        Ɛ = torch.randn_like(x)
        a1 = torch.einsum("b,bij->bij", sqrt_alpha_hat, x)
        a2 = torch.einsum("b,bij->bij", sqrt_one_minus_alpha_hat, Ɛ)
        # sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
        return a1 + a2, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, images, text_features):
        logging.info(f"Sampling {n} new trajectories....")
        tmp = []
        with torch.no_grad():
            x = torch.randn((n, self.dim_x, self.time_dim)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                tmp_traj = x.cpu().detach().numpy()
                tmp.append(tmp_traj)
                predicted_noise = model(x, images, text_features, t)
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )

        return x, tmp

    def controlnet_sample(self, model, n, images, text_features, pre_action):
        logging.info(f"Sampling {n} new trajectories....")
        tmp = []
        with torch.no_grad():
            x = torch.randn((n, self.dim_x, self.time_dim)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                tmp_traj = x.cpu().detach().numpy()
                tmp.append(tmp_traj)
                predicted_noise, _ = model(x, images, text_features, pre_action, t)
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )

        return x, tmp

    def betas_for_alpha_bar(
        self,
        num_diffusion_timesteps,
        max_beta=0.99,
        alpha_transform_type="cosine",
    ):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].

        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.


        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                        prevent singularities.
            alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                        Choose from `cosine` or `exp`

        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """
        if alpha_transform_type == "cosine":

            def alpha_bar_fn(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        elif alpha_transform_type == "exp":

            def alpha_bar_fn(t):
                return math.exp(t * -12.0)

        else:
            raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)


# -----------------------------------------------------------------------------#
# ---------------------------------    test     -------------------------------#
# -----------------------------------------------------------------------------#
# diffusion = Diffusion()
# beta = diffusion.betas_for_alpha_bar(num_diffusion_timesteps=50)
# beta_2 = diffusion.prepare_noise_schedule()
# print(beta)
# print("---")
# print(beta_2)

# time = 32
# dim_x = 3
# state = torch.randn(32, dim_x, time)
# t_input = torch.randn(32, 1)
# img = torch.randn(32 * 8, 3, 224, 224)
# lang = torch.randn(32, 512)

# beta = torch.linspace(1e-4, 0.02, 50)
# alpha = 1.0 - beta
# alpha_hat = torch.cumprod(alpha, dim=0)
# print(alpha_hat.shape)

# import pdb

# time = 32
# dim_x = 3
# state = torch.randn(32, dim_x, time)

# diffusion = Diffusion()
# t = diffusion.sample_timesteps(32)
# x_t, noise = diffusion.noise_traj(state, t)

# # # pdb.set_trace()
# tensor1 = torch.tensor([2, 2])
# tnesor2 = torch.randn(2, 3, 3)
# # a = torch.mul(tensor1, tnesor2, axis=0)
# a = torch.einsum("b,bij->bij", tensor1, tnesor2)
# print(a)
# print("--")
# print(tnesor2)
