import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.utils import (
    ImgEmbedding,
    LangEmbedding,
    miniLangEmbedding,
    Fusion,
    OpenlidLangEmbedding,
)

# from utils import ImgEmbedding, LangEmbedding, miniLangEmbedding, Fusion
import clip
import math
import random
import pdb


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # emb = rearrange(emb, "bs k t -> (bs k) t")
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """

        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# -----------------------------------------------------------------------------#
# --------------------------------- U-net model -------------------------------#
# -----------------------------------------------------------------------------#
class SensorModel(nn.Module):
    """
    define resnet for observation
    sensor model takes a stack of RGB and output embeddings
    """

    def __init__(self, state_est, dim_x, emd_size, input_channel=3):
        super().__init__()
        self.state_est = state_est
        self.dim_x = dim_x
        self.emd_size = emd_size
        self.input_channel = 3
        self.sensor_model = ImgEmbedding(
            self.state_est, self.dim_x, self.emd_size, self.input_channel
        )

    def forward(self, obs):
        _, obs_emb = self.sensor_model(obs)
        return obs_emb


class ClipSensorModel(nn.Module):
    """
    define resnet for observation
    sensor model takes a stack of RGB and output embeddings
    """

    def __init__(self, state_est, dim_x, emd_size=256, input_channel=3):
        super().__init__()
        self.state_est = state_est
        self.dim_x = dim_x
        self.emd_size = 256
        self.input_channel = 3
        self.sensor_model = LangEmbedding(emd_size=self.emd_size)

    def forward(self, obs):
        obs_emb = self.sensor_model(obs)
        return obs_emb


class UNetwithControl(nn.Module):
    """
    input x : [bs, dim_x, window_size]
    """

    def __init__(self, dim_x, window_size, embed_size=256, state_est=1):
        super().__init__()
        self.dim_x = dim_x
        self.time_dim = window_size
        self.embed_size = embed_size

        dim_mults = (1, 2, 4, 8)
        dims = [dim_x, *map(lambda m: self.time_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        """
        positional embedding for t
        """
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(window_size),
            nn.Linear(window_size, window_size * 4),
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(window_size * 4, self.embed_size),
        )

        # """
        # auxiliary layer
        # """
        # self.auxiliary = nn.Sequential(
        #     Rearrange("batch t dim -> batch (t dim)"),
        #     nn.Linear(self.dim_x * self.time_dim, self.time_dim * 32),
        #     nn.Mish(),
        #     Rearrange("batch (t dim) -> batch t dim", t=self.time_dim),
        # )

        """
        define encoder 
        """
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        """
        define bottle neck
        """
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=self.embed_size * 2, horizon=self.time_dim
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=self.embed_size * 2, horizon=self.time_dim
        )

        """
        define decoder
        """
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        """
        define fc for language embedding
        """
        # self.lang_model = OpenlidLangEmbedding(emd_size=self.embed_size)
        self.lang_model = LangEmbedding(emd_size=self.embed_size)

        """
        fusion layer
        """
        self.fusion_layer = Fusion(emd_size=self.embed_size)

        """
        output layer
        """
        self.final_conv = nn.Sequential(
            Conv1dBlock(self.time_dim, self.time_dim, kernel_size=5),
            nn.Conv1d(self.time_dim, self.dim_x, 1),
        )
        # self.final_conv = nn.Sequential(
        #     Rearrange("batch t dim -> batch (t dim)"),
        #     nn.Linear(self.time_dim * 32, self.dim_x * self.time_dim),
        #     Rearrange("batch (t dim) -> batch t dim", t=self.time_dim),
        # )

    def forward(self, x, obs, lang, time):
        t = self.time_mlp(time)
        lang_emb, label = self.lang_model(lang)

        # add fusion layer
        emb = self.fusion_layer(obs, lang_emb)
        t = torch.cat([t, emb], axis=-1)

        # # no fusion layer
        # t = torch.cat([t, obs, lang_emb], axis=-1)

        h = []

        # x = self.auxiliary(x)

        for resnet, resnet2, downsample in self.downs:
            # print("---")
            # print(x.shape)
            x = resnet(x, t)
            # print(x.shape)
            x = resnet2(x, t)
            # print("skip", x.shape)
            h.append(x)
            x = downsample(x)
            # print(x.shape)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            # print(x.shape, "and, ", h[-1].shape)
            x = torch.cat((x, h.pop()), dim=1)
            # print(x.shape)
            x = resnet(x, t)
            # print(x.shape)
            x = resnet2(x, t)
            # print(x.shape)
            x = upsample(x)
            # print(x.shape)
        x_out = self.final_conv(x)
        return x_out


# -----------------------------------------------------------------------------#
# ------------------------------- ControlNet model ----------------------------#
# -----------------------------------------------------------------------------#
class ControlNet(nn.Module):
    """
    input x : [bs, dim_x, window_size]
    """

    def __init__(self, dim_x, window_size, embed_size=256, state_est=1):
        super().__init__()
        self.dim_x = dim_x
        self.time_dim = window_size
        self.embed_size = embed_size

        dim_mults = (1, 2, 4, 8)
        dims = [dim_x, *map(lambda m: self.time_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        """
        positional embedding for t
        """
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(window_size),
            nn.Linear(window_size, window_size * 4),
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(window_size * 4, self.embed_size),
        )

        # -------------------------------------------------------------------#
        # ---------------------------- U-Net model --------------------------#
        # -------------------------------------------------------------------#

        """
        define encoder 
        """
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        """
        define bottle neck
        """
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=self.embed_size * 2, horizon=self.time_dim
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=self.embed_size * 2, horizon=self.time_dim
        )

        """
        define decoder
        """
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        """
        define fc for language embedding
        """
        # self.lang_model = OpenlidLangEmbedding(emd_size=self.embed_size)
        self.lang_model = LangEmbedding(emd_size=self.embed_size)

        """
        fusion layer
        """
        self.fusion_layer = Fusion(emd_size=self.embed_size)

        """
        output layer
        """
        self.final_conv = nn.Sequential(
            Conv1dBlock(self.time_dim, self.time_dim, kernel_size=5),
            nn.Conv1d(self.time_dim, self.dim_x, 1),
        )

        # -------------------------------------------------------------#
        # -------------------------- ControlNet -----------------------#
        # -------------------------------------------------------------#
        """
        define encoder 
        """
        self.copy_downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.copy_downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        """
        define bottle neck
        """
        mid_dim = dims[-1]
        self.copy_mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=self.embed_size * 2, horizon=self.time_dim
        )
        self.copy_mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=self.embed_size * 2, horizon=self.time_dim
        )

        """
        define zero conv
        """
        self.mid_controlnet_block = self.zero_module(
            nn.Conv1d(mid_dim, mid_dim, kernel_size=1)
        )

        self.controlnet_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.controlnet_blocks.append(
                self.zero_module(nn.Conv1d(dim_out * 2, dim_in, 1))
            )

    def zero_module(self, module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module

    def forward(self, x, obs, lang, control_input, time):
        """
        add regular conditions
        """
        t = self.time_mlp(time)
        lang_emb, label = self.lang_model(lang)

        # add fusion layer
        emb = self.fusion_layer(obs, lang_emb)
        t = torch.cat([t, emb], axis=-1)
        input_x = x

        """
        base model encoder
        """
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        """
        controlnet encoder
        """
        x_hat = control_input + input_x
        h_hat = []
        for resnet, resnet2, downsample in self.copy_downs:
            x_hat = resnet(x_hat, t)
            x_hat = resnet2(x_hat, t)
            h_hat.append(x_hat)
            x_hat = downsample(x_hat)
        x_hat = self.copy_mid_block1(x_hat, t)
        x_hat = self.copy_mid_block2(x_hat, t)

        """
        add feature for the middle blocks
        """
        x_hat = self.mid_controlnet_block(x_hat)
        x = x + x_hat

        """
        base model decoder + controlnet feature
        """
        for resnet, resnet2, upsample in self.ups:
            # print("---")
            x = x + h_hat.pop()
            x = torch.cat((x, h.pop()), dim=1)
            # print(x.shape)
            x = resnet(x, t)
            # print(x.shape)
            x = resnet2(x, t)
            # print(x.shape)
            x = upsample(x)
            # print(x.shape)
        x_out = self.final_conv(x)
        return x_out


# -----------------------------------------------------------------------------#
# ---------------------------------    test     -------------------------------#
# -----------------------------------------------------------------------------#
# # # state = action = [bs, en, time, dim_x]
# time = 96
# dim_x = 10
# state = torch.randn(64, dim_x, time)
# t_input = torch.randn(64)
# img = torch.randn(64, 3, 224, 224)
# lang = torch.randn(64, 512)
# img_emb = torch.randn(64, 256)
# # state = rearrange(state, "bs en dim time -> (bs en) dim time")

# # sensor_Model = SensorModel(state_est=1, dim_x=10, emd_size=256, input_channel=3)
# # out = sensor_Model(img)
# # # print(out.shape)
# # lang = torch.randint(0, 1, (64,))
# # # mini_emb = miniLangEmbedding(emd_size=256, input_channel=2)
# # # out = mini_emb(lang)

# model = UNetwithControl(dim_x=dim_x, window_size=time)
# x = model(state, img_emb, lang, t_input)
# print("out shape ", x.shape)
# # print(obs_out.shape)
# # print(model.downs)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total number of parameters: ", pytorch_total_params)

# sensor = ImgEmbedding(8, 3, 32, 3)
# img = torch.randn(4 * 8, 3, 224, 224)
# out1, out2 = sensor(img)
# pdb.set_trace()
# # -----------------------------------------------------------------------------#
# model = ControlNet(dim_x=dim_x, window_size=time)
# control_input = torch.randn(64, dim_x, time)
# x= model(state, img_emb, lang, control_input, t_input)
# print(x.shape)

# a = [1, 2, 3, 4]
# b = [5, 6, 7, 8]
# for it1, it2 in zip(a, b):
#     print(it1, it2)


# # -----------------------------------------------------------------------------#

# # lang = torch.LongTensor([0])
# lang = torch.randint(0, 5, (10,))
# print(lang.shape)
# mini_emb = miniLangEmbedding(emd_size=256, input_channel=5)
# out = mini_emb(lang)
# print(out.shape)
