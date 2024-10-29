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


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = True,
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        # pdb.set_trace()
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        d_x, atten = self.attention(self.ln_1(x))
        x = x + d_x
        x = x + self.mlp(self.ln_2(x))
        return x, atten


class AttnModule(nn.Module):
    """
    this module assumes input x = [bs, feature, time_dim]
    """

    def __init__(self, feat, time_dim):
        super().__init__()
        self.feat = feat
        self.time_dim = time_dim
        self.attention_layer_1 = ResidualAttentionBlock(
            d_model=self.feat, n_head=8, attn_mask=None
        )
        self.positional_encoding_layer = PositionalEncoder(
            d_model=self.feat, dropout=0.1, max_seq_len=2000, batch_first=True
        )

    def forward(self, x):
        x = rearrange(x, "bs f t -> bs t f")
        x = self.positional_encoding_layer(x)
        x, _ = self.attention_layer_1(x)
        x = rearrange(x, "bs t f -> bs f t")
        return x


class Downsample1d(nn.Module):
    def __init__(self, dim, linear_dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
        self.linear_dim = linear_dim
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample1d(nn.Module):
    def __init__(self, dim, linear_dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
        self.linear_dim = linear_dim
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        return x


class LinearProjection(nn.Module):
    def __init__(self, feat, dim):
        super().__init__()
        self.feat = feat
        self.dim = dim
        self.linear = nn.Linear(self.feat * dim, self.feat * dim)

    def forward(self, x):
        dim = x.shape[-1]
        x = rearrange(x, "bs f dim -> bs (f dim)")
        x = self.linear(x)
        x = F.relu(x)
        x = rearrange(x, "bs (f dim) -> bs f dim", f=self.feat)
        return x


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
    def __init__(
        self,
        inp_channels,
        out_channels,
        embed_dim,
        horizon,
        kernel_size=5,
        linear_dim=24,
    ):
        super().__init__()
        self.linear_dim = linear_dim

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
        self.out_channels = out_channels

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """

        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# -----------------------------------------------------------------------------#
# --------------------------------- U-net model -------------------------------#
# -----------------------------------------------------------------------------#


class StatefulUNet(nn.Module):
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

        seq_divide = (2, 4, 8)
        linear_out = [self.time_dim, *map(lambda m: int(self.time_dim / m), seq_divide)]
        reverse_linear_out = list(reversed(linear_out))

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
                            linear_dim=linear_out[ind],
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                            linear_dim=linear_out[ind],
                        ),
                        (
                            Downsample1d(dim_out, linear_dim=linear_out[ind + 1])
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        """
        define bottle neck
        """
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=self.embed_size * 2,
            horizon=self.time_dim,
            linear_dim=linear_out[-1],
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=self.embed_size * 2,
            horizon=self.time_dim,
            linear_dim=linear_out[-1],
        )
        self.addition_module = AttnModule(feat=mid_dim, time_dim=linear_out[-1])

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
                            linear_dim=reverse_linear_out[ind],
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                            linear_dim=reverse_linear_out[ind],
                        ),
                        (
                            Upsample1d(dim_in, linear_dim=reverse_linear_out[ind + 1])
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        """
        define fc for language embedding
        """
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

    def forward(self, x, obs, lang, time):
        t = self.time_mlp(time)
        lang_emb, label = self.lang_model(lang)

        # add fusion layer
        emb = self.fusion_layer(obs, lang_emb)
        t = torch.cat([t, emb], axis=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            # print("---")
            x = resnet(x, t)
            # print(x.shape)
            x = resnet2(x, t)
            # print(x.shape)
            # print("skip", x.shape)
            h.append(x)
            x = downsample(x)
            # print(x.shape)

        # print("===")

        x = self.mid_block1(x, t)
        # print(x.shape)
        x = self.mid_block2(x, t)
        x = self.addition_module(x)

        for resnet, resnet2, upsample in self.ups:
            # print("---")
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
class StatefulControlNet(nn.Module):
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

        seq_divide = (2, 4, 8)
        linear_out = [self.time_dim, *map(lambda m: int(self.time_dim / m), seq_divide)]
        reverse_linear_out = list(reversed(linear_out))

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
                            linear_dim=linear_out[ind],
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                            linear_dim=linear_out[ind],
                        ),
                        (
                            Downsample1d(dim_out, linear_dim=linear_out[ind + 1])
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        """
        define bottle neck
        """
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=self.embed_size * 2,
            horizon=self.time_dim,
            linear_dim=linear_out[-1],
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=self.embed_size * 2,
            horizon=self.time_dim,
            linear_dim=linear_out[-1],
        )
        self.addition_module = AttnModule(feat=mid_dim, time_dim=linear_out[-1])

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
                            linear_dim=reverse_linear_out[ind],
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                            linear_dim=reverse_linear_out[ind],
                        ),
                        (
                            Upsample1d(dim_in, linear_dim=reverse_linear_out[ind + 1])
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        """
        define fc for language embedding
        """
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
                            linear_dim=linear_out[ind],
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=self.embed_size * 2,
                            horizon=self.time_dim,
                            linear_dim=linear_out[ind],
                        ),
                        (
                            Downsample1d(
                                dim_out, linear_dim=reverse_linear_out[ind + 1]
                            )
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        """
        define bottle neck
        """
        mid_dim = dims[-1]
        self.copy_mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=self.embed_size * 2,
            horizon=self.time_dim,
            linear_dim=linear_out[-1],
        )
        self.copy_mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=self.embed_size * 2,
            horizon=self.time_dim,
            linear_dim=linear_out[-1],
        )
        self.copy_addition_module = AttnModule(feat=mid_dim, time_dim=linear_out[-1])

        """
        define zero conv
        """
        self.mid_controlnet_block = self.zero_module(
            nn.Conv1d(mid_dim, mid_dim, kernel_size=1)
        )

        self.controlnet_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
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
        x = self.addition_module(x)

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
        x_hat = self.copy_addition_module(x_hat)

        """
        add feature for the middle blocks
        """
        x_hat = self.mid_controlnet_block(x_hat)
        x = x + x_hat

        """
        base model decoder + controlnet feature
        """
        for ind, (resnet, resnet2, upsample) in enumerate(self.ups):
            # print("---")
            x = x + h_hat.pop()
            x = torch.cat((x, h.pop()), dim=1)
            x_hat = self.controlnet_blocks[ind](x)
            # print(x.shape)
            x = resnet(x, t)
            # print(x.shape)
            x = resnet2(x, t)
            x = x + x_hat
            # print(x.shape)
            x = upsample(x)

            # print(x.shape)
        x_out = self.final_conv(x)
        return x_out


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # state = action = [bs, en, time, dim_x]
# time = 24
# dim_x = 10
# state = torch.randn(64, dim_x, time).to(device)
# t_input = torch.randn(64).to(device)
# img = torch.randn(64, 3, 224, 224).to(device)
# lang = torch.randn(64, 512).to(device)
# img_emb = torch.randn(64, 256).to(device)

# # state = rearrange(state, "bs en dim time -> (bs en) dim time")

# # sensor_Model = SensorModel(state_est=1, dim_x=10, emd_size=256, input_channel=3)
# # out = sensor_Model(img)
# # # print(out.shape)
# # lang = torch.randint(0, 1, (64,))
# # # mini_emb = miniLangEmbedding(emd_size=256, input_channel=2)
# # # out = mini_emb(lang)

# model = StatefulControlNet(dim_x=dim_x, window_size=time)
# model.cuda()
# x = model(state, img_emb, lang, state, t_input)
# print("out shape ", x.shape)
