import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from typing import List, Union
import copy
import math
import random
import pdb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_channel):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=64 // 16, num_channels=64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                # nn.BatchNorm2d(planes),
                nn.GroupNorm(num_groups=planes // 16, num_channels=planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)  # -> 64, 56 x 56
        x = self.layer1(x)  # -> 128, 28 x 28
        x = self.layer2(x)  # -> 256, 14 x 14
        x = self.layer3(x)  # -> 512, 7 x 7
        return x


class ImgEmbedding(nn.Module):
    """
    latent sensor model takes the inputs stacks of images t-n:t-1
    and generate the latent state representations for the transformer
    process model, here we use resnet34 as the basic encoder to project
    down the vision inputs

    images -> [batch, time, channels, height, width]
    out -> [batch, time, latent_dim_x]
    """

    def __init__(self, time_step, dim_x, emd_size, input_channel):
        super(ImgEmbedding, self).__init__()
        self.time_step = time_step
        self.dim_x = dim_x
        layers = [2, 2, 2, 2]
        self.model = ResNet(ResidualBlock, layers, input_channel)
        self.linear1 = torch.nn.Linear(512 * 7 * 7, 2048)
        self.linear2 = torch.nn.Linear(2048, 1024)
        self.bayes1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.bayes2 = torch.nn.Linear(in_features=512, out_features=emd_size)
        self.bayes3 = torch.nn.Linear(in_features=emd_size, out_features=dim_x)

    def forward(self, images):
        # images = rearrange(images, "bs t ch h w -> (bs t) ch h w")
        x = self.model(images)
        x = rearrange(x, "bs feat h w -> bs (feat h w)")
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.bayes1(x)
        x = F.relu(x)
        x = self.bayes2(x)
        embedding = x
        x = F.relu(x)
        x = self.bayes3(x)
        embedding = rearrange(embedding, "(bs en) dim -> bs en dim", en=self.time_step)
        embedding = torch.sum(embedding, dim=1)
        x = rearrange(x, "(bs en) dim -> bs en dim", en=self.time_step)
        out = rearrange(x, "bs time dim -> bs dim time")
        return out, embedding


class LangEmbedding(nn.Module):
    """
    project the embedded language into the dimension needed

    lang -> [batch, 512]
    out -> [batch, latent_dim_x]
    """

    def __init__(self, emd_size, input_channel=512):
        super(LangEmbedding, self).__init__()
        self.emd_size = emd_size
        self.linear1 = torch.nn.Linear(in_features=input_channel, out_features=emd_size)
        self.linear2 = torch.nn.Linear(in_features=emd_size, out_features=emd_size)
        self.cls = torch.nn.Linear(in_features=emd_size, out_features=3)

    def forward(self, lang):
        x = self.linear1(lang)
        x = F.relu(x)
        x = self.linear2(x)
        feat = F.relu(x)
        label = self.cls(feat)
        return x, label


class OpenlidLangEmbedding(nn.Module):
    """
    project the embedded language into the dimension needed

    lang -> [batch, 512]
    out -> [batch, latent_dim_x]
    """

    def __init__(self, emd_size, input_channel=512):
        super(OpenlidLangEmbedding, self).__init__()
        self.emd_size = emd_size
        self.linear1 = torch.nn.Linear(in_features=input_channel, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=emd_size)

    def forward(self, lang):
        x = self.linear1(lang)
        x = F.relu(x)
        x = self.linear2(x)
        return x, x


class miniLangEmbedding(nn.Module):
    """
    project the embedded language into the dimension needed

    lang -> [batch, 1]
    out -> [batch, latent_dim_x]
    """

    def __init__(self, emd_size, input_channel=2):
        super(miniLangEmbedding, self).__init__()
        self.emd_size = emd_size
        self.emd_layer = torch.nn.Embedding(input_channel, self.emd_size)

    def forward(self, lang):
        x = self.emd_layer(lang)
        return x


class Fusion(nn.Module):
    def __init__(self, emd_size):
        super(Fusion, self).__init__()
        self.input_size = emd_size * 2
        self.linear1 = torch.nn.Linear(
            in_features=self.input_size, out_features=emd_size
        )
        self.linear2 = torch.nn.Linear(in_features=emd_size, out_features=emd_size)
        self.linear_add1 = torch.nn.Linear(in_features=emd_size, out_features=emd_size)

    def forward(self, obs, lang):
        x = self.linear_add1(lang)
        x = F.relu(x)
        x = torch.cat([obs, x], axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
