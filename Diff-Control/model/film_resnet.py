# Modified ResNet with Conditional Batch Norm (CBN) layers instead of the batch norm layers
# Features from block 4 are used for the VQA task

import torch.nn as nn
import math
import json
import torch.utils.model_zoo as model_zoo
import copy
import pdb

# from film_layer import FilmResBlock as ResBlock

# from sequential_modified import Sequential

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def load_config(config_file):
    with open(config_file, "rb") as f_config:
        config_str = f_config.read()
        config = json.loads(config_str.decode("utf-8"))

    return config


# config = load_config(args.config)

# use_cbn = config["model"]["image"]["use_cbn"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


"""
This modules returns both the conv feature map and the lstm question embedding (unchanges)
since subsequent CBN layers in nn.Sequential will require both inputs
"""


class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias
        )

    def forward(self, x, lstm_emb):
        out = self.conv(x)
        return out, lstm_emb


# # https://github.com/rosinality/film-pytorch/blob/master/model.py
# class BasicBlock(nn.Module):
#     def __init__(self, filter_size):
#         super().__init__()

#         self.conv1 = nn.Conv2d(filter_size, filter_size, [1, 1], 1, 1)
#         self.conv2 = nn.Conv2d(filter_size, filter_size, [3, 3], 1, 1, bias=False)
#         self.bn = nn.BatchNorm2d(filter_size, affine=False)

#     def forward(self, input, gamma, beta):
#         out = self.conv1(input)
#         resid = F.relu(out)
#         out = self.conv2(resid)
#         out = self.bn(out)

#         gamma = gamma.unsqueeze(2).unsqueeze(3)
#         beta = beta.unsqueeze(2).unsqueeze(3)

#         out = gamma * out + beta

#         out = F.relu(out)
#         out = out + resid

#         return out


# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=True,
        downsample=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x, gamma, beta):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, lstm_size, emb_size, num_layers=18):
        self.inplanes = 64
        self.lstm_size = lstm_size
        self.emb_size = emb_size
        self.accu_layers = copy.deepcopy(layers)
        for i in range(1, len(layers)):
            self.accu_layers[i] = self.accu_layers[i] + self.accu_layers[i - 1]
        self.num_layers = sum(self.accu_layers)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.film = nn.Linear(
            emb_size,
            (64 * layers[0] + 128 * layers[1] + 256 * layers[2] + 512 * layers[3]) * 2,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        # in_channels, out_channels, stride=1, downsample=None
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.ModuleList()
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x, task_embed):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        film = self.film(task_embed)

        # x = self.layer1(x, film[: self.accu_layers[0] * 2])
        offset = 0
        for i in range(len(self.layer1)):
            base = i * 64 * 2
            x = self.layer1[i](
                x, film[:, base : base + 64], film[:, base + 64 : base + 128]
            )

        offset += (self.accu_layers[0]) * 64 * 2
        for i in range(len(self.layer2)):
            base = offset + i * 128 * 2
            x = self.layer2[i](
                x, film[:, base : base + 128], film[:, base + 128 : base + 256]
            )

        offset += (self.accu_layers[1] - self.accu_layers[0]) * 128 * 2
        for i in range(len(self.layer3)):
            base = offset + i * 256 * 2
            x = self.layer3[i](
                x, film[:, base : base + 256], film[:, base + 256 : base + 512]
            )

        offset += (self.accu_layers[2] - self.accu_layers[1]) * 256 * 2
        for i in range(len(self.layer4)):
            base = offset + i * 512 * 2
            x = self.layer4[i](
                x, film[:, base : base + 512], film[:, base + 512 : base + 1024]
            )

        x = self.avgpool(x)

        return x


def resnet18(lstm_size, emb_size, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], lstm_size, emb_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model


def resnet34(lstm_size, emb_size, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], lstm_size, emb_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
    return model
