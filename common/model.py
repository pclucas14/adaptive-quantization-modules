import pdb
import math
import utils
import torch
from torch import nn
from torch.nn import functional as F

# Generators
# ------------------------------------------------------------------------------

class ResBlock(nn.Module):
    """ Residual Block used in the Generator """

    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x

        return out


class Encoder(nn.Module):
    """ Generator Encoder Block. Arguments described in `args.py` """

    def __init__(self, in_channel, channel, downsample, n_res_blocks=1):

        super().__init__()

        # assumptions
        ks = 4
        embed_dim = channel
        num_residual_hiddens = channel

        assert downsample in (2, 4)

        if downsample == 4:
            blocks = [
                nn.Conv2d(in_channel, channel, ks, 2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, ks, 2, padding=1),
            ]

        elif downsample == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, ks, 2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_blocks):
            blocks += [ResBlock(channel, num_residual_hiddens)]

        blocks += [nn.ReLU(inplace=True)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    """ Generator Decoder Block. Arguments described in `args.py` """

    def __init__(self, channel, out_channel, upsample, n_res_blocks=1):
        super().__init__()

        # assumptions
        ks = 4
        in_channel = channel
        num_residual_hiddens = channel

        assert upsample in (2, 4)

        blocks = []

        for i in range(n_res_blocks):
            blocks += [ResBlock(channel, num_residual_hiddens)]

        blocks += [nn.ReLU(inplace=True)]

        if upsample == 4:
            blocks += [
                nn.ConvTranspose2d(channel, channel, ks, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, out_channel, ks, 2, 1),
            ]

        elif upsample == 2:
            blocks += [
                nn.ConvTranspose2d(channel, out_channel, ks, 2, 1)
            ]

        self.blocks = nn.Sequential(*blocks)


    def forward(self, x):
        return self.blocks(x)


# Classifiers
# ------------------------------------------------------------------------------
"""  The remainder of the code was taken directly from the GEM repository
     (https://github.com/facebookresearch/GradientEpisodicMemory) to ensure
     that the classifier module is the same as the reported baselines
"""

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        last_hid = nf * 8 * block.expansion

        # 128 x 128 mini-imagenet
        if input_size[-1] == 128:
            last_hid = 2560

        # 84 x 84 mini-imagenet
        elif input_size[-1] == 84:
            last_hid = 320

        self.linear = nn.Linear(last_hid, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out

def ResNet18(nclasses, nf=20, input_size=(3, 32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size)


