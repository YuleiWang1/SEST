import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class CANet(nn.Module):
    def __init__(self, num_channel, num_feature):
        super(CANet, self).__init__()
        self.num_channel = num_channel
        self.num_feature = num_feature

        layers = list()
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(in_channels=num_channel, out_channels=num_feature, kernel_size=1, stride=1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=num_feature, out_channels=num_channel, kernel_size=1, stride=1))
        layers.append(nn.Sigmoid())

        self.CA_block = nn.Sequential(*layers)

    def forward(self, inputs):
        CA_atten = self.CA_block(inputs)

        return CA_atten


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # return self.sigmoid(out)
        out = x * self.sigmoid(out)
        return out


class SE(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, n, c = x.size()
        t = x.transpose(1, 2)
        y = self.avg_pool(t).squeeze(-1)
        y = self.fc(y)
        out = x * y.unsqueeze(1)
        return out


class SE2d(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, h, w, c = x.size()
        t = x.view(b, c, h, w)
        y = self.avg_pool(t).squeeze(2).squeeze(2)
        y = self.fc(y)
        out = x * y.unsqueeze(1).unsqueeze(1)
        return out


class CAMA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CAMA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, n, c = x.size()
        t = x.transpose(1, 2)
        avg_out = self.avg_pool(t).squeeze(-1)
        max_out = self.max_pool(t).squeeze(-1)
        y = self.sigmoid(self.fc(avg_out) + self.fc(max_out))
        out = x * y.unsqueeze(1)
        return out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=8, memory_blocks=128):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.subnet = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
            )
        self.upnet = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.mb = torch.nn.Parameter(torch.randn(channel // reduction, memory_blocks), requires_grad=True)
        self.low_dim = channel // reduction

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)

        # b, n, c = x.shape
        b, h, w, c = x.shape
        # t = x.transpose(1, 2)
        t = x.view(b, c, h, w)
        # y = self.avg_pool(t).squeeze(-1)
        # y = torch.squeeze(self.avg_pool(t))
        y = self.avg_pool(t).squeeze(2).squeeze(2)

        low_rank_f = self.subnet(y).unsqueeze(2)

        # mbg = self.mb.unsqueeze(0).repeat(b, 1, 1).clamp_(0, 1)
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg

        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)
        y1 = f_dic_c @ mbg.transpose(1, 2)

        # y2 = self.upnet(y1)
        y2 = self.upnet(y1).unsqueeze(2)
        out = x*y2

        return out

class SELayer1d(nn.Module):
    def __init__(self, channel, reduction=8, memory_blocks=128):
        super(SELayer1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.subnet = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
            )
        self.upnet = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.mb = torch.nn.Parameter(torch.randn(channel // reduction, memory_blocks), requires_grad=True)
        self.low_dim = channel // reduction

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)

        b, n, c = x.shape
        # b, h, w, c = x.shape
        t = x.transpose(1, 2)
        # t = x.view(b, c, h, w)
        y = self.avg_pool(t).squeeze(-1)
        # y = self.avg_pool(t).squeeze(2).squeeze(2)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1).clamp_(0, 1)
        # mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)  # 去掉钳制函数
        f1 = (low_rank_f.transpose(1, 2)) @ mbg

        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)
        y1 = f_dic_c @ mbg.transpose(1, 2)

        y2 = self.upnet(y1)
        # y2 = self.upnet(y1).unsqueeze(2)
        out = x*y2

        return out


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=8, memory_blocks=128):
        super(CAB, self).__init__()
        self.num_feat = num_feat
        self.cab = nn.Sequential(
            nn.Linear(num_feat, num_feat // compress_ratio),
            nn.GELU(),
            nn.Linear(num_feat // compress_ratio, num_feat),
            SELayer(num_feat, squeeze_factor, memory_blocks)
            # SELayer1d(num_feat, squeeze_factor, memory_blocks)
        )

    def forward(self, x):
        return self.cab(x)


class CABlock1d(nn.Module):
    def __init__(self, channel, reduction=8, memory_blocks=128):
        super(CABlock1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.subnet = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.upnet = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.mb = torch.nn.Parameter(torch.randn(channel // reduction, memory_blocks), requires_grad=True)
        self.low_dim = channel // reduction

    def forward(self, x):
        b, n, c = x.shape
        t = x.transpose(1, 2)

        y_avg = self.avg_pool(t).squeeze(-1)
        y_max = self.max_pool(t).squeeze(-1)

        low_rank_f_avg = self.subnet(y_avg).unsqueeze(2)
        low_rank_f_max = self.subnet(y_max).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)

        f1 = (low_rank_f_avg.transpose(1, 2)) @ mbg + (low_rank_f_max.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)

        y1 = f_dic_c @ mbg.transpose(1, 2)

        y2 = self.upnet(y1)

        out = x*y2

        return out

if __name__ == '__main__':

    # model = CANet(31, 5)
    # model = ChannelAttention(48)
    # model = SELayer(48)
    # model = SELayer1d(48)
    # model = SE(48)
    model = SE2d(48)
    # model = CAMA(48)
    # model = CABlock1d(48)
    # model = CABlock(48)
    # input = torch.randn((192, 64, 48))
    input = torch.randn((3, 64, 64, 48))
    # ca_atten = model(input)
    ca_atten = model(input)
    print(np.shape(ca_atten))
