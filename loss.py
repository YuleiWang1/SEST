import torch
# import tensorflow as tf
from torch import nn


class spectral_loss(torch.nn.Module):
    def __init__(self):
        super(spectral_loss, self).__init__()
        pass

    def forward(self, y, gt):
        gen_dis = torch.sqrt(torch.sum(y * y, dim=1))
        target_dis = torch.sqrt(torch.sum(gt * gt, dim=1))
        temp = torch.sum(gt * y, dim=1)
        # spectral_loss = torch.mean(math.acos(temp / (gen_dis * target_dis)))
        spectral_loss = torch.mean(torch.acos(temp / (gen_dis * target_dis)))

        return spectral_loss


class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]



if __name__ == '__main__':
    spectral_loss_criterion = spectral_loss()
    content_loss_criterion = nn.MSELoss()
    tv_loss = HybridLoss()
    y = torch.randn((3, 31, 64, 64))
    # y = torch.ones((8, 64, 64, 31))
    gt = torch.randn((3, 31, 64, 64))
    # gt = torch.ones((8, 64, 64, 31))
    loss_s = spectral_loss_criterion(y, gt)
    loss_c = content_loss_criterion(y, gt)
    loss_h = tv_loss(y, gt)
    loss = loss_c + loss_s
    print(loss_c)
    print(loss)
    print(loss_h)

