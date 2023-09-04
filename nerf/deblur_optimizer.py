import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .spline import SE3_to_se3_N


class DeblurOptimizer(nn.Module):
    def __init__(self,
                 poses=None, 
                 **kwargs,
                 ):
        super().__init__()
        poses_start_se3 = SE3_to_se3_N(poses[:, :3, :4])
        poses_end_se3 = poses_start_se3
        low, high = 0.0001, 0.005
        rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6, device=poses.device) + low
        # print(rand[:3])
        poses_start_se3 = poses_start_se3 + rand
        start_end = torch.cat([poses_start_se3, poses_end_se3], -1)
        # N = len(self.train_dataset.poses)
        self.se3 = nn.Parameter(start_end)

    def forward(self, indices):
        return self.se3[indices, :]

    # optimizer utils
    def get_params(self, lr):
        params = {'params': self.se3, 'lr': lr}
        return params
