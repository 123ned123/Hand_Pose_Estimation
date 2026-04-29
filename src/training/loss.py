import torch
import torch.nn as nn


class JointMSELoss(nn.Module):
    def forward(self, pred, target):
        return nn.functional.mse_loss(pred, target)


def mpjpe(pred, target):
    """Mean Per Joint Position Error in same units as target coordinates."""
    return torch.mean(torch.norm(pred - target, dim=-1))
