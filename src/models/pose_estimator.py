import torch
import torch.nn as nn
from .backbone import build_backbone

NUM_JOINTS = 21


class HandPoseEstimator(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_JOINTS * 3),
        )

    def forward(self, x):
        features = self.backbone(x)
        joints = self.head(features)
        return joints.view(-1, NUM_JOINTS, 3)
