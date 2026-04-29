import torch
import torch.nn as nn
import torchvision.models as models

class HandPoseRegressor(nn.Module):
    def __init__(self, num_joints=21):
        super(HandPoseRegressor, self).__init__()
        # Use a pre-trained ResNet50 as the feature extractor
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        # Output size is num_joints * 3 (x, y, z for each joint)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_joints * 3)
        )

    def forward(self, x):
        return self.backbone(x)