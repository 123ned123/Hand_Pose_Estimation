import torch.nn as nn
from torchvision import models


def build_backbone(name="resnet50", pretrained=True):
    if name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if name == "efficientnet_b3":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unknown backbone: {name}")
