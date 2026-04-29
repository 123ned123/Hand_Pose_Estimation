import torch
from PIL import Image
from src.models.pose_estimator import HandPoseEstimator
from src.data.preprocess import val_transform


def load_model(weights_path, backbone="resnet50", device="cpu"):
    model = HandPoseEstimator(backbone_name=backbone)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)


def predict(model, image_path, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        joints = model(tensor)  # (1, 21, 3)
    return joints.squeeze(0).cpu().numpy()
