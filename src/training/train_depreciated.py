import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import HandPoseDataset
from src.data.preprocess import train_transform, val_transform
from src.models.pose_estimator import HandPoseEstimator
from src.training.loss import JointMSELoss, mpjpe


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = HandPoseDataset(cfg["data_dir"], split="train", transform=train_transform)
    val_ds = HandPoseDataset(cfg["data_dir"], split="val", transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    model = HandPoseEstimator(backbone_name=cfg["backbone"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = JointMSELoss()

    for epoch in range(cfg["epochs"]):
        model.train()
        for images, joints in train_loader:
            images, joints = images.to(device), joints.to(device)
            pred = model(images)
            loss = criterion(pred, joints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_errors = []
        with torch.no_grad():
            for images, joints in val_loader:
                images, joints = images.to(device), joints.to(device)
                pred = model(images)
                val_errors.append(mpjpe(pred, joints).item())

        print(f"Epoch {epoch+1}/{cfg['epochs']}  MPJPE: {sum(val_errors)/len(val_errors):.4f}")

    torch.save(model.state_dict(), "checkpoints/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
