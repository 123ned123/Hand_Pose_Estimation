import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloader
from src.models.pose_estimator import HandPoseRegressor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize model, loss function, and optimizer
    model = HandPoseRegressor(num_joints=21).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load data
    train_loader = get_dataloader('data/processed/train_annotations.json', 'data/processed/images/', batch_size=32)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, keypoints in train_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, keypoints)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.6f}")

    # Save the trained model weights
    torch.save(model.state_dict(), "hand_pose_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()