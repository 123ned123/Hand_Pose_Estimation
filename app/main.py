# two-stage inference: receiving an image,
# running YOLO to detect/crop the hand, and running the 3D PyTorch model on the crop.

import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from src.models.pose_estimator import HandPoseRegressor

# Instantiate the FastAPI application
app = FastAPI(title="Hand Pose Estimation API")

# Initialize models globally so they load once on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load YOLOv8 for 2D detection
yolo_model = YOLO("yolov8n.pt") 

# 2. Load custom 3D Pose Estimator
pose_model = HandPoseRegressor(num_joints=21)
# pose_model.load_state_dict(torch.load("hand_pose_model.pth", map_location=device))
pose_model.to(device)
pose_model.eval()

# Transformation pipeline for the 3D model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict_hand_pose(file: UploadFile = File(...)):
    # Read the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Step 1: Detect hand with YOLO
    results = yolo_model(image)
    
    # Check if a hand (or person) was detected
    if len(results[0].boxes) == 0:
        return {"error": "No hands detected in the image."}
    
    # Get bounding box coordinates for the first detected hand
    # Assuming class 0 is person/hand depending on your YOLO fine-tuning
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    
    # Step 2: Crop the hand from the image
    cropped_hand = image.crop((x1, y1, x2, y2))
    
    # Step 3: Prepare the crop for the PyTorch 3D model
    input_tensor = preprocess(cropped_hand).unsqueeze(0).to(device)
    
    # Step 4: Estimate 3D pose
    with torch.no_grad():
        pred_keypoints = pose_model(input_tensor)
    
    # Reshape the output from (1, 63) to (21, 3)
    keypoints_3d = pred_keypoints.view(21, 3).cpu().numpy().tolist()
    
    return {
        "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "keypoints_3d": keypoints_3d
    }

# To run this server, use the command: uvicorn app.main:app --reload