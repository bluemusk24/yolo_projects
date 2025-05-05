from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Predict with the model
result = model.track(source="ultralytics/rugby.mp4", save=True, show=True)