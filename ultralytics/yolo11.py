from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# run inference on these videos

#results = model("ultralytics/766005_Zoom Stylish Berlin Street Vogue_By_Omri_Ohana_Artlist_HD.mp4", save=True)

result = model("ultralytics/6103843_Recreation Game Fun Streetball_By_Raven_Production_Artlist_HD.mp4", save=True)