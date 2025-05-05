from ultralytics import YOLO

model = YOLO("AI-Football/models/best.pt")

results = model.predict("AI-Football/football.mp4", save=True, show=True)
print(results[0])

for box in results[0].boxes:
    print(box)
