from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.predict("AI-Football/football.mp4", save=True, show=True)
print(results[0])

for box in results[0].boxes:
    print(box)


# get the data in tensor array
#for result in results:
#    print(result.boxes.data)  # torch.Tensor array
    # reference https://docs.ultralytics.com/modes/predict/#working-with-results)

    # get bounding boxes
#    print(result.boxes.xywh)  # torch.Tensor array
    # reference https://docs.ultralytics.com/modes/predict/#boxes)

    # get confidence score
#    print(result.boxes.conf)