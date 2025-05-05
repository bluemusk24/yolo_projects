from ultralytics import YOLO, ASSETS

model = YOLO("yolo11s.pt", task="detect")
results = model(source="ultralytics/bus.jpg") 


# get the data in tensor array
for result in results:
    print(result.boxes.data)  # torch.Tensor array
    # reference https://docs.ultralytics.com/modes/predict/#working-with-results)

    # get bounding boxes
    print(result.boxes.xywh)  # torch.Tensor array
    # reference https://docs.ultralytics.com/modes/predict/#boxes)

    # get confidence score
    print(result.boxes.conf)