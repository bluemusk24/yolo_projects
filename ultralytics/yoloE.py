from ultralytics import YOLOE


# Prompt Free (YoloE model)
#model = YOLOE("yoloe-11l-seg-pf.pt")

#results = model.predict("ultralytics/berlin_street.mp4", show=True, save=True, show_conf=False)

# Extract results
#for r in results:
#    boxes = r.boxes.xywh.cpu.tolist()
#    cls = r.boxes.cls.cpu.tolist()

#    for b, c in zip(boxes, cls):
#        print(f"box : {b}, class : {model.names[c]}")



# Text Free (YoloE model)
model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict("ultralytics/bus.jpg", save=True, show=True)