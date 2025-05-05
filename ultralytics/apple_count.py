import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("ultralytics/apple_count.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                                
#region_points = [(0, 0), (w, 0), (w, h), (0, h)]    # region_points for whole image (4 corners)
region_points = [(20, 400), (1080, 400)]  # For line counting
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # For rectangle region counting
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting


# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo11n.pt",
    #model="yolo11n-obb.pt",   #for object counting
    line_width=5
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows