## Ultralytics (Yolo11) Lectures:

* https://artlist.io/ --> to get mp4 videos for computer vision project

### Using GUI in WSL2 for Windows OS
```bash
wsl
echo $WAYLAND_DISPLAY
sudo apt update
sudo apt install libxcb-xinerama0
conda activate <computer_vision> # virtual environment
export QT_QPA_PLATFORM=xcb
sudo apt install libxcb1 libx11-6 libxrender1 libsm6
```

### 1. Object Detection and Tracking with Yolo11:

[yolo11.py](ultralytics/yolo11.py)
   
```bash
python3 ultralytics/yolo11.py

python3 AI-Football/yolo.py
```
* It will still save the annotated video to a ```runs/detect/predict...``` folder.

### 2. How to Benchmark Ultralytics YOLO11 Models | How to Compare Model Performance on Different Hardware Format?

[Benchmark Docs](https://docs.ultralytics.com/modes/benchmark/)

[benchmark notebook](ultralytics/Yolo11_Benchmark.ipynb) on Google Colab (GPU Access)

### 3. Ultralytics YOLO11 Pose Estimation Tutorial | Real-Time Object Tracking and Human Pose Detection

[Pose Estimation Docs](https://docs.ultralytics.com/tasks/pose/). It detects all parts of human body.

[pose_estimation.py](ultralytics/pose_estimation.py)

```bash
python3 ultralytics/pose_estimation.py
```

### 4. Auto Annotation with Meta's Segment Anything 2 Model using Ultralytics | SAM 2.1 | Data Labeling

[Auto Annotation Docs](https://docs.ultralytics.com/models/sam-2/). To annotate labels of data automatically 

[auto_annotation.py](ultralytics/auto_annotation.py)

```bash
python3 ultralytics/auto_annotation.py
```

* a ```ultralytics/bus_auto_annotate_labels/bus.txt``` file is created in the directory to view the polygon points of annotated classes and labels.

### 5. Hand Keypoints Estimation with Ultralytics YOLO11 | Human Hand Pose Estimation Tutorial

[Hand Keypoint Docs](https://docs.ultralytics.com/datasets/pose/hand-keypoints/). Detecting 21 different points in the wrist and fingers. 

[hand_keypoints notebook](ultralytics/hand_keypoints.ipynb)

* Note: the best-trained model can be downloaded from ```runs/weights``` folder on Google Colab and used for personal project on hand keypoints.

### 6. Car Parts Segmentation with Ultralytics YOLO11: A Step-by-Step Image Segmentation Tutorial

[CarParts Segmentation Docs](https://docs.ultralytics.com/datasets/segment/carparts-seg/)

[carparts segmentation notebook](ultralytics/carparts_segmentation.ipynb)

* Note: the best-trained model can be downloaded from ```runs/weights``` folder on Google Colab and used for personal project on car parts segmentation.

### 7. In-Depth Guide to Text & Circle Annotations with Python Live Demos | Ultralytics Annotations

[Texts&Circle Docs](https://docs.ultralytics.com/usage/simple-utilities/#bounding-boxes-circle-annotation-circle-label)

### 8. How to do Object Counting in Different Regions using Ultralytics YOLO11 | Ultralytics Solutions 🚀

[Region Counting Docs](https://docs.ultralytics.com/guides/region-counting/)

[region_counter.py](ultralytics/region_counter.py). This was copied from [Ultralytics github](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py)

* Run inference using the CPU
```bash
python3 region_counter.py --source "path/to/video.mp4" --save-img --device cpu

# example
python3 ultralytics/region_counter.py --source "ultralytics/6103843_Recreation GameFun Streetball_By_Raven_Production_Artlist_HD.mp4" --save-img --device cpu
```

* check for outcome of the above run: ```ultralytics_rc_output/exp/6103843_Recreation Game Fun Streetball_By_Raven_Production_Artlist_HD.avi```

### 9. Insights into Model Evaluation and Fine-Tuning | Tips for Improving Mean Average Precision | YOLO11🚀

[Model Evaluation and Fine-Tuning Docs](https://docs.ultralytics.com/guides/model-evaluation-insights/). No codes here

### 10. How to Train Ultralytics YOLO11 Model on Custom Dataset using Google Colab Notebook | Step-by-Step 🚀

[yolo11_training notebook](ultralytics/yolo11_trainingTutorial.ipynb)

### 11. History of Computer Vision Models | OpenCV | Pattern Recognition | YOLO Models | Ultralytics YOLO11🚀

[History of Computer Vision](https://www.ultralytics.com/blog/a-history-of-vision-models). No codes here

### 12. How to count using Ultralytics YOLO11 Oriented Bounding Boxes (YOLO11-OBB) | Object Counting 🚀

[Object Count Docs](https://docs.ultralytics.com/guides/object-counting/#__tabbed_1_2)

[object_counting.py](ultralytics/object_counting.py)
```bash
python3 ultraltics/object_counting.py
```

### 13. How to Tune Hyperparameters for Better Model Performance | Ultralytics YOLO11 Hyperparameters 🚀

[Hyperparamters Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/). No codes here

### 14. How to Train Ultralytics YOLO11 on CIFAR-100 | Step-by-Step Image Classification Tutorial 🚀

[CIFAR-100 Docs](https://docs.ultralytics.com/datasets/classify/cifar100/)

[cifar-100 notebook](ultralytics/cifar_100.ipynb)

### 15. How to Test Machine Learning Models | Avoid Data Leakage in Computer Vision | Ultralytics YOLO11🚀

[Model-Testing Docs](https://docs.ultralytics.com/guides/model-testing/). No codes here

### 16. How to Run Ultralytics Solutions from the Command Line (CLI) | Ultralytics YOLO11 | Object Counting🚀

[Ultralytics Solutions Docs](https://docs.ultralytics.com/solutions/)

* run below on terminal with any solution name ['count', 'crop', 'blur', 'workout', 'heatmap', 'isegment', 'queue', 'speed', 'analytics', 'trackzone', 'inference', 'visioneye']
```bash
yolo solutions heatmap source="ultralytics/6381959_Milan Italy Bridge Road_By_Escape_Routine_Artlist_HD.mp4" show=True # specify video file path.
```

### 17. How to Export the Ultralytics YOLO11 to ONNX, OpenVINO and Other Formats using Ultralytics HUB 🚀

[Model Export Docs](https://docs.ultralytics.com/modes/export/). No codes here

### 18. How to train Ultralytics YOLO11 Model on Medical Pills Detection Dataset in Google Colab | Pharma AI

[Medical Pills Docs](https://docs.ultralytics.com/datasets/detect/medical-pills/#dataset-yaml)

[medical-pill notebook](ultralytics/medical_pill.ipynb)

### 19. How to Track Objects in Region using Ultralytics YOLO11 | TrackZone | Ultralytics Solutions 🚀

[TrackZone Docs](https://docs.ultralytics.com/guides/trackzone/)

[trackzone.py](ultralytics/trackzone.py)

### 20. How to do Package Segmentation using Ultralytics YOLO11 | Industrial Packages | Model Training 🎉

[Package Segmentation Docs](https://docs.ultralytics.com/datasets/segment/package-seg/)

[package-segmentation notebook](ultralytics/package_segmentation.ipynb)

### 21. How to build Security Alarm System using Ultralytics YOLO11 | Ultralytics Solutions 🚀

[Security Alarm Docs](https://docs.ultralytics.com/guides/security-alarm-system/). Send alarms/alert via email whenever ther is an intruder in a region. Also, ensure to generate password from the ```APP PASSWORD GENERATOR in the Docs.```

[security_alarm.py](ultralytics/security_alarm.py)

### 22. How to use Batch Inference with Ultralytics YOLO11 | Speed Up Object Detection in Python 🎉

[Batch Inference Docs](https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode). Inference for running multiple images

[batch_inference.py](ultralytics/batch_infence.py)

### 23. YOLO Models Comparison: Ultralytics YOLO11 vs. YOLOv10 vs. YOLOv9 vs. Ultralytics YOLOv8 🎉

[Models Comparison Docs](https://docs.ultralytics.com/models/)

[yolo_comparison.ipynb](ultralytics/models_comparison.ipynb)

### 24. How to use Ultralytics Visual Studio Code Extension | Ready-to-Use Code Snippets | Ultralytics YOLO🎉

[Ultralytics VS_Code](https://docs.ultralytics.com/integrations/vscode/)

[test.py](ultralytics/test.py)

### 25. How to use YOLOE with Ultralytics: Open Vocabulary & Real-Time Seeing Anything | Text/Visual Prompt🚀

[YoloE Docs](https://docs.ultralytics.com/models/yoloe/)

[yoloE.py](ultralytics/yoloE.py)

### 26. How to define Computer Vision Project's Goal | Problem Statement and VisionAI Tasks Connection 🚀

[Computer Project Docs](https://docs.ultralytics.com/guides/defining-project-goals/)

### 27. How to Perform Real-Time Object Counting with Ultralytics YOLO11 | Apples on a Moving Conveyor Belt🍏

[apple_count.py](ultralytics/apple_count.py)



## PROJECT: Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python