<H1 align="center">
YOLOv8 Object Detection with DeepSORT Tracking(ID + Trails) </H1>

## Colab File Link 
The google colab file link for yolov8 object detection and tracking is provided below, you can check the implementation in Google Colab, and its a single click implementation
,you just need to select the Run Time as GPU, and click on Run All.

[`Google Colab File`](https://colab.research.google.com/drive/1wEsHknk11SWak80SJ7uGWZf5-BBrfMZx?usp=sharing)

## Steps to run Code

- Clone the repository
```
git clone https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking.git
```
- Goto the cloned folder.
```
cd YOLOv8-DeepSORT-Object-Tracking
```
- Install the requirements
```
pip install -r requirements.txt

```
- To use the YOLOv8 Command Line Interface, install the Ultralytics Package
```

pip install ultralytics

```
- Setting the Directory.
```
cd yolo/v8/detect

```
- Downloading the DeepSORT Files From The Google Drive ,place them into the yolo/v8/detect folder
```

https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8?usp=sharing


```
- Downloading a Sample Video from the Google Drive
```
gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
```

- Run the code with mentioned command below.

- For yolov8 object detection + Tracking
```
python tracking.py model=yolov8l.pt source="test3.mp4" show=True
```

[![Watch the Complete Video Tutorial with Step by Step Explanation here]
(http://img.youtube.com/vi/9jRRZ-WL698/./figure/figure1.png)]
(https://www.youtube.com/watch?v=9jRRZ-WL698)
http://img.youtube.com/vi/<9jRRZ-WL698>/hqdefault.jpg

### RESULTS

#### Vehicles Detection, Tracking and Counting 
![](./figure/figure1.png)

#### Vehicles Detection, Tracking and Counting

![](./figure/figure3.png)
