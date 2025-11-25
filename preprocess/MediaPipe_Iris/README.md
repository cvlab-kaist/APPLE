# MediaPipe Iris Pytorch

## Introduction
This repository demonstrate Google's [MediaPipe](https://google.github.io/mediapipe/) Iris model using Python and Pytorch.

Official Project Page: https://google.github.io/mediapipe/solutions/iris.html

## Iris Prediction 
MediaPipe Iris Detection process consists of two parts: face landmarks detection and iris prediction with respect to eye-cropped images. 

### Face Landmark Model
Face landmarks are detected using the [Face Landmark Model](https://google.github.io/mediapipe/solutions/face_mesh.html), which is the same as the MediaPipe Face Mesh model. The model predicts each face's bounding boxes and 468 3D face landmarks in real-time. The definitions of the landmark points can be found [here](https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts). 

### Iris Model

#### Preprocess
Iris model takes in a 64x64 center-cropped RGB left-eye image with a 25% margin on each side. 

Therefore, after obtaining the face landmarks, we have to extract eye region out for both eyes (rightEyeUpper0, rightEyeLower0, leftEyeUpper0, leftEyeLower0), see [here](https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts). The center of each pupil is determined by the average of the region. 

The 25% margin on each side is achieved by first determine the cropped size($d_{crop}$) using the distance between the left-most and right-most points, i.e. the eye width($d_{ew}$), for each eye region respectively.
$$d_{crop} = 2 \times d_{ew}$$

We center-cropped the image using the pupil center and the calculated cropped size, then resize the image to 64x64. We also need to flip the image horizontally if it is a right eye.

#### Main model
The model is based on [MobileNetV2](https://arxiv.org/abs/1801.04381) modified with customized blocks for real-time performance. It contains two outputs: eyeContours(213) and iris(15). 
* Eyecontours: the eye with eyebrow region(71 points in 3D coordinates) 
* Iris: pupil center(1 point in 3D) + iris contour(4 points in 3D)

Google provides the [tflite model](https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite) in the official project page. In this repository, I converted the model from tflite to the structure of pytorch.

More details:
- [Model Card](https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/preview)
- [Paper](https://arxiv.org/pdf/2006.11341.pdf)

## Get Started

### Setup the environment
```
git clone https://github.com/Morris88826/MediaPipe_Iris.git
cd MediaPipe_Iris

conda create --name mediapipe_iris python=3.7
conda activate mediapipe_iris
pip install -r requirements.txt
```

### Extract weights from tflite model
Run extract_iris_landmark_model.py file to extract weights in the tflite model. The extracted weights will be stored in the data folder. One can verify the result by running libs/iris.py. This will save the checkpoint file in pytorch format(.pth) if not yet created in the data folder.

```
python extract_iris_landmark_model.py 
python libs/iris.py
```

### Test the result
```
python main.py # this will run the demo
python main.py --source {path_to_the_video}
```