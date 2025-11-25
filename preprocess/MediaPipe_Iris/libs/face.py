import cv2
import json
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FaceDetector():
    def __init__(self) -> None:
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def predict(self, image):
        # INPUT: RGB image
        HEIGHT, WIDTH, _ = image.shape
        results = self.face_detector.process(image)
        
        detections = []
        for detection in results.detections:
            _detection = {
                "bbox": [], # xmin, ymin, width, height
                "right_eye": [], # x, y position
                "left_eye": [], # x, y position
                "nose_tip": [], # x, y position
                "mouth_center": [], # x, y position
                "right_ear_tragion": [], # x, y position
                "left_ear_tragion": [], # x, y position
            }
            bbox = detection.location_data.relative_bounding_box            
            right_eye = self.mp_face_detection.get_key_point(detection, self.mp_face_detection.FaceKeyPoint.RIGHT_EYE)
            left_eye = self.mp_face_detection.get_key_point(detection, self.mp_face_detection.FaceKeyPoint.LEFT_EYE)
            nose_tip = self.mp_face_detection.get_key_point(detection, self.mp_face_detection.FaceKeyPoint.NOSE_TIP)
            mouth_center = self.mp_face_detection.get_key_point(detection, self.mp_face_detection.FaceKeyPoint.MOUTH_CENTER)
            right_ear_tragion = self.mp_face_detection.get_key_point(detection, self.mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
            left_ear_tragion = self.mp_face_detection.get_key_point(detection, self.mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
            
            _detection["bbox"] = [bbox.xmin*WIDTH, bbox.ymin*HEIGHT, bbox.width*WIDTH, bbox.height*HEIGHT]
            _detection["right_eye"] = [right_eye.x*WIDTH, right_eye.y*HEIGHT]
            _detection["left_eye"] = [left_eye.x*WIDTH, left_eye.y*HEIGHT]
            _detection["nose_tip"] = [nose_tip.x*WIDTH, nose_tip.y*HEIGHT]
            _detection["mouth_center"] = [mouth_center.x*WIDTH, mouth_center.y*HEIGHT]
            _detection["right_ear_tragion"] = [right_ear_tragion.x*WIDTH, right_ear_tragion.y*HEIGHT]
            _detection["left_ear_tragion"] = [left_ear_tragion.x*WIDTH, left_ear_tragion.y*HEIGHT]

            detections.append(_detection)
        return detections
    
    def visualize(self, image, detections):
        fig, ax = plt.subplots()
        ax.imshow(image)

        for detection in detections:
            bbox = detection["bbox"]
            right_eye = detection["right_eye"]
            left_eye = detection["left_eye"]
            nose_tip = detection["nose_tip"]
            mouth_center = detection["mouth_center"]
            right_ear_tragion = detection["right_ear_tragion"]
            left_ear_tragion = detection["left_ear_tragion"]

            ax.scatter(right_eye[0], right_eye[1], s=3)
            ax.scatter(left_eye[0], left_eye[1], s=3)
            ax.scatter(nose_tip[0], nose_tip[1], s=3)
            ax.scatter(mouth_center[0], mouth_center[1], s=3)
            ax.scatter(right_ear_tragion[0], right_ear_tragion[1], s=3)
            ax.scatter(left_ear_tragion[0], left_ear_tragion[1], s=3)
            
            rect = patches.Rectangle((bbox[0], bbox[1]), width=bbox[2], height=bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.title("Face Detector")
        plt.show()


class FaceLandmarksDetector():
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

        with open('./data/face_landmarks.json', 'rb') as jsonfile:
            self.landmark_indices = json.load(jsonfile)
    def predict(self, image):
        # INPUT: RGB image
        HEIGHT, WIDTH, _ = image.shape
        results = self.face_mesh_detector.process(image)
        
        landmarks = []
        for detection in results.multi_face_landmarks:
            _landmarks = []
            for landmark in detection.landmark:
                _landmarks.append([landmark.x*WIDTH, landmark.y*HEIGHT, landmark.z*WIDTH])
            
            _landmarks = np.array(_landmarks)
            landmarks.append(_landmarks)
        
        return landmarks

    def get_face_landmarks_indices_by_regions(self, regions:list):
        indices = []
        for region in regions:
            if region in self.landmark_indices.keys():
                indices += self.landmark_indices[region]
        return indices

    def get_face_landmarks_indices_by_region(self, region:str):
        if region in self.landmark_indices.keys():
            return self.landmark_indices[region]
        return None

    def visualize(self, image, detections, regions=None, indices=None):
        fig, ax = plt.subplots()
        ax.imshow(image)

        for landmarks in detections:

            if regions is not None:
                for region in regions:
                    if region in self.landmark_indices.keys():
                        _indices = self.landmark_indices[region]
                        ax.scatter(landmarks[_indices, 0], landmarks[_indices, 1], s=3)
            elif indices is not None:
                ax.scatter(landmarks[indices, 0], landmarks[indices, 1], s=3)
            else:
                ax.scatter(landmarks[:, 0], landmarks[:, 1], s=3)
        plt.title("Face Landmarks Detector")
        plt.show()

