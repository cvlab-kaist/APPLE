import os
import glob
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from libs.helper_func import vid2images, images2vid
from libs.face import FaceDetector, FaceLandmarksDetector
from libs.iris import IrisDetector

def main(args):
    video_name = args.source.split('/')[-1].split('.')[0]

    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.mkdir('./tmp')
    
    vid2images(args.source, out_path='./tmp')

    if not os.path.exists('./results'):
        os.mkdir('./results')
    output_dir = './results/{}'.format(video_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir+'/images'):
        os.mkdir(output_dir+'/images')

    face_landmarks_detector = FaceLandmarksDetector()    
    iris_detector = IrisDetector()
    for image_path in tqdm(sorted(glob.glob('./tmp/*.png'))):
        input_image = np.array(Image.open(image_path).convert('RGB'))

        face_landmarks_detections = face_landmarks_detector.predict(input_image)

        for face_landmarks_detection in face_landmarks_detections:
            left_eye_image, right_eye_image, left_config, right_config = iris_detector.preprocess(input_image, face_landmarks_detection)

            left_eye_contour, left_eye_iris = iris_detector.predict(left_eye_image)
            right_eye_contour, right_eye_iris = iris_detector.predict(right_eye_image, isLeft=False)
            
            ori_left_eye_contour, ori_left_iris = iris_detector.postprocess(left_eye_contour, left_eye_iris, left_config)
            ori_right_eye_contour, ori_right_iris = iris_detector.postprocess(right_eye_contour, right_eye_iris, right_config)
            plt.imshow(input_image)
            # plt.scatter(ori_left_eye_contour[:, 0], ori_left_eye_contour[:, 1], s=3)
            plt.scatter(ori_left_iris[:, 0], ori_left_iris[:, 1], s=2)
            # plt.scatter(ori_right_eye_contour[:, 0], ori_right_eye_contour[:, 1], s=3)
            plt.scatter(ori_right_iris[:, 0], ori_right_iris[:, 1], s=2)
            plt.axis('off')
            plt.savefig(output_dir+'/images/{}'.format(image_path.split('/')[-1]))
            plt.close()

    images2vid(output_dir+'/images', output_dir=output_dir)
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')

def demo():
    input_image = np.array(Image.open('./examples/01.png').convert('RGB'))
    
    face_detector = FaceDetector()
    face_detections = face_detector.predict(input_image)
    face_detector.visualize(input_image, face_detections)

    face_landmarks_detector = FaceLandmarksDetector()
    face_landmarks_detections = face_landmarks_detector.predict(input_image)
    face_landmarks_detector.visualize(input_image, face_landmarks_detections)

    for face_landmarks_detection in face_landmarks_detections:
        iris_detector = IrisDetector()
        left_eye_image, right_eye_image, left_config, right_config = iris_detector.preprocess(input_image, face_landmarks_detection)

        left_eye_contour, left_eye_iris = iris_detector.predict(left_eye_image)
        right_eye_contour, right_eye_iris = iris_detector.predict(right_eye_image, isLeft=False)

        fig, [ax1, ax2] = plt.subplots(1,2)        
        ax1.imshow(right_eye_image)
        ax1.scatter(right_eye_iris[:, 0], right_eye_iris[:, 1], s=3)
        ax1.scatter(right_eye_contour[:, 0], right_eye_contour[:, 1], s=3)
        ax1.set(title='right eye')
        ax2.imshow(left_eye_image)
        ax2.scatter(left_eye_iris[:, 0], left_eye_iris[:, 1], s=3)
        ax2.scatter(left_eye_contour[:, 0], left_eye_contour[:, 1], s=3)
        ax2.set(title='left eye')
        plt.show()
        
        ori_left_eye_contour, ori_left_iris = iris_detector.postprocess(left_eye_contour, left_eye_iris, left_config)
        ori_right_eye_contour, ori_right_iris = iris_detector.postprocess(right_eye_contour, right_eye_iris, right_config)
        plt.imshow(input_image)
        plt.scatter(ori_left_eye_contour[:, 0], ori_left_eye_contour[:, 1], s=3)
        plt.scatter(ori_left_iris[:, 0], ori_left_iris[:, 1], s=2)
        plt.scatter(ori_right_eye_contour[:, 0], ori_right_eye_contour[:, 1], s=3)
        plt.scatter(ori_right_iris[:, 0], ori_right_iris[:, 1], s=2)
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', default="", help="Path to the video")
    args = parser.parse_args()

    if args.source is "":
        demo()
    else:
        main(args)