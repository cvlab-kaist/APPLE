import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import yaml
import json
from tqdm import tqdm
import cv2 # For drawing circles
import argparse # For dummy args object

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

# --- torchrun setup ---
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
# --- torchrun setup end ---

face_detector = FaceDetector()

# device = 'cuda' if torch.cuda.is_available() else 'cpu' # replaced by torchrun setup
face_landmarks_detector = FaceLandmarksDetector()
iris_detector = IrisDetector()
iris_detector.iris_detector.to(device)

# 시각화 설정
AES_THRESHOLD = 5.0 # AES 점수 임계값
DOT_SIZE = 3 # 랜드마크 점의 크기
# dataset_path = "/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan"
dataset_path = '/mnt/data2/dataset/ffhq_eval/trg'

# 사용자 제공 ImageDataset 클래스
class ImageDataset(Dataset):
    """이미지 경로 리스트를 받아 이미지를 로드하고 전처리하는 데이터셋입니다."""
    def __init__(self, image_paths, face_landmarks_detector, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.face_landmarks_detector = face_landmarks_detector

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 원본 이미지 로드 (center, scale 계산용)
        original_img = Image.open(img_path).convert('RGB')
        original_width, original_height = original_img.size

        # get_coords 함수가 랜드마크를 원본 이미지 좌표계로 변환할 수 있도록
        # center, scale, rotate 값을 설정합니다.
        # 여기서는 원본 이미지 전체를 바운딩 박스로 간주하고, 이를 모델 입력 크기로 스케일링하는 상황을 가정합니다.
        center = np.array([original_width / 2, original_height / 2], dtype=np.float32)
        # scale은 원본 이미지의 긴 변을 기준으로, 모델 입력 크기(예: 200.0 픽셀)에 대한 비율을 반영합니다.
        # Face300W 데이터셋의 scale 계산 방식과 유사하게 설정합니다.
        # scale = np.array([max(original_width, original_height) / 200.0, max(original_width, original_height) / 200.0], dtype=np.float32) # 512 / 200 = 2.56
        scale = max(original_width, original_height) / 200.0
        rotate = 0.0 # 회전이 없다고 가정


        # input_image = np.array(original_img)
        # face_detections = face_detector.predict(input_image)
        # face_detector.visualize(original_img, face_detections)
        # face_landmarks_detections = self.face_landmarks_detector.predict(input_image)

        # 모델 입력용 이미지 전처리
        # img = self.transform(original_img)

        img = np.array(original_img)

        return img, img_path, center, scale, rotate #, face_landmarks_detections[0]

def draw_points(draw, pts, color, r=2):
    for x, y in np.rint(pts).astype(int):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

def main():

    # 사용자 제공 데이터셋 로딩 로직

    # score_json_path = os.path.join(dataset_path, 'score.json')
    # if not os.path.exists(score_json_path):
    #     if rank == 0:
    #         print(f"오류: score.json 파일을 찾을 수 없습니다: {score_json_path}")
    #     return

    # with open(score_json_path, "r") as f:
    #     score_dict = json.load(f)
    # if rank == 0:
    #     print(f"총 {len(score_dict)}개의 이미지에 대한 점수 정보 로드 완료.")
    # high_aes_keys = [k for k, v in score_dict.items() if v.get("aes", -1) > AES_THRESHOLD]
    # if rank == 0:
    #     print(f"총 {len(high_aes_keys)}개의 이미지가 AES 점수 기준 > {AES_THRESHOLD} 충족.")

    # image_paths = []
    # # tqdm disable 조건 수정: rank 0에서만 tqdm 표시
    # for k in tqdm(high_aes_keys, desc="이미지 경로 수집 중", disable=rank!=0):
    #     rel = k + ".jpg"
    #     full = os.path.join(dataset_path, rel)
    #     if os.path.exists(full): # 파일이 실제로 존재하는지 확인
    #         image_paths.append(full)

    # Sort
    from natsort import natsorted
    image_paths = glob.glob(os.path.join(dataset_path, '*.jpg'))
    image_paths = natsorted(image_paths)
    
    # --- torchrun data splitting ---
    image_paths = image_paths[rank::world_size]
    # --- torchrun data splitting end ---

    if rank == 0:
        print(f"추론할 이미지 총 {len(image_paths) * world_size}개 수집 완료 (각 rank당 약 {len(image_paths)}개).")
    else:
        print(f"Rank {rank}: 추론할 이미지 {len(image_paths)}개.")


    dataset = ImageDataset(image_paths, face_landmarks_detector, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)

    for img, img_path, center, scale, rotate in tqdm(dataloader, total=len(dataloader), desc=f"이미지 추론 중 (Rank {rank})", disable=rank!=0):
        try:
            img_path = img_path[0] # 배치 사이즈 1이므로 첫번째 요소 사용
            input_image = img[0].numpy() # 배치 사이즈 1이므로 첫번째 요소 사용, numpy 배열로 변환

            img_dir = os.path.dirname(img_path) # 이미지 디렉토리 경로 
            img_basename = os.path.basename(img_path).split('.')[0] # 이미지 파일명 (확장자 제외)

            save_fname = os.path.join(img_dir, 'iris', f'{img_basename}.png')
            save_vis_fname = os.path.join(img_dir, 'iris_vis', f'{img_basename}.png')
            if os.path.exists(save_fname) and os.path.exists(save_vis_fname):
                print(f"이미 추론된 파일이 존재함: {save_fname}, {save_vis_fname}, 건너뜀.")
                continue
            os.makedirs(os.path.dirname(save_fname), exist_ok=True)
            os.makedirs(os.path.dirname(save_vis_fname), exist_ok=True)



            face_landmarks_detections = face_landmarks_detector.predict(input_image)
            face_landmarks_detection = face_landmarks_detections[0]  # 배치 사이즈 1이므로 첫번째 요소 사용
            left_eye_image, right_eye_image, left_config, right_config = iris_detector.preprocess(input_image, face_landmarks_detection)

            left_eye_contour, left_eye_iris = iris_detector.predict(left_eye_image)
            right_eye_contour, right_eye_iris = iris_detector.predict(right_eye_image, isLeft=False)

            ori_left_eye_contour, ori_left_iris = iris_detector.postprocess(left_eye_contour, left_eye_iris, left_config)
            ori_right_eye_contour, ori_right_iris = iris_detector.postprocess(right_eye_contour, right_eye_iris, right_config)
            pil = Image.fromarray(input_image)
            draw = ImageDraw.Draw(pil)
            draw_points(draw, ori_left_iris[:, :2], color='red', r=2)
            draw_points(draw, ori_right_iris[:, :2], color='red', r=2)
            pil.save(save_vis_fname)

            # Black background
            black_img = np.zeros_like(input_image)
            pil_black = Image.fromarray(black_img)
            draw_black = ImageDraw.Draw(pil_black)
            draw_points(draw_black, ori_left_iris[:, :2], color='red', r=2)
            draw_points(draw_black, ori_right_iris[:, :2], color='red', r=2)
            pil_black.save(save_fname)  
            
        except Exception as e:
            print(f"오류 발생: {e} - 이미지 경로: {img_path}")
            continue


    if rank == 0:
        print("모든 이미지에 대한 추론 및 시각화 완료.")

if __name__ == '__main__':
    main()
