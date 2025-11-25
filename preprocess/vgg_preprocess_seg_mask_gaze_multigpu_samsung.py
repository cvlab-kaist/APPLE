# 싱긇GPU보다 느림!
# --- (추가) 멀티프로세싱 import ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # 사용할 GPU 설정
import torch
from torchvision import transforms
import multiprocessing
from tqdm import tqdm
from glob import glob
from PIL import Image
# from diffusers.utils import make_image_grid
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import cv2
import shutil

# (선택) OpenCV 내부 스레드 과도한 오버구독 방지
# cv2.setNumThreads(1)

from faceparsing.models.parsing import BiSeNet
from faceparsing.utils.module import SpecificNorm
from mmseg.apis import init_model

device = 'cuda'

# Load models for this process
netSeg = BiSeNet(n_classes=19).to(device).eval()
spNorm = SpecificNorm().to(device).eval()
netSeg.load_state_dict(torch.load('../checkpoints/FaceParser.pth', weights_only=False))

config_path = './deeplabv3plus_celebA_train_wo_natocc_wsot.py'
checkpoint_path = '../checkpoints/deeplabv3_iter_27600.pth'
occ_model = init_model(config_path, checkpoint_path, device=device)

attributes = {
'background' : 0, 
'skin' : 1,
'r_brow' : 2,
'l_brow' : 3,
'r_eye' : 4,
'l_eye' : 5,
'eye_g' : 6,
'l_ear' : 7,
'r_ear' : 8,
'ear_r' : 9,
'nose' : 10,
'mouth' : 11,
'u_lip' : 12,
'l_lip' : 13,
'neck' : 14,
'neck_l' : 15,
'cloth' : 16,
'hair' : 17,
'hat' : 18,
}
color_list = [[0, 0, 0], 
            [255, 0, 0], 
            [0, 204, 204], 
            [0, 0, 204], 
            [255, 153, 51], 
            [204, 0, 204], 
            [255, 0, 255],
            [204, 0, 0], 
            [102, 51, 0], 
            [0, 0, 0], 
            [76, 153, 0], 
            [102, 204, 0], 
            [255, 255, 0], 
            [0, 0, 153],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

image_folder_path = '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan'
from natsort import natsorted
image_path_list = natsorted(glob(os.path.join(image_folder_path, '*/*.jpg'))) # /mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n000003/0001_01.jpg
landmark_path_list = []
iris_landmark_path_list = []

### 1) 이미지 파일, 3DMM 랜드마크 파일 경로 리스트 생성 ###
for image_path in image_path_list:
    image_base = os.path.dirname(image_path) # /mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n000003
    file_name = os.path.basename(image_path).split('.')[0] # 0001_01
    landmark_path = os.path.join(image_base, f'3dmm/{file_name}/{file_name}_ldm68.png') # 이미지에 대응하는 랜드마크 경로
    iris_landmark_path = os.path.join(image_base, f'iris/{file_name}.png') # 이미지에 대응하는 랜드마크 경로
    landmark_path_list.append(landmark_path)
    iris_landmark_path_list.append(iris_landmark_path)


def get_segmentation(img_path, occ_model, netSeg, spNorm, device, target_hw=512):
    from mmseg.apis import inference_model
    # Occlusion mask
    result = inference_model(occ_model, img_path)
    mask_img = Image.fromarray(result.pred_sem_seg.data.cpu().numpy().squeeze().astype(np.uint8) * 255).convert('RGB').resize((target_hw, target_hw), Image.BILINEAR)
    mask_img_np = np.array(mask_img)


    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((target_hw, target_hw)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_input = transform(img).to(device)
    if len(img_input.shape) == 3:
        img_input = img_input.unsqueeze(0)
    targ_mask = netSeg(spNorm(img_input))[0]
    parsing   = targ_mask.squeeze(0).detach().cpu().numpy().argmax(0)
    targ_seg = np.zeros((target_hw, target_hw, 3))
    targ_mask = np.zeros((target_hw, target_hw, 3))
    for idx, color in enumerate(color_list):
        targ_seg[parsing == idx] = color
        if idx != 0 :
            if color != [0, 0, 0]:
                targ_mask[parsing == idx] = [255, 255, 255]

    targ_seg_PIL = Image.fromarray(targ_seg.astype(np.uint8))
    targ_mask_PIL = Image.fromarray(targ_mask.astype(np.uint8))
    

    # Get intersection
    targ_mask_np = np.array(targ_mask_PIL)
    intersection_mask_np = np.logical_and(mask_img_np > 0, targ_mask_np > 0).astype(np.uint8) * 255
    intersection_mask_PIL = Image.fromarray(intersection_mask_np)
    intersection_seg_np = np.array(targ_seg_PIL)
    intersection_seg_np[intersection_mask_np == 0] = 0
    intersection_seg_PIL = Image.fromarray(intersection_seg_np.astype(np.uint8))
    
    # Results
    return targ_seg_PIL, targ_mask_PIL, intersection_seg_PIL, intersection_mask_PIL


def process_idx(idx, occ_model, netSeg, spNorm, device):
    try :
        image_base = os.path.dirname(image_path_list[idx]) # /mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n000003
        file_name = os.path.basename(image_path_list[idx]).split('.')[0] # 0001_01
        image_path = image_path_list[idx]

        # 최종 출력 경로 설정 및 이미 처리된 경우 건너뛰기
        # output_dir = os.path.join(image_base, 'condition_blended_image_blurdownsample8_seg_landmark')
        output_dir2 = os.path.join(image_base, 'condition_blended_image_blurdownsample8_segGlass_landmark_iris')
        # os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir2, exist_ok=True)

        # output_path_1 = os.path.join(output_dir, f'{file_name}.png')
        output_path_2 = os.path.join(output_dir2, f'{file_name}.png')
        # if os.path.exists(output_path_1) and os.path.exists(output_path_2):
        if os.path.exists(output_path_2):
            # print(f"Skipping index {idx}, output already exists.")
            return True

        # 3) Segmentaion 및 Mask 획득
        image = Image.open(image_path)

        seg_folder_path = os.path.join(image_base, 'segmap')
        seg_intersection_folder_path = os.path.join(image_base, 'segmap_intersection')
        mask_folder_path = os.path.join(image_base, 'mask')
        mask_intersection_folder_path = os.path.join(image_base, 'mask_intersection')

        seg_fname = os.path.join(seg_folder_path, file_name + '.png')
        seg_inter_fname = os.path.join(seg_intersection_folder_path, file_name + '.png')
        mask_fname =  os.path.join(mask_folder_path, file_name + '.png')
        mask_inter_fname = os.path.join(mask_intersection_folder_path, file_name + '.png')

        # # 있으면 스킵
        # if os.path.exists(seg_fname) and os.path.exists(seg_inter_fname) and os.path.exists(mask_fname) and os.path.exists(mask_inter_fname):
        #     seg = Image.open(seg_fname)
        #     mask = Image.open(mask_fname)
        #     seg_inter = Image.open(seg_inter_fname)
        #     mask_inter = Image.open(mask_inter_fname)
        # # 없으면 생성
        # else:
        seg, mask, seg_inter, mask_inter = get_segmentation(image_path, occ_model, netSeg, spNorm, device, target_hw=image.size[1])

        os.makedirs(seg_folder_path, exist_ok=True)
        os.makedirs(seg_intersection_folder_path, exist_ok=True)
        os.makedirs(mask_folder_path, exist_ok=True)
        os.makedirs(mask_intersection_folder_path, exist_ok=True)


        seg.save(seg_fname)
        mask.save(mask_fname)
        seg_inter.save(seg_inter_fname)
        mask_inter.save(mask_inter_fname)

        ##### Processing for condition image ##### 
        # Intersection 버전을 사용해서 프로세싱합니다.
        landmark = Image.open(landmark_path_list[idx])
        image_np = np.array(image)
        seg_np = np.array(seg_inter)
        mask_np = np.array(mask_inter)
        landmark_np = np.array(landmark)
        iris_landmark_np = np.array(Image.open(iris_landmark_path_list[idx]))

        h, w, c = image_np.shape
        downsample_size = 8
        sample_img_np_downsampled = image.resize((downsample_size, downsample_size), Image.LANCZOS).resize((w, h), Image.LANCZOS)

        # 3DMM 랜드마크 마스크 뽑기
        sample_landmark_np_sum = np.sum(landmark_np, axis=-1)
        landmark_idx = sample_landmark_np_sum > 0
        # landmark_idx = landmark_idx[..., None].repeat(3, axis=-1)
        
        # iris 랜드마크 마스크 뽑기
        iris_landmark_idx = iris_landmark_np.sum(axis=-1) > 0 # shape (h, w)
        # iris_landmark_idx = iris_landmark_idx[..., None].repeat(3, axis=-1) # shape (h, w, 3)


        # Union of seg and landmark
        seg_landmark = deepcopy(seg_np)
        seg_landmark[landmark_idx] = 255

        # select seg attribute by removing target
        remove_target = ['skin', 'l_ear', 'r_ear']
        seg_landmark_selected = deepcopy(seg_landmark)
        for target in remove_target:
            target_idx = attributes[target]
            target_color = color_list[target_idx]
            target_mask = (seg_landmark_selected[..., 0] == target_color[0]) & \
                        (seg_landmark_selected[..., 1] == target_color[1]) & \
                        (seg_landmark_selected[..., 2] == target_color[2])
            seg_landmark_selected[target_mask] = 0

        # seg_glass
        seg_glass = np.zeros_like(seg_np)
        glass_target = 'eye_g'
        t_idx = attributes[glass_target]
        t_color = color_list[t_idx]
        glass_mask = (seg_np[..., 0] == t_color[0]) & (seg_np[..., 1] == t_color[1]) & (seg_np[..., 2] == t_color[2])
        seg_glass[glass_mask] = t_color
        glass_mask = glass_mask[..., None].repeat(3, axis=-1)

        # skin mask
        skin_target = list(attributes.keys())
        seg_skin = np.zeros_like(seg_np)
        for t in skin_target:
            t_idx = attributes[t]
            t_color = color_list[t_idx]
            t_mask = (seg_np[..., 0] == t_color[0]) & (seg_np[..., 1] == t_color[1]) & (seg_np[..., 2] == t_color[2])
            seg_skin[t_mask] = t_color
        mask_skin = (seg_skin[..., 0] != 0) | (seg_skin[..., 1] != 0) | (seg_skin[..., 2] != 0)
        mask_skin = mask_skin[..., None].repeat(3, axis=-1)

        # Blur + landmark
        mask_for_seg = (seg_landmark_selected[..., 0] != 0) | (seg_landmark_selected[..., 1] != 0) | (seg_landmark_selected[..., 2] != 0)
        mask_for_seg = mask_for_seg[..., None].repeat(3, axis=-1)
        condition_blur_landmark = deepcopy(image_np)
        condition_blur_landmark = condition_blur_landmark * (1 - mask_skin) + np.array(sample_img_np_downsampled) * mask_skin # A1) 피부를 블러처리합니다.
        condition_blur_landmark = condition_blur_landmark.astype(np.uint8)
        condition_blur_landmark[landmark_idx] = 255 # A2) 3DMM 랜드마크를 덧붙입니다.
        # iris 랜드마크 빨간색으로 덧붙이기
        condition_blur_landmark[iris_landmark_idx] = [255, 0, 0]

        condition_blur_landmark_glass = condition_blur_landmark * (1-glass_mask) + cv2.addWeighted(condition_blur_landmark, 0.9, seg_glass, 0.1, 0) * glass_mask # A3) 안경 부분을 Seg 처리합니다.
        condition_blur_landmark_glass = condition_blur_landmark_glass.astype(np.uint8)

        condition_blur_segSelected_landmark = deepcopy(image_np)
        condition_blur_segSelected_landmark = condition_blur_segSelected_landmark * (1 - mask_skin) + np.array(sample_img_np_downsampled) * mask_skin
        condition_blur_segSelected_landmark = condition_blur_segSelected_landmark * (1 - mask_for_seg) + seg_landmark_selected * mask_for_seg
        condition_blur_segSelected_landmark = condition_blur_segSelected_landmark.astype(np.uint8)

        blended_image = cv2.addWeighted(condition_blur_landmark, 0.8, condition_blur_segSelected_landmark, 0.2, 0)
        blended_image_pil = Image.fromarray(blended_image)
        file_name = os.path.basename(image_path_list[idx]).split('.')[0]
        # blended_image_pil.save(os.path.join(output_dir, f'{file_name}.png'))

        condition_blur_landmark_glass_pil = Image.fromarray(condition_blur_landmark_glass)
        condition_blur_landmark_glass_pil.save(os.path.join(output_dir2, f'{file_name}.png'))
        return True
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return False

def worker(process_id, gpu_id, indices_chunk):
    # device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')


    for idx in tqdm(indices_chunk, desc=f"Process {process_id}", position=process_id, dynamic_ncols=True):
        process_idx(idx, occ_model, netSeg, spNorm, device)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    num_processes = 32

    indices = list(range(len(image_path_list)))
    chunks = [indices[i::num_processes] for i in range(num_processes)]

    processes = []
    process_id_counter = 0
    for _ in range(num_processes):
        indices_chunk = chunks[process_id_counter]
        p = multiprocessing.Process(target=worker, args=(process_id_counter, None, indices_chunk))
        processes.append(p)
        p.start()
        process_id_counter += 1
            
    for p in processes:
        p.join()

    print("All processes finished.")
