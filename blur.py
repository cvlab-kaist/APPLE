from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image, ImageFilter
from diffusers.utils import make_image_grid

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


from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image, ImageFilter
from diffusers.utils import make_image_grid


def create_condition_images(
    image: Image.Image,
    seg: Image.Image,
    mask: Image.Image,
    landmark: Image.Image,
    iris: Image.Image,
    # attributes: Dict[str, int],
    # color_list: List[Tuple[int, int, int]],
    condition: str = 'downsample',  # 'blur' or 'downsample' or 'None' or 'mask'
    downsample_size: int = 8,
    blur_radius: int = 64,
    remove_targets: List[str] = None,   # seg에서 제거할 attribute
    glass_target: str = "eye_g",        # 안경 영역 이름
    skin_targets: List[str] = None      # blur를 덮어씌울 영역 (None이면 전체 attributes)
) -> Dict[str, np.ndarray]:
    """
    이미지, 세그멘테이션, landmark를 이용해 여러 조건 이미지를 생성한다.

    Args:
        image: 원본 이미지 (PIL.Image, RGB)
        seg: 세그멘테이션 이미지 (PIL.Image, RGB 색으로 class 구분)
        mask: 추가 마스크 (현재 로직에서는 사용하지 않지만 인터페이스 유지용)
        landmark: 랜드마크 visualization 이미지 (PIL.Image, non-zero 픽셀이 landmark)
        attributes: class_name -> index 매핑 딕셔너리
        color_list: 각 index에 대응하는 (R, G, B) 튜플 리스트
        blur_radius: Gaussian blur radius
        remove_targets: seg_landmark에서 제거할 attribute 리스트
        glass_target: 안경(eye glasses)에 해당하는 attribute 이름
        skin_targets: blur를 덮어씌울 attribute 리스트 (None이면 attributes 전체 사용)

    Returns:
        Dict[str, np.ndarray]: 아래 키를 갖는 H×W×3 uint8 이미지들
            - "image": 원본
            - "condition_blur"
            - "condition_blur_landmark"
            - "condition_blur_landmark_glass"
            - "condition_seg_landmark"
            - "condition_segSelected_landmark"
            - "condition_blur_segSelected_landmark"
            - "blended_image"
            - "seg_landmark"
            - "seg_landmark_selected"
    """
    global attributes, color_list
    
    # ---------------------------
    # 0. 기본 파라미터 정리
    # ---------------------------
    if remove_targets is None:
        remove_targets = ["skin", "l_ear", "r_ear"]

    if skin_targets is None:
        skin_targets = list(attributes.keys())

    # ---------------------------
    # 1. PIL -> numpy 변환
    # ---------------------------
    image_np = np.array(image)          # (H, W, 3)
    seg_np = np.array(seg)              # (H, W, 3)
    landmark_np = np.array(landmark)    # (H, W, 3)
    iris_landmark_np = np.array(iris)    # (H, W, 3)


    # ---------------------------
    # 2. Downsample->Upsample 이미지 생성
    # ---------------------------
    if condition == 'blur':
        # Gaussian blur 적용 (radius는 원하는 정도로 조절)
        blurred_pil = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_np = np.array(blurred_pil)
    elif condition == 'downsample':
        w, h = image.size
        blurred_np = image.resize((downsample_size, downsample_size), Image.LANCZOS).resize((w, h), Image.LANCZOS)
        blurred_np = np.array(blurred_np)
    elif condition == 'None':
        blurred_np = image_np
    elif condition == 'mask':
        # 그냥 검은 이미지
        blurred_np = np.zeros_like(image_np)
    else:
        raise ValueError(f"Unknown condition type: {condition}")

    # ---------------------------
    # 3. landmark 마스크 생성
    #    - landmark 이미지에서 non-zero 픽셀 위치를 landmark로 봄
    # ---------------------------
    landmark_sum = np.sum(landmark_np, axis=-1)        # (H, W)
    landmark_mask = landmark_sum > 0                   # (H, W)
    landmark_mask_3c = np.repeat(landmark_mask[..., None], 3, axis=-1)  # (H, W, 3)

    # seg + landmark (landmark 영역은 흰색으로)
    seg_landmark = deepcopy(seg_np)
    seg_landmark[landmark_mask_3c] = 255
    
    iris_landmark_idx = iris_landmark_np.sum(axis=-1) > 0 # shape (h, w)


    # ---------------------------
    # 4. seg_landmark_selected: 특정 attribute 제거한 버전
    # ---------------------------
    seg_landmark_selected = deepcopy(seg_landmark)
    for target in remove_targets:
        if target not in attributes:
            continue
        target_idx = attributes[target]
        target_color = color_list[target_idx]  # (R, G, B)
        target_mask = (
            (seg_landmark_selected[..., 0] == target_color[0]) &
            (seg_landmark_selected[..., 1] == target_color[1]) &
            (seg_landmark_selected[..., 2] == target_color[2])
        )  # (H, W) bool
        seg_landmark_selected[target_mask] = 0

    # ---------------------------
    # 5. 안경(seg_glass) 영역 추출
    # ---------------------------
    seg_glass = np.zeros_like(seg_np)
    if glass_target in attributes:
        g_idx = attributes[glass_target]
        g_color = color_list[g_idx]
        glass_mask_single = (
            (seg_np[..., 0] == g_color[0]) &
            (seg_np[..., 1] == g_color[1]) &
            (seg_np[..., 2] == g_color[2])
        )  # (H, W) bool
        seg_glass[glass_mask_single] = g_color
        glass_mask = np.repeat(glass_mask_single[..., None], 3, axis=-1)  # (H, W, 3)
    else:
        # glass_target이 없으면 전체 False
        glass_mask_single = np.zeros(seg_np.shape[:2], dtype=bool)
        glass_mask = np.repeat(glass_mask_single[..., None], 3, axis=-1)

    # ---------------------------
    # 6. skin(mask_skin) 영역 구하기
    #    - skin_targets에 해당하는 모든 class를 하나의 mask로 합침
    # ---------------------------
    seg_skin = np.zeros_like(seg_np)
    for t in skin_targets:
        if t not in attributes:
            continue
        t_idx = attributes[t]
        t_color = color_list[t_idx]
        t_mask = (
            (seg_np[..., 0] == t_color[0]) &
            (seg_np[..., 1] == t_color[1]) &
            (seg_np[..., 2] == t_color[2])
        )
        seg_skin[t_mask] = t_color

    # non-zero 픽셀을 skin 영역으로
    mask_skin_single = (
        (seg_skin[..., 0] != 0) |
        (seg_skin[..., 1] != 0) |
        (seg_skin[..., 2] != 0)
    )  # (H, W) bool
    mask_skin = np.repeat(mask_skin_single[..., None], 3, axis=-1)  # (H, W, 3)
    
    if mask is not None:
        mask_np = np.array(mask)        # (H, W, 3)
        mask_bool = (mask_np.sum(axis=-1) > 0)   # (H, W) bool
        mask_bool_3c = np.repeat(mask_bool[..., None], 3, axis=-1)
        mask_skin = mask_bool_3c
    

    # ---------------------------
    # 7. condition_blur: skin 영역만 blur 덮어씌운 이미지
    # ---------------------------
    condition_blur = deepcopy(image_np)
    condition_blur = condition_blur * (~mask_skin) + blurred_np * mask_skin
    condition_blur = condition_blur.astype(np.uint8)

    # ---------------------------
    # 8. condition_blur_landmark: blur + landmark 영역(흰색)
    # ---------------------------
    condition_blur_landmark = deepcopy(condition_blur)
    condition_blur_landmark[landmark_mask_3c] = 255
    condition_blur_landmark[iris_landmark_idx] = [255, 0, 0]
    condition_blur_landmark = condition_blur_landmark.astype(np.uint8)

    # ---------------------------
    # 9. condition_blur_landmark_glass:
    #    blur+landmark 이미지에 안경 영역을 살짝 overlay
    # ---------------------------
    glass_overlay = cv2.addWeighted(
        condition_blur_landmark, 0.9,
        seg_glass, 0.1,
        0
    )
    condition_blur_landmark_glass = (
        condition_blur_landmark * (~glass_mask) +
        glass_overlay * glass_mask
    )
    condition_blur_landmark_glass = condition_blur_landmark_glass.astype(np.uint8)

    # ---------------------------
    # 10. condition_seg_landmark:
    #     원본 이미지에 seg_landmark를 해당 영역에만 덮어씌운 이미지
    # ---------------------------
    mask_for_seg = np.any(seg_landmark != 0, axis=-1)  # (H, W) bool
    mask_for_seg_3c = np.repeat(mask_for_seg[..., None], 3, axis=-1)

    condition_seg_landmark = deepcopy(image_np)
    condition_seg_landmark = (
        condition_seg_landmark * (~mask_for_seg_3c) +
        seg_landmark * mask_for_seg_3c
    )
    condition_seg_landmark = condition_seg_landmark.astype(np.uint8)

    # ---------------------------
    # 11. condition_segSelected_landmark:
    #      제거된 attribute가 빠진 seg_landmark_selected 사용
    # ---------------------------
    mask_for_seg_selected = np.any(seg_landmark_selected != 0, axis=-1)
    mask_for_seg_selected_3c = np.repeat(mask_for_seg_selected[..., None], 3, axis=-1)

    condition_segSelected_landmark = deepcopy(image_np)
    condition_segSelected_landmark = (
        condition_segSelected_landmark * (~mask_for_seg_selected_3c) +
        seg_landmark_selected * mask_for_seg_selected_3c
    )
    condition_segSelected_landmark = condition_segSelected_landmark.astype(np.uint8)

    # ---------------------------
    # 12. condition_blur_segSelected_landmark:
    #      blur + 선택된 seg_landmark_selected 합성
    # ---------------------------
    condition_blur_segSelected_landmark = deepcopy(image_np)
    condition_blur_segSelected_landmark = (
        condition_blur_segSelected_landmark * (~mask_skin) + blurred_np * mask_skin
    )
    condition_blur_segSelected_landmark = (
        condition_blur_segSelected_landmark * (~mask_for_seg_selected_3c) +
        seg_landmark_selected * mask_for_seg_selected_3c
    )
    condition_blur_segSelected_landmark = condition_blur_segSelected_landmark.astype(np.uint8)

    # ---------------------------
    # 13. blended_image:
    #      blur+landmark 이미지와 segSelected 버전을 blend
    # ---------------------------
    blended_image = cv2.addWeighted(
        condition_blur_landmark, 0.8,
        condition_blur_segSelected_landmark, 0.2,
        0
    )
    blended_image = blended_image.astype(np.uint8)

    # ---------------------------
    # 14. 결과 모아서 반환
    # ---------------------------
    results = {
        "image": image_np,
        "condition_blur": condition_blur,
        "condition_blur_landmark": condition_blur_landmark,
        "condition_blur_landmark_glass": condition_blur_landmark_glass,
        "condition_seg_landmark": condition_seg_landmark,
        "condition_segSelected_landmark": condition_segSelected_landmark,
        "condition_blur_segSelected_landmark": condition_blur_segSelected_landmark,
        "blended_image": blended_image,
        "seg_landmark": seg_landmark,
        "seg_landmark_selected": seg_landmark_selected,
    }

    return results