<div align="center">
  <h1>Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping</h1>

  <p>
    <a href="https://scholar.google.com/citations?user=A2PurdIAAAAJ&hl=en">Jiwon Kang<sup>1</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=IWvP0A4AAAAJ&hl=en">Yeji Choi<sup>1</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=0H3dcPoAAAAJ&hl=en">JoungBin Lee<sup>1</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=7cyLEQ0AAAAJ&hl=en/">Wooseok Jang<sup>1</sup></a>
    ·
    Jinhyeok Choi<sup>1</sup>
    ·
    Taekeun Kang<sup>2</sup>
    ·
    Yongjae Park<sup>2</sup>
    ·
    Myungin Kim<sup>2</sup>
    ·
    <a href="https://cvlab-kaist.ac.kr/">Seungryong Kim<sup>1</sup></a>
  </p>

  <p>
    <sup>1</sup>KAIST AI &nbsp; <sup>2</sup>SAMSUNG
  </p>

  <h3><a href="#">Paper</a> | <a href="https://cvlab-kaist.github.io/APPLE/">Project Page</a></h3>

  <img src="teaser.webp" width="100%">
</div>

## Abstract
Face swapping aims to transfer the identity of a source face onto a target face while preserving target-specific attributes such as pose, expression, lighting, skin tone, and makeup. However, since real ground truth for face swapping is unavailable, achieving both accurate identity transfer and high-quality attribute preservation remains challenging. Recent diffusion-based approaches attempt to improve visual fidelity through conditional inpainting on masked target images, but the masked condition removes crucial appearance cues, resulting in plausible yet misaligned attributes due to the lack of explicit supervision. To address these limitations, we propose **APPLE** (**A**ttribute-**P**reserving **P**seudo-**L**ab**e**ling), a diffusion-based teacher–student framework that enhances attribute fidelity through attribute-aware pseudo-label supervision. We reformulate face swapping as a conditional deblurring task to more faithfully preserve target-specific attributes such as lighting, skin tone, and makeup. In addition, we introduce an attribute-aware inversion scheme to further improve detailed attribute preservation. Through an elaborate attribute-preserving design for teacher learning, **APPLE** produces high-quality pseudo triplets that explicitly provide the student with direct face-swapping supervision. Overall, **APPLE** achieves state-of-the-art performance in terms of attribute preservation and identity transfer, producing more photorealistic and target-faithful results.


- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 설치](#2-설치)
  - [2.1. 요구사항](#21-요구사항)
  - [2.2. 저장소 복제](#22-저장소-복제)
  - [2.3. Conda 환경 설정](#23-conda-환경-설정)
  - [2.4. 데이터셋 준비](#24-데이터셋-준비)
- [3. 훈련 (Teacher Model)](#3-훈련-teacher-model)
  - [3.1. 데이터 전처리](#31-데이터-전처리)
    - [3.1.1. 3DMM Landmark 추출](#311-3dmm-landmark-추출)
    - [3.1.2. Gaze Landmark 추출](#312-gaze-landmark-추출)
    - [3.1.3. 최종 조건 이미지 생성](#313-최종-조건-이미지-생성)
  - [3.2. 모델 훈련](#32-모델-훈련)
- [4. 추론 (Teacher Model)](#4-추론-teacher-model)
  - [4.1. FFHQ 데이터셋 추론](#41-ffhq-데이터셋-추론)
  - [4.2. 수도 데이터셋 생성](#42-수도-데이터셋-생성)
- [5. 추론 (Student Model)](#5-추론-student-model)

## 1. 프로젝트 개요

이 문서는 Diffusion Model (Teacher Model)의 훈련 및 추론 과정을 설명을 목표로 함.

## 2. 설치

### 2.1. 요구사항

- NVIDIA GPU
- Anaconda (Conda)

### 2.2. 저장소 복제

```bash
git clone https://github.com/your-repo/fluxswap.git
cd fluxswap
```

> **참고**: `<PROJECT_ROOT>`는 이 `fluxswap` 디렉토리의 절대 경로를 의미합니다.

### 2.3. Conda 환경 설정

이 프로젝트는 `3DDFA_env`, `mediapipe`, `faceswap_omini` 세 가지 Conda 환경을 사용합니다.

**1. 3DDFA_env**
- [원본 Github](https://github.com/wang-zidu/3DDFA-V3)
- 3DMM Landmark 추출에 사용됩니다.
- 3DDFA의 체크포인트 및 환경은, [원본 Github의 안내](https://github.com/wang-zidu/3DDFA-V3/tree/main/assets)를 따라 설치해주세요.

**2. mediapipe**
- [원본 Github](https://github.com/Morris88826/MediaPipe_Iris)
- Gaze Landmark 추출에 사용됩니다.

```bash
conda env create --file preprocess/mediapipe.yaml
```

**3. faceswap_omini**
- 최종 조건 이미지 생성, 모델 훈련 및 추론에 사용됩니다.

```bash
conda env create --file preprocess/faceswap_omini.yaml
conda activate faceswap_omini

# mmcv 및 mmsegmentation 설치
pip install -e preprocess/mmcv
pip install -e preprocess/mmsegmentation
```

### 2.4. 데이터셋 준비

- **VGGFace2-HQ**: 훈련에 사용되는 주요 데이터셋입니다.
  - 이 문서에서는 데이터셋이 특정 경로(예: `<VGGFACE2_HQ_PATH>`)에 저장되어 있다고 가정합니다.
- **FFHQ**: 평가(Evaluation)에 사용됩니다.
  - FFHQ 데이터셋은 `src`와 `trg` 폴더로 구성되며, 각 폴더는 VGGFace2-HQ와 유사한 전처리 구조를 가집니다. (자세한 구조는 [4.1. FFHQ 데이터셋 추론](#41-ffhq-데이터셋-추론) 참고)

## 3. 훈련 (Teacher Model)

### 3.0. VGGFace2-HQ 데이터 다운로드
- [HuggingFace](https://huggingface.co/datasets/RichardErkhov/VGGFace2-HQ/tree/main)에 업로드되어 있는 데이터셋을 사용합니다.
- `original` 폴더의 압축을 해제한 데이터를 사용합니다.

### 3.1. 데이터 전처리

VGGFace2-HQ 데이터셋을 사용하여 총 3단계의 전처리 과정을 거칩니다.

#### 3.1.1. 3DMM Landmark 추출

- **Conda 환경**: `3DDFA 제공 Conda`
- **수정 파일**: `<PROJECT_ROOT>/preprocess/3DDFA-V3/demo_from_folder_jiwon_vgg.py`
  - `line 24`: VGGFace2-HQ 데이터셋 경로 ( `<VGGFACE2_HQ_PATH>` )로 수정해야 합니다.
- **실행**:
  - 단일 GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/3DDFA-V3/
    ./run_vgg.sh
    ```
  - 다중 GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/3DDFA-V3/
    ./run_vgg_multigpu.sh
    ```
- **결과**: `<VGGFACE2_HQ_PATH>/3dmm/` 폴더에 저장됩니다.

#### 3.1.2. Gaze Landmark 추출

- **Conda 환경**: `mediapipe`
- **수정 파일**: `<PROJECT_ROOT>/preprocess/MediaPipe_Iris/inference.py`
  - `line 34`, `dataset_path`: VGGFace2-HQ 데이터셋 경로 ( `<VGGFACE2_HQ_PATH>` )로 수정해야 합니다.
- **실행**:
  - 단일 GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/MediaPipe_Iris/
    ./inference.sh
    ```
  - 다중 GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/MediaPipe_Iris/
    ./inference_torchrun.sh
    ```
- **결과**: `<VGGFACE2_HQ_PATH>/iris/` 폴더에 저장됩니다.

#### 3.1.3. 최종 조건 이미지 생성

- **Conda 환경**: `faceswap_omini`
- **수정 파일**: `<PROJECT_ROOT>/preprocess/vgg_preprocess_seg_mask_gaze_multigpu_samsung.py`
  - `line 73`, `image_folder_path`: VGGFace2-HQ 데이터셋 경로 ( `<VGGFACE2_HQ_PATH>` )로 수정해야 합니다.
- **실행**:
  ```bash
  # faceswap_omini 환경 활성화
  conda activate faceswap_omini
  # 스크립트 실행 
  python <PROJECT_ROOT>/preprocess/vgg_preprocess_seg_mask_gaze_multigpu_samsung.py
  ```
- **결과**: `<VGGFACE2_HQ_PATH>/condition_blended_image_blurdownsample8_segGlass_landmark_iris` 폴더에 저장됩니다.


#### 3.1.4. 데이터셋 필터링

- VGGFace2-HQ 이미지들에 대해 미리 LAION Aesthetics를 이용해 Score를 계산하고, 데이터 필터링에 활용합니다.
- `<PROJECT_ROOT>/preprocess/vgg_preprocess_score_multigpu.py` 파일로 `score.json` 파일 생성이 가능합니다.
- 저희가 사용한 예시 파일은 `<PROJECT_ROOT>/preprocess/score.json` 입니다.

### 3.2. 모델 훈련

- **Conda 환경**: `faceswap_omini`
- **설정 파일**: `<PROJECT_ROOT>/train/config/baseline_vgg_0.35.yaml`
  - `netarc_path`: 사용할 Arc2Face 모델 경로로 수정해야 합니다.
  - `dataset_path`: VGGFace2-HQ 데이터셋 경로 ( `<VGGFACE2_HQ_PATH>` )로 수정해야 합니다.
- **실행**:
  ```bash
  cd <PROJECT_ROOT>/train/script
  ./baseline_vgg.sh
  ```

## 4. 추론 (Teacher Model)

- **Conda 환경**: `faceswap_omini`
- **사용 체크포인트 (예시)**: `<PROJECT_ROOT>/checkpoints/teacher`

### 4.1. FFHQ 데이터셋 추론

평가용 FFHQ 데이터셋에 대한 추론 예시입니다.

- **`base_path`**: 프로젝트 루트 경로 ( `<PROJECT_ROOT>` )
- **`ffhq_base_path`**: 전처리된 FFHQ 데이터셋 경로. 아래와 같은 구조를 가정합니다.
  ```
  <FFHQ_BASE_PATH>/
  ├── src
  │   ├── 3dmm
  │   ├── condition_...
  │   ├── ...
  │   └── 000000.jpg
  └── trg
      ├── 3dmm
      ├── condition_...
      │   ...
      └── 000000.jpg
  ```
- **`id_guidance_scale`**: 높게 설정할수록 ID 정체성(identity) 반영률이 높아지지만, 속성(attribute) 보존율은 감소할 수 있습니다. (최소값: 1.0)

**Inversion 미사용 시**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 pulid_omini_inference_ffhq_args_multigpu.py \
    --base_path <PROJECT_ROOT> \
    --ffhq_base_path <FFHQ_BASE_PATH> \
    --checkpoint_path <PROJECT_ROOT>/checkpoints/teacher \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'
```

**Inversion 사용 시**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pulid_omini_inference_ffhq_inversion_args_multigpu.py \
    --base_path <PROJECT_ROOT> \
    --ffhq_base_path <FFHQ_BASE_PATH> \
    --checkpoint_path <PROJECT_ROOT>/checkpoints/teacher \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'
```

### 4.2. 수도 데이터셋 생성

VGGFace2-HQ 기반으로 수도(pseudo) 데이터셋을 생성합니다.
VGGFace2-HQ 데이터셋은 전처리가 되어있어야 합니다.

- **실행**:
  - `<PROJECT_ROOT>/pulid_omini_dataset_gen_fluxpseudovgg_multigpu.sh` 쉘 스크립트를 실행합니다.
  - `line 34`, `lora_file_path`: 스크립트 내에서 사용할 체크포인트 경로를 설정할 수 있습니다.

## 5. 추론 (Student Model)

- **Conda 환경**: `faceswap_omini`
- **사용 체크포인트 (예시)**: `<PROJECT_ROOT>/checkpoints/student`

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 pulid_omini_inference_ffhq_args_multigpu.py \
    --base_path <PROJECT_ROOT> \
    --ffhq_base_path <FFHQ_BASE_PATH> \
    --checkpoint_path <PROJECT_ROOT>/checkpoints/student \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
```