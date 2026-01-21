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

For instructions in Korean, please refer to [README_kor.md](README_kor.md).

- [1. Project Overview](#1-project-overview)
- [2. Installation](#2-installation)
  - [2.1. Requirements](#21-requirements)
  - [2.2. Clone Repository](#22-clone-repository)
  - [2.3. Conda Environment Setup](#23-conda-environment-setup)
  - [2.4. Dataset Preparation](#24-dataset-preparation)
- [3. Training (Teacher Model)](#3-training-teacher-model)
  - [3.1. Data Preprocessing](#31-data-preprocessing)
    - [3.1.1. 3DMM Landmark Extraction](#311-3dmm-landmark-extraction)
    - [3.1.2. Gaze Landmark Extraction](#312-gaze-landmark-extraction)
    - [3.1.3. Final Condition Image Generation](#313-final-condition-image-generation)
  - [3.2. Model Training](#32-model-training)
- [4. Inference (Teacher Model)](#4-inference-teacher-model)
  - [4.1. FFHQ Dataset Inference](#41-ffhq-dataset-inference)
  - [4.2. Pseudo Dataset Generation](#42-pseudo-dataset-generation)
- [5. Inference (Student Model)](#5-inference-student-model)

## 1. Project Overview

This document aims to explain the training and inference process of the Diffusion Model (Teacher Model).

## 2. Installation

### 2.1. Requirements

- NVIDIA GPU
- Anaconda (Conda)

### 2.2. Clone Repository

```bash
git clone https://github.com/your-repo/fluxswap.git
cd fluxswap
```

> **Note**: `<PROJECT_ROOT>` refers to the absolute path of this `fluxswap` directory.

### 2.3. Conda Environment Setup

This project uses three Conda environments: `3DDFA_env`, `mediapipe`, and `faceswap_omini`.

**1. 3DDFA_env**
- [Original Github](https://github.com/wang-zidu/3DDFA-V3)
- Used for 3DMM Landmark extraction.
- Please follow the [instructions on the original Github](https://github.com/wang-zidu/3DDFA-V3/tree/main/assets) to install the 3DDFA checkpoints and environment.

**2. mediapipe**
- [Original Github](https://github.com/Morris88826/MediaPipe_Iris)
- Used for Gaze Landmark extraction.

```bash
conda env create --file preprocess/mediapipe.yaml
```

**3. faceswap_omini**
- Used for final condition image generation, model training, and inference.

```bash
conda env create --file preprocess/faceswap_omini.yaml
conda activate faceswap_omini

# Install mmcv and mmsegmentation
pip install -e preprocess/mmcv
pip install -e preprocess/mmsegmentation
```

### 2.4. Dataset Preparation

- **VGGFace2-HQ**: The main dataset used for training.
  - This document assumes the dataset is stored at a specific path (e.g., `<VGGFACE2_HQ_PATH>`).
- **FFHQ**: Used for evaluation.
  - The FFHQ dataset consists of `src` and `trg` folders, each having a preprocessing structure similar to VGGFace2-HQ. (See [4.1. FFHQ Dataset Inference](#41-ffhq-dataset-inference) for detailed structure).

## 3. Training (Teacher Model)

### 3.0. VGGFace2-HQ Data Download
- Use the dataset uploaded to [HuggingFace](https://huggingface.co/datasets/RichardErkhov/VGGFace2-HQ/tree/main).
- Decompress the `original` folder and use the data.

### 3.1. Data Preprocessing

The VGGFace2-HQ dataset undergoes a total of 3 preprocessing steps.

#### 3.1.1. 3DMM Landmark Extraction

- **Conda Environment**: `3DDFA provided Conda`
- **File to Modify**: `<PROJECT_ROOT>/preprocess/3DDFA-V3/demo_from_folder_jiwon_vgg.py`
  - `line 24`: Modify to the VGGFace2-HQ dataset path (`<VGGFACE2_HQ_PATH>`).
- **Execution**:
  - Single GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/3DDFA-V3/
    ./run_vgg.sh
    ```
  - Multi-GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/3DDFA-V3/
    ./run_vgg_multigpu.sh
    ```
- **Result**: Saved in `<VGGFACE2_HQ_PATH>/3dmm/` folder.

#### 3.1.2. Gaze Landmark Extraction

- **Conda Environment**: `mediapipe`
- **File to Modify**: `<PROJECT_ROOT>/preprocess/MediaPipe_Iris/inference.py`
  - `line 34`, `dataset_path`: Modify to the VGGFace2-HQ dataset path (`<VGGFACE2_HQ_PATH>`).
- **Execution**:
  - Single GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/MediaPipe_Iris/
    ./inference.sh
    ```
  - Multi-GPU:
    ```bash
    cd <PROJECT_ROOT>/preprocess/MediaPipe_Iris/
    ./inference_torchrun.sh
    ```
- **Result**: Saved in `<VGGFACE2_HQ_PATH>/iris/` folder.

#### 3.1.3. Final Condition Image Generation

- **Conda Environment**: `faceswap_omini`
- **File to Modify**: `<PROJECT_ROOT>/preprocess/vgg_preprocess_seg_mask_gaze_multigpu_samsung.py`
  - `line 73`, `image_folder_path`: Modify to the VGGFace2-HQ dataset path (`<VGGFACE2_HQ_PATH>`).
- **Execution**:
  ```bash
  # Activate faceswap_omini environment
  conda activate faceswap_omini
  # Run script
  python <PROJECT_ROOT>/preprocess/vgg_preprocess_seg_mask_gaze_multigpu_samsung.py
  ```
- **Result**: Saved in `<VGGFACE2_HQ_PATH>/condition_blended_image_blurdownsample8_segGlass_landmark_iris` folder.


#### 3.1.4. Dataset Filtering

- Calculate scores using LAION Aesthetics for VGGFace2-HQ images in advance and use them for data filtering.
- You can generate the `score.json` file with `<PROJECT_ROOT>/preprocess/vgg_preprocess_score_multigpu.py`.
- An example file used is `<PROJECT_ROOT>/preprocess/score.json`.

### 3.2. Model Training

- **Conda Environment**: `faceswap_omini`
- **Config File**: `<PROJECT_ROOT>/train/config/baseline_vgg_0.35.yaml`
  - `netarc_path`: Modify to the Arc2Face model path to be used.
  - `dataset_path`: Modify to the VGGFace2-HQ dataset path (`<VGGFACE2_HQ_PATH>`).
- **Execution**:
  ```bash
  cd <PROJECT_ROOT>/train/script
  ./baseline_vgg.sh
  ```

## 4. Inference (Teacher Model)

- **Conda Environment**: `faceswap_omini`
- **Checkpoint Used (Example)**: `<PROJECT_ROOT>/checkpoints/teacher`

### 4.1. FFHQ Dataset Inference

Example of inference on the FFHQ evaluation dataset.

- **`base_path`**: Project root path (`<PROJECT_ROOT>`)
- **`ffhq_base_path`**: Preprocessed FFHQ dataset path. Assumes the following structure:
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
- **`id_guidance_scale`**: Higher settings increase ID identity reflection but may decrease attribute preservation. (Minimum value: 1.0)

**Without Inversion**

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

**With Inversion**

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

### 4.2. Pseudo Dataset Generation

Generate a pseudo dataset based on VGGFace2-HQ.
The VGGFace2-HQ dataset must be preprocessed.

- **Execution**:
  - Run the `<PROJECT_ROOT>/pulid_omini_dataset_gen_fluxpseudovgg_multigpu.sh` shell script.
  - `line 34`, `lora_file_path`: You can set the checkpoint path to be used within the script.

## 5. Inference (Student Model)

- **Conda Environment**: `faceswap_omini`
- **Checkpoint Used (Example)**: `<PROJECT_ROOT>/checkpoints/student`

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 pulid_omini_inference_ffhq_args_multigpu.py \
    --base_path <PROJECT_ROOT> \
    --ffhq_base_path <FFHQ_BASE_PATH> \
    --checkpoint_path <PROJECT_ROOT>/checkpoints/student \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
```
