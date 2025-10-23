import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
import torch
import torch.distributed as dist
from transformer_flux_ca import FluxTransformer2DModelCA
# from diffusers import FluxTransformer2DModel
import random
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image

def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)


# --- Distributed setup ---
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = f'cuda:{local_rank}'
# -------------------------

guidance_scale=1.0
image_guidance_scale = 1.0
id_guidance_scale = 1.0
use_gaze = False


weight_dtype = torch.bfloat16
# transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer')
# transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None)
transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None, use_gaze=use_gaze, local_rank=local_rank)
# no grad
torch.set_grad_enabled(False)

from pulid.utils import resize_numpy_image_long
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_txt
from diffusers import FluxPipeline
from pipeline_flux_ca import FluxPipelineCA
from omini.pipeline.flux_omini import generate_ca, Condition, convert_to_condition


flux = FluxPipelineCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", transformer=transformer, torch_dtype=weight_dtype).to(device)


# ckpt = 43000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_scratch_lora64_20251004-005548/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}/ffhq_eval'

# ckpt = 60000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_20251010-115546/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 8000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_20251014-014645/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_idLoss_t<=0.33_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 8000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_t<=0.5_20251014-021510/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_idLoss_t<=0.5_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

ckpt = 80000
lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_20251014-162532/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'
output_dir = f'/mnt/data6/vgg_swapped/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/'

# ckpt = 8000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt60000_gaze_20251018-024629/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt60000_gaze_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

adapter_name = 'default'
if rank == 0:
    print(f"Loading LoRA for adapter '{adapter_name}' from {lora_file_path}")
flux.load_lora_weights(lora_file_path, adapter_name=adapter_name)
if use_gaze:
    flux.transformer.gaze_ca.load_state_dict(torch.load(
        os.path.join(os.path.dirname(lora_file_path), 'gaze_ca.pth')    
    ))
    if rank == 0:
        print(f"[INFO] Loaded gaze CA weights from {os.path.join(os.path.dirname(lora_file_path), 'gaze_ca.pth')}")

flux.transformer.components_to_device(device, weight_dtype)
flux.set_progress_bar_config(disable=True)

# Compile the model with torch.compile for faster inference
# flux.transformer = torch.compile(flux.transformer, mode="reduce-overhead", fullgraph=True)


os.makedirs(output_dir, exist_ok=True)

import os, json, random, csv
from pathlib import Path
from collections import defaultdict
from natsort import natsorted


import json
random.seed(42)

dataset_path = "/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan"
json_path = os.path.join(dataset_path, "score.json")
path_pairs_path = os.path.join(dataset_path, "pairs.pt")

if not os.path.exists(path_pairs_path):
    # 1) AES 필터링된 이미지 리스트 만들기
    with open(json_path, "r") as f:
        score_dict = json.load(f)

    training_base_list = set(os.listdir(dataset_path))  # 폴더 이름들 (n000001 ...)
    high_aes_keys = [k for k, v in score_dict.items() if v.get("aes", -1) > 5.5]

    # 키 형식이 "n000008/0145_01" 같은 경우를 가정해 .jpg 경로 구성
    img_list = []
    for k in high_aes_keys:
        rel = k + ".jpg"  # e.g., "n000008/0145_01.jpg"
        full = os.path.join(dataset_path, rel)
        # 폴더가 실제로 존재하고 파일도 존재하는 경우만
        if os.path.exists(full) and rel.split("/")[0] in training_base_list:
            img_list.append(full)

    print(f"AES>5.5 통과 이미지: {len(img_list)}")

    # 2) {id: [imgs]} 생성 (id는 폴더명)
    id_to_images = defaultdict(list)
    for p in img_list:
        id_str = os.path.basename(os.path.dirname(p))  # n000008
        id_to_images[id_str].append(p)

    # 3) 각 id 내부 정렬 + 빈 id 제거
    for k in list(id_to_images.keys()):
        id_to_images[k] = natsorted(id_to_images[k])
        if len(id_to_images[k]) == 0:
            del id_to_images[k]

    print(f"사용 가능한 ID 수: {len(id_to_images)}")

    # 4) 균등 id 샘플링으로 pair 만들기 (src_id != trg_id)
    ids = natsorted(id_to_images.keys())
    N = len(ids)
    assert N > 1

    total_length = 35_000
    pairs = set()  # ← 중복 방지용 set

    rng = random.Random(42)
    while len(pairs) < total_length:
        si = rng.randrange(N)
        ti = rng.randrange(N - 1)
        if ti >= si:
            ti += 1

        src_id, trg_id = ids[si], ids[ti]
        src_path = rng.choice(id_to_images[src_id])
        trg_path = rng.choice(id_to_images[trg_id])
        pair = (src_path, trg_path)

        if pair not in pairs:
            pairs.add(pair)

    path_pairs = natsorted(list(pairs))
    print(f"✅ {len(path_pairs)} unique pairs generated (no duplicates)")
    torch.save(path_pairs, path_pairs_path)
    print(f"Saved pairs to {path_pairs_path}")
else :
    print(f"Loading existing pairs from {path_pairs_path}")
    path_pairs = torch.load(path_pairs_path)

# import time
# cur_time_for_rng = int(time.time())  # ✅ 예: 1712345678
# rng = random.Random(42 + cur_time_for_rng)
# rng.shuffle(path_pairs)

# Distribute pairs
distributed_pairs = path_pairs[rank::world_size]

# Unzip for the loop
if distributed_pairs:
    src_img_path_list, trg_img_path_list = zip(*distributed_pairs)
else:
    src_img_path_list, trg_img_path_list = [], []


for src_img_path, trg_img_path in tqdm(zip(src_img_path_list, trg_img_path_list), desc=f'GPU {rank} Processing', total=len(src_img_path_list), disable=(rank!=0)):
    try :
        prompt="a photo of human face",
        neg_prompt = ""
        true_cfg = 1.0
        use_true_cfg = True if true_cfg > 1.0 else False

        src_num = os.path.basename(src_img_path).split('.')[0]
        src_id = os.path.basename(os.path.dirname(src_img_path))
        trg_num = os.path.basename(trg_img_path).split('.')[0]
        trg_id = os.path.basename(os.path.dirname(trg_img_path))
        # Continue if exists
        img_save_fname = f"{output_dir}/{src_id}_{src_num}_{trg_id}_{trg_num}.png"
        grid_save_fname = f"{output_dir}/grid/{src_id}_{src_num}_{trg_id}_{trg_num}_grid.png"
        if os.path.exists(img_save_fname):
            print(f"Image {img_save_fname} already exists. Skipping...")
            continue

        trg_img_path_base = os.path.dirname(trg_img_path)
        trg_img_path = os.path.join (trg_img_path_base, 'condition_blended_image_blurdownsample8_segGlass_landmark', f"{trg_num}.png")
        condition_img = Image.open(trg_img_path).convert('RGB')


        # 시간 재기
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        if use_gaze:
            import numpy as np
            gaze_path = os.path.join (trg_img_path_base, 'gaze', f"{trg_num}.npy")
            gaze_embed = torch.from_numpy(  np.load(gaze_path) ).unsqueeze(0).to(device, dtype=weight_dtype) # (1, gaze_dim)
        else:
            gaze_embed = None
        
        # Resize
        condition_img = condition_img.resize((512,512))
        condition_type = 'deblurring'
        position_delta = [0,0]
        position_scale = 1.0
        condition = Condition(condition_img, 'default', position_delta, position_scale)

        # Read ID from source image
        id_image = cv2.imread(src_img_path)
        # print(f"id_image shape: {id_image.shape}")
        id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
        id_image = resize_numpy_image_long(id_image, 1024)
        id_image_pil = Image.fromarray(id_image)
        id_image_pil = id_image_pil.resize((512,512))
        # id_image_pil.save(f"{output_dir}/{src_num}_id.png")


        # flux.t5, flux.clip = flux.t5.to(device), flux.clip.to(device)
        # flux.pulid_model.components_to_device(device)
        id_embeddings, uncond_id_embeddings = flux.transformer.get_id_embedding(id_image, cal_uncond=True)
        id_embeddings = id_embeddings.to(device, dtype=weight_dtype)
        # if use_true_cfg:
        uncond_id_embeddings = uncond_id_embeddings.to(device, dtype=weight_dtype)

        # Nan check
        if torch.isnan(id_embeddings).any():
            print(f"WARNING: NaN detected in id_embeddings for {src_img_path}. Skipping.")
            continue

        # inp = prepare_txt(t5=flux.t5, clip=flux.clip, prompt=prompt, device=device)
        # inp_neg = prepare_txt(t5=flux.t5, clip=flux.clip,  prompt=neg_prompt, device=device) if use_true_cfg else None
        # flux.t5, flux.clip = flux.t5.cpu(), flux.clip.cpu()

        from diffusers.utils import make_image_grid
        # generate image
        # prompt = "a photo of human face"
        # negative_prompt = ""

        # prompt = "photo of a woman in red dress in a garden"
        # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
        prompt = 'a photo of human face'
        negative_prompt = ''

        image_list = []

        img = generate_ca(
                flux,
                prompt=prompt,
                conditions=[condition],
                height=512,
                width=512,
                generator=torch.Generator('cpu').manual_seed(0),
                kv_cache=False,
                id_embed=id_embeddings,
                uncond_id_embed=uncond_id_embeddings,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                id_guidance_scale=id_guidance_scale,
                gaze_embed=gaze_embed,
            )

        # img = flux(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     width=512,
        #     height=512,
        #     num_inference_steps=30,
        #     guidance_scale=g,
        #     generator=torch.Generator('cpu').manual_seed(0),
        #     id_embed=id_embeddings,
        #     uncond_id_embed=uncond_id_embeddings,
        # )
        print(f"GPU {rank} :  Saving image to {img_save_fname}")
        image = img.images[0] if isinstance(img.images, list) else img.images
        image.save(img_save_fname)
        # condition_img.save(f"{output_dir}/{src_num}_cond.png")

        grid = make_image_grid([id_image_pil, condition_img, image], rows=1, cols=3)
        os.makedirs(f"{output_dir}/grid", exist_ok=True)
        grid.save(grid_save_fname)

        # 시간 재기 끝
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)  # milliseconds
        print(f"GPU {rank} :  Time taken for processing {src_num}: {elapsed_time/1000:.2f} seconds")
    except Exception as e:
        print(f"Error processing pair ({src_img_path}, {trg_img_path}): {e}")

# --- Cleanup ---
dist.destroy_process_group()
