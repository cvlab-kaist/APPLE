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

# Config
guidance_scale=1.0
image_guidance_scale = 1.0
id_guidance_scale = 1.0
use_gaze = False
inverse_steps = 28
num_inference_steps = 28
inverse_cond = 'noID_trgCond' # ['trgID_trgCond', 'noID_trgCond', 'trgID_noCond', 'noID_noCond']
use_uncond_id = True
print(f"=== Inversion with condition: {inverse_cond} ===")

# Specify LoRA path and output dir
lora_file_path = f'/workspace/jiwon/fluxswap/runs/baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]/ckpt/step199999_global50000/default.safetensors'
output_dir = f'/workspace/jiwon/dataset/vggface2_aes5.1_pairs/flux_vgg50k_inv{inverse_steps}_infer{num_inference_steps}_uncondID{use_uncond_id}'
os.makedirs(output_dir, exist_ok=True)


import os, json, random, csv
from pathlib import Path
from collections import defaultdict
from natsort import natsorted


import json
random.seed(42)

dataset_path = "/workspace/jiwon/dataset/vgg_iris2"
json_path = os.path.join(dataset_path, "score.json")
path_pairs_path = os.path.join(output_dir, "pairs.pt")
# os.remove(path_pairs_path) if os.path.exists(path_pairs_path) else None # DEBUG용

if not os.path.exists(path_pairs_path):
    # 1) AES 필터링된 이미지 리스트 만들기
    with open(json_path, "r") as f:
        score_dict = json.load(f)

    training_base_list = set(os.listdir(dataset_path))  # 폴더 이름들 (n000001 ...)
    high_aes_keys = [k for k, v in score_dict.items() if v.get("aes", -1) > 5.1]

    # 키 형식이 "n000008/0145_01" 같은 경우를 가정해 .jpg 경로 구성
    img_list = []
    for k in tqdm(high_aes_keys, total=len(high_aes_keys), desc="Filtering images with AES>5.1 and existing MaskID/Condition"):
        id_name = k.split('/')[0] # e.g., "n000008"
        file_name = k.split('/')[1] # e.g., "0145_01"
        rel = k + ".jpg"  # e.g., "n000008/0145_01.jpg"
        full = os.path.join(dataset_path, rel)
        # 폴더가 실제로 존재하고 파일도 존재하는 경우만
        cond_path = os.path.join(dataset_path, os.path.dirname(rel), 'condition_blended_image_blurdownsample8_segGlass_landmark_iris', file_name + '.png')
        # masked_id_path = os.path.join(dataset_path, os.path.dirname(rel), 'masked_pulid_id', file_name + '.npy')

        if os.path.exists(full) and os.path.exists(cond_path) and rel.split("/")[0] in training_base_list:
            img_list.append(full)

    print(f"AES>5.1 통과 및 Condition 있는 이미지: {len(img_list)}")

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

    total_length = 100_000
    start = 0
    pairs = set()  # ← 중복 방지용 set

    # 1. 가능한 모든 (src_id, trg_id) 쌍 만들기
    all_pairs = [(src_id, trg_id) for src_id in ids for trg_id in ids if src_id != trg_id]

    # 2. 무작위 섞기
    rng = random.Random(42)
    rng.shuffle(all_pairs)

    # 3. 필요한 개수만 꺼내기
    path_pairs = []
    for src_id, trg_id in all_pairs[:total_length]:
        src_path = rng.choice(id_to_images[src_id])
        trg_path = rng.choice(id_to_images[trg_id])
        path_pairs.append((src_path, trg_path))
        
    path_pairs = natsorted(path_pairs)[start:start+total_length]
    print(f"✅ {len(path_pairs)} unique pairs generated (no duplicates)")
    torch.save(path_pairs, path_pairs_path)
    print(f"[INFO] Saved pairs to {path_pairs_path}")
else :
    print(f"[INFO] Loading existing pairs from {path_pairs_path}")
    path_pairs = torch.load(path_pairs_path)



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
transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None, use_gaze=use_gaze, local_rank=local_rank)
# no grad
torch.set_grad_enabled(False)

from pulid.utils import resize_numpy_image_long
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_txt
from diffusers import FluxPipeline
# from pipeline_flux_ca import FluxPipelineCA
from omini.pipeline.flux_omini import generate_ca, Condition, convert_to_condition, generate_ca_inv


flux = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", transformer=transformer, torch_dtype=weight_dtype).to(device)


#### Load LoRA weights
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


## Shuffle with time-based seed
import time
cur_time_for_rng = int(time.time())  # ✅ 예: 1712345678
rng = random.Random(42 + cur_time_for_rng)
rng.shuffle(path_pairs)

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
        grid_save_fname = f"{output_dir}/grid/{src_id}_{src_num}_{trg_id}_{trg_num}_grid.jpg"
        if os.path.exists(img_save_fname):
            print(f"Image {img_save_fname} already exists. Skipping...")
            continue

        cond_img_path_base = os.path.dirname(trg_img_path)
        cond_img_path = os.path.join (cond_img_path_base, 'condition_blended_image_blurdownsample8_segGlass_landmark_iris', f"{trg_num}.png")
        condition_img = Image.open(cond_img_path).convert('RGB')


        # 시간 재기
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        if use_gaze:
            import numpy as np
            trg_img_path_base = os.path.dirname(trg_img_path)
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
        
        
        id_embeddings, uncond_id_embeddings = flux.transformer.get_id_embedding(id_image, cal_uncond=True)
        id_embeddings = id_embeddings.to(device, dtype=weight_dtype)
        uncond_id_embeddings = uncond_id_embeddings.to(device, dtype=weight_dtype)
        
        # ID embed (Trg)
        trg_img = Image.open(trg_img_path).convert('RGB').resize((512,512))
        id_image_trg = cv2.imread(trg_img_path)
        print(f"id_image_trg shape: {id_image_trg.shape}")
        id_image_trg = cv2.cvtColor(id_image_trg, cv2.COLOR_BGR2RGB)
        id_image_trg = resize_numpy_image_long(id_image_trg, 1024)
        id_image_trg_pil = Image.fromarray(id_image_trg)
        id_image_trg_pil = id_image_trg_pil.resize((512,512))
        # id_image_trg_pil.save(f"{output_dir}/{src_num}_id_trg.png")   
        # trg_id_embeddings, trg_uncond_id_embeddings = flux.transformer.get_id_embedding(id_image_trg, cal_uncond=True)
        # trg_id_embeddings = trg_id_embeddings.to(device, dtype=weight_dtype)
        # trg_uncond_id_embeddings = trg_uncond_id_embeddings.to(device, dtype=weight_dtype)

        # Nan check
        if torch.isnan(id_embeddings).any():
            print(f"WARNING: NaN detected in id_embeddings for {src_img_path}. Skipping.")
            continue


        from diffusers.utils import make_image_grid
        prompt = 'a photo of human face'
        negative_prompt = ''

        image_list = []

        # 1. Inv 
        inverted_latents = generate_ca_inv(
                flux,
                prompt=prompt,
                conditions=[condition] if 'trgCond' in inverse_cond else [],
                height=512,
                width=512,
                generator=torch.Generator('cpu').manual_seed(0),
                kv_cache=False,
                id_embed=uncond_id_embeddings if use_uncond_id else None,
                uncond_id_embed=uncond_id_embeddings if use_uncond_id else None,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                id_guidance_scale=id_guidance_scale,
                gaze_embed=gaze_embed,
                # Inversion
                inverse=True,
                inverse_steps=inverse_steps,
                num_inference_steps=inverse_steps,
                inverse_img=trg_img,
                output_type='latent',
                return_dict=False
                )[0]

        # 4. Inv -> (Src ID, Trg Cond)
        img = generate_ca_inv(
                flux,
                prompt=prompt,
                conditions=[condition],
                height=512,
                width=512,
                latents=inverted_latents,
                generator=torch.Generator('cpu').manual_seed(0),
                kv_cache=False,
                id_embed=id_embeddings,
                uncond_id_embed=uncond_id_embeddings,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                id_guidance_scale=id_guidance_scale,
                gaze_embed=gaze_embed,
                # Inversion
                inverse_steps=inverse_steps,
                num_inference_steps=num_inference_steps,
            )
        image = img.images[0] if isinstance(img.images, list) else img.images
        image.save(img_save_fname)
        image_list.append(image)
        
        grid = make_image_grid([id_image_pil, trg_img, condition_img] + image_list, rows=1, cols=3 + len(image_list))
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
