import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import json
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import torch.multiprocessing as mp

# Used model classes
from imscore.aesthetic.model import LAIONAestheticScorer
from imscore.preference.model import CLIPScore
from imscore.pickscore.model import PickScorer


def process_images_on_gpu(gpu_id, image_paths, result_dict):
    """
    A worker function that processes a subset of images on a specific GPU.
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    # Load models for this process
    models = {
        'aes': LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic").to(device),
        # 'clip': CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14").to(device),
        # 'pick': PickScorer.from_pretrained("RE-N-Y/pickscore").to(device),
    }

    prompt = "a photo of human face"
    local_json_dict = {}

    for img_path in tqdm(image_paths, desc=f"GPU-{gpu_id}", position=gpu_id, dynamic_ncols=True):
        img_id = os.path.dirname(img_path).split('/')[-1]
        img_base = os.path.basename(img_path).split('.')[0]
        
        try:
            pixels = Image.open(img_path).convert("RGB")
            pixels = np.array(pixels)
            pixels = rearrange(torch.from_numpy(pixels), "h w c -> 1 c h w") / 255.0
            pixels = pixels.to(device)

            score_dict = {}
            with torch.no_grad():
                for model_name, model in models.items():
                    score = model.score(pixels, [prompt])
                    score_dict[model_name] = score.item()
            
            local_json_dict[f'{img_id}/{img_base}'] = score_dict
        except Exception as e:
            print(f"Error processing {img_path} on GPU {gpu_id}: {e}")

    result_dict.update(local_json_dict)

def main():
    """
    Main function to distribute image processing across multiple GPUs.
    """
    images = natsorted(glob('/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/*/*.jpg'))
    images = [img for img in images if img.endswith('.png') or img.endswith('.jpg')]

    # images = images[:100]  # For testing, limit to first 100 images

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Please run on a machine with GPUs.")
        return

    print(f"Found {num_gpus} GPUs. Starting processing...")

    with mp.Manager() as manager:
        result_dict = manager.dict()
        processes = []
        
        # 'spawn' start method is recommended for CUDA multiprocessing
        mp.set_start_method('spawn', force=True)

        world_size = num_gpus
        for rank in range(world_size):
            image_chunk = images[rank::world_size]
            p = mp.Process(target=process_images_on_gpu, args=(rank, image_chunk, result_dict))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        final_json_dict = dict(result_dict)

    json_dict_path = '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/score.json'
    print(f"All processes finished. Saving results to {json_dict_path}")
    # natsort 이용해서 key 정렬
    final_json_dict = {k: final_json_dict[k] for k in natsorted(final_json_dict.keys())}
    with open(json_dict_path, 'w') as f:
        json.dump(final_json_dict, f, indent=4)

if __name__ == '__main__':
    main()
