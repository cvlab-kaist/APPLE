import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from imscore.aesthetic.model import ShadowAesthetic, LAIONAestheticScorer
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.preference.model import SiglipPreferenceScorer, CLIPScore
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
from imscore.vqascore.model import VQAScore
from imscore.cyclereward.model import CycleReward
from imscore.evalmuse.model import EvalMuse
from imscore.hpsv3.model import HPSv3
from natsort import natsorted
from glob import glob

import torch
import numpy as np
from PIL import Image
from einops import rearrange
import os
from tqdm import tqdm

# popular aesthetic/preference scorers
models = {
    'aes' : LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic"), # LAION aesthetic scorer
    'clip' : CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14"), # CLIPScore
    'pick' : PickScorer.from_pretrained("RE-N-Y/pickscore"), # PickScore preference scorer
}

# to 'cuda'
for key in models.keys():
    models[key] = models[key].to('cuda')


images = natsorted(glob('/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/*/*.jpg'))
images = [img for img in images if img.endswith('.png') or img.endswith('.jpg')]

images = images[:10]

prompt = "a photo of human face"

json_dict = {}

for img_path in tqdm(images, desc="Processing images", total=len(images), dynamic_ncols=True):
    print(f"Processing {img_path}...")
    
    # /mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n000603/0240_01.jpg
    img_id = os.path.dirname(img_path).split('/')[-1] # n000603
    img_base = os.path.basename(img_path).split('.')[0] # 0240_01
    pixels = Image.open(img_path)
    pixels = np.array(pixels)
    pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0
    pixels = pixels.to('cuda')

    score_dict = {}
    for model_name, model in models.items():
        # print(f"  Using {model_name}...")
        # score = model.score(pixels, [prompt]*pixels.shape[0]) # full differentiable reward

        score = model.score(pixels, [prompt]) # full differentiable reward
        score_dict[model_name] = score.item()
    
    json_dict[img_id+'/'+img_base] = score_dict

json_dict_path = '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/score.json'
import json
with open(json_dict_path, 'w') as f:
    json.dump(json_dict, f, indent=4)


