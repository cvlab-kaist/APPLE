import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np

from PIL import Image, ImageDraw

from datasets import load_dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate_ca

from glob import glob
from natsort import natsorted
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor
import torch.nn.functional as F
import diffusers
diffusers.utils.logging.set_verbosity_error()

'''
id_embed_candidates_cache

{'n006573': ['/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n006573/masked_pulid_id/0285_01.npy',
  '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n006573/masked_pulid_id/0213_01.npy',
  '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n006573/masked_pulid_id/0012_01.npy',
  '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n006573/masked_pulid_id/0001_01.npy',
  '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n006573/masked_pulid_id/0410_01.npy',
'''


class FFHQDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_path= '/mnt/data2/dataset/datasets/rahulbhalley/ffhq-1024x1024/versions/1/images1024x1024',
        num_validation = 5,
        mode = 'train',
        condition_type = 'condition_blended_image_blurdownsample8_segGlass_landmark',
        *kwargs,
    ):

        self.dataset_path = dataset_path
        img_list = sorted(glob(f"{dataset_path}/*.png"))

        if mode == 'train':
            img_list = img_list[:-num_validation]
        else:
            img_list = img_list[-num_validation:]
        
        # dirname_list = [os.path.dirname(f) for f in img_list]
        basename_list = [os.path.basename(f).split('.')[0] for f in img_list]
        id_embed_list = [os.path.join(dataset_path, 'pulid_id', f"{basename}.npy") for basename in basename_list]
        uncond_id_embed_path = os.path.join(dataset_path, 'pulid_id', 'uncond.npy')
        condition_list = [os.path.join(dataset_path, condition_type, basename + '.png') for basename in basename_list]


        self.image_paths = img_list
        self.id_embed_paths = id_embed_list
        self.controlnet_paths = condition_list
        self.uncond_id_embed = torch.Tensor(np.load(uncond_id_embed_path))


    def __len__(self):
        # assert len(self.image_paths) == len(self.mask_paths)
        # assert len(self.mask_paths) == len(self.image_paths)
        return len(self.image_paths)

    def __getitem__(self, idx):
        while True:
            try:

                # Processing
                img = Image.open(self.image_paths[idx]).convert('RGB')
                controlnet_img = Image.open(self.controlnet_paths[idx]).convert('RGB')
                face_id_embed = torch.Tensor(np.load(self.id_embed_paths[idx]))
                # original_width, original_height = img.size
                

                # # Resizing
                # img = img.resize((self.size, self.size))
                # controlnet_img = controlnet_img.resize((self.size, self.size))

                # # Transform to pytorch tensor w/ [-1,1]
                # img = self.train_transforms(img)
                # controlnet_img = self.controlnet_transforms(controlnet_img)
                # # seg_img = self.controlnet_transforms(seg_img)

                # # drop
                # text = self.args.validation_prompt
                # drop_image_embed = 0
                # rand_num = random.random()
                # if rand_num < self.i_drop_rate:
                #     drop_image_embed = 1
                # elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                #     text = ""
                # elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                #     text = ""
                #     drop_image_embed = 1
                # if drop_image_embed:
                #     face_id_embed = torch.zeros_like(face_id_embed)
                # # get text and tokenize
                # # text_input_ids = self.tokenizer(
                # #     text,
                # #     max_length=self.tokenizer.model_max_length,
                # #     padding="max_length",
                # #     truncation=True,
                # #     return_tensors="pt"
                # # ).input_ids
                # prompt_embeds, pooled_prompt_embeds = self.compute_text_embeddings(
                #     text, self.text_encoders, self.tokenizers
                # )

                return {
                    "img": img,
                    # "original_size": (original_height, original_width),
                    # "prompt_embeds": prompt_embeds,
                    # "pooled_prompt_embeds": pooled_prompt_embeds,
                    "face_id_embed": face_id_embed,
                    "uncond_id_embed": self.uncond_id_embed,
                    # "drop_image_embed": drop_image_embed,
                    'controlnet_img': controlnet_img,
                }

            
            except Exception as e:
                print(f"[WARN] idx {idx} failed with error: {e}, retrying...")
                idx = random.randint(0, self.__len__() - 1)


class VGGDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_path='/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan',
        swapped_path= '/mnt/data6/vgg_swapped/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt80000_gs1.0_imgGS1.0_idGS1.0',
        num_validation = 5,
        mode = 'train',
        condition_type = 'condition_blended_image_blurdownsample8_segGlass_landmark',
        train_gaze = False,
        gaze_type = 'unigaze',
        pseudo = False,
        id_from = 'original', # options: ['original', 'masked']
        swapped_condition_type = None, # options: [None, 'condition_pose']
        id_embed_candidates_cache = None,
        get_random_id_embed_every_step = False,
        validation_with_other_src_id_embed = False,
    ):
        # 예시) /mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n000002/0001_01.jpg
        # 예시) /mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan/n000002/masked_pulid_id/0001_01.npy

        self.dataset_path = dataset_path
        self.train_gaze = train_gaze
        self.swapped_condition_type = swapped_condition_type
        self.id_embed_candidates_cache = id_embed_candidates_cache
        self.get_random_id_embed_every_step = get_random_id_embed_every_step
        self.validation_with_other_src_id_embed = validation_with_other_src_id_embed
        print(f"[INFO] VGGDataset {mode} set w/ get_random_id_embed_every_step: {self.get_random_id_embed_every_step}")
        print(f"[INFO] condition_type: {condition_type}")

        if id_from == 'original':
            id_dirname = 'pulid_id'
        elif id_from == 'masked':
            id_dirname = 'masked_pulid_id'
        else:
            raise ValueError(f"Invalid id_from value: {id_from}")
        self.id_dirname = id_dirname

        random.seed(0) # Seed for reproducibility
        if not pseudo:
            print("[INFO] 1st Stage Training: Using real dataset with for VGGDataset")
            
            # 1) AES Score 기준으로 필터링
            import json
            json_path = os.path.join(dataset_path, 'score.json')
            with open(json_path, 'r') as f:
                score_dict = json.load(f)

            training_base_list = list(natsorted(os.listdir(dataset_path)))
            # training_base_list = list(natsorted(os.listdir(dataset_path)))[:-2000] # 마지막 2000개는 validation set으로 사용
            high_aes_keys = [k for k, v in score_dict.items() if v['aes'] > 5.5]
            img_list = [os.path.join(dataset_path, k + '.jpg') for k in high_aes_keys if k.split('/')[0] in training_base_list] 
            print(f"[DEBUG] Sample Image List : {img_list[:5]}")
            # img_list = natsorted(img_list)[:1000] # for debugging, limit to 1000 images
            print(f"Filtered images based on AES score > 5.5: {len(img_list)} images remain.")

            random.shuffle(img_list)
            if mode == 'train':
                img_list = img_list[:-num_validation]
            else:
                img_list = img_list[-num_validation:]

            dirname_list = [os.path.dirname(f).split('/')[-1] for f in img_list] # e.g. n000002, # Target
            basename_list = [os.path.basename(f).split('.')[0] for f in img_list] # e.g. 0001_01# Target

            gaze_paths = None
            if train_gaze:
                # 2) imgs_list를 gaze_path 기준으로 다시 필터링 : gaze가 안된 이미지 (e.g. 80도 이상)도 존재
                # Update : img_list, gaze_paths, dirname_list, basename_list 
                gaze_paths = [os.path.join(dataset_path, dirname, gaze_type, f"{basename}.npy") for (dirname, basename) in zip(dirname_list, basename_list) ]
                print(f"[INFO] Using gaze type: {gaze_type}")
                img_list = [img for img, gaze_path in zip(img_list, gaze_paths) if os.path.exists(gaze_path)]   # gaze path가 존재하는 이미지들만 선택
                gaze_paths = [gaze_path for gaze_path in gaze_paths if os.path.exists(gaze_path)] # gaze path도 동기화
                dirname_list = [os.path.dirname(f) for f in img_list] # 동기화 
                basename_list = [os.path.basename(f).split('.')[0] for f in img_list] # 동기화

                print(f"[INFO] After filtering based on gaze paths: {len(img_list)} images remain.")

                # 모든 gaze path가 존재하는지 확인
                for gaze_path in gaze_paths:
                    if not os.path.exists(gaze_path):
                        raise ValueError(f"Gaze path does not exist: {gaze_path}")


            # 3) Source의 Image 및 ID embed 설정
            # target과 동일한 인물 내에서 랜덤하게 하나 선택
            src_img_list = []
            id_embed_list = []
            effective_dirname_list = []
            effective_basename_list = []
            if id_embed_candidates_cache is None:
                id_embed_candidates_cache = {} # 캐시
                for dirname in tqdm(set(dirname_list), desc="Caching ID embed candidates"):
                    id = dirname # e.g. n000002
                    id_embed_candidates = glob(os.path.join(dataset_path, id, id_dirname, '*.npy')) #
                    id_embed_candidates_cache[id] = id_embed_candidates
            for num, (dirname, basename) in enumerate(zip(dirname_list, basename_list)):
                id = dirname # e.g. n000002
                if mode == 'test' and self.validation_with_other_src_id_embed:
                    print("[INFO] Validation with other src ID embed enabled. Validation by FaceSwap setting for 1st Stage.")
                    random.seed(num) # Seed for reproducibility
                    id = random.choice([d for d in natsorted(list(id_embed_candidates_cache.keys())) if d != id]) # test 시에는 자기 자신 제외한 다른 인물에서 선택
                    id_embed_candidates = id_embed_candidates_cache.get(id, []) 
                    print(f"[DEBUG] For target ID {dirname}, selected src ID {id} with {len(id_embed_candidates)} candidates.")
                    if len(id_embed_candidates) == 0:
                        print (f"[WARN] No ID embed candidates found for {id}, skipping sample.")
                        continue
                    else :
                        effective_dirname_list.append(dirname)
                        effective_basename_list.append(basename)
                        random.seed(num) # Seed for reproducibility 
                        selected_id_embed = random.choice(id_embed_candidates) # 해당 ID 내에서 랜덤하게 ID Embed 선택
                        id_embed_list.append(selected_id_embed) 
                        src_img_list.append(os.path.join(dataset_path, id, os.path.basename(selected_id_embed).replace('.npy', '.jpg'))) # 해당 ID embed에 대응하는 이미지
                else :
                    # 정상적으로 로딩, train 시에는 target과 동일 인물 내에서 선택
                    id_embed_candidates = id_embed_candidates_cache.get(id, [])
                    if len(id_embed_candidates) == 0:
                        print (f"[WARN] No ID embed candidates found for {id}, skipping sample.")
                        continue
                    else :
                        effective_dirname_list.append(dirname)
                        effective_basename_list.append(basename)
                        selected_id_embed = random.choice(id_embed_candidates) # 해당 ID 내에서 랜덤하게 ID Embed 선택
                        id_embed_list.append(selected_id_embed) 
                        src_img_list.append(os.path.join(dataset_path, dirname, os.path.basename(selected_id_embed).replace('.npy', '.jpg'))) # 해당 ID embed에 대응하는 이미지
            dirname_list = effective_dirname_list
            basename_list = effective_basename_list
            img_list = [os.path.join(dataset_path, dirname, f"{basename}.jpg") for (dirname, basename) in zip(dirname_list, basename_list)]

            uncond_id_embed_path = os.path.join(dataset_path, 'n000002/pulid_id/uncond.npy')
            condition_list = [os.path.join(dataset_path, dirname, condition_type, basename + '.png') for (dirname, basename) in zip(dirname_list, basename_list)]

            # 최종 셋팅
            self.src_img_list = src_img_list
            self.image_paths = img_list
            self.id_embed_paths = id_embed_list
            self.controlnet_paths = condition_list
            self.gaze_paths = gaze_paths

            # # DEBUG 5개씩
            # print(f"[DEBUG] Sample src_img_list : {self.src_img_list[:5]}")
            # print(f"[DEBUG] Sample image_paths : {self.image_paths[:5]}")
            # print(f"[DEBUG] Sample id_embed_paths : {self.id_embed_paths[:5]}")
            # print(f"[DEBUG] Sample controlnet_paths : {self.controlnet_paths[:5]}")
            # if gaze_paths is not None:
            #     print(f"[DEBUG] Sample gaze_paths : {self.gaze_paths[:5]}")
            
            self.uncond_id_embed = torch.Tensor(np.load(uncond_id_embed_path))

            assert len(self.image_paths) == len(self.id_embed_paths)
            assert len(self.image_paths) == len(self.controlnet_paths)

            print(f"[INFO] Real dataset size: {len(self.image_paths)} images.")
        else :
            print(f"[INFO] Using pseudo dataset for VGGDataset, condition type: {self.swapped_condition_type}")
            if self.swapped_condition_type is not None:
                swapped_path = os.path.join(swapped_path, self.swapped_condition_type)

            if train_gaze:
                raise NotImplementedError("Pseudo dataset with gaze conditioning not implemented yet.")

            self.swapped_trg_path = swapped_path
            swapped_trg_imgs = natsorted(glob(f"{self.swapped_trg_path}/*.png")) # .../ n006195_0211_01_n006500_0203_02.png -> {src_id}_{src_num}_{trg_id}_{trg_num}.png
            print(f"[INFO] Found {len(swapped_trg_imgs)} swapped images in {self.swapped_trg_path}")
            
            # 파싱
            import re
            pattern = r"^(n\d{6})_(\d{4}_\d{2})_(n\d{6})_(\d{4}_\d{2})\.png$"
            src_ids, src_nums, trg_ids, trg_nums = [], [], [], []    
            for trg_img_path in tqdm(swapped_trg_imgs, desc="Parsing swapped image filenames"):
                filename = os.path.basename(trg_img_path)
                match = re.match(pattern, filename)
                if match:
                    src_id, src_num, trg_id, trg_num = match.groups()
                    src_ids.append(src_id)
                    src_nums.append(src_num)
                    trg_ids.append(trg_id)
                    trg_nums.append(trg_num)
                else:
                    raise ValueError(f"Filename does not match pattern: {filename}")
            
            # ID 임베딩 준비, trg_id마다 랜덤한 ID 임베딩 선택
            src_img_list = []
            id_embed_list = []
            controlnet_list = []
            controlnet_ids = []
            controlnet_nums = []
            if id_embed_candidates_cache is None:
                id_embed_candidates_cache = {} # 캐시
                for trg_id in tqdm(set(trg_ids)): # 캐싱
                    id_embed_candidates = glob(os.path.join(dataset_path, trg_id, id_dirname, '*.npy')) #
                    id_embed_candidates_cache[trg_id] = id_embed_candidates
            for idx, (trg_id, trg_num) in tqdm(enumerate(zip(trg_ids, trg_nums)), desc="Preparing pseudo dataset"):
                id_embed_candidates = id_embed_candidates_cache[trg_id]
                # 자기 자신 제외
                id_embed_candidates = [p for p in id_embed_candidates if os.path.basename(p) != f"{trg_num}.npy"]
                if len(id_embed_candidates) == 0:
                    print (f"[WARN] No ID embed candidates found for {trg_id}, skipping sample.")
                    continue
                else :
                    # Select candidate
                    selected_id_embed = random.choice(id_embed_candidates)
                    # Select src img / ID embed
                    src_img_list.append(os.path.join(dataset_path, trg_id, os.path.basename(selected_id_embed).replace('.npy', '.jpg')))
                    id_embed_list.append(selected_id_embed)
                    controlnet_list.append(swapped_trg_imgs[idx])
                    controlnet_ids.append(trg_id)
                    controlnet_nums.append(trg_num)

            print(f"[INFO] Pseudo dataset prepared with {len(id_embed_list)} samples.")

            # GT target 
            trg_list = [os.path.join(dataset_path, trg_id, f"{trg_num}.jpg") for (trg_id, trg_num) in zip(controlnet_ids, controlnet_nums)]

            uncond_id_embed_path = os.path.join(dataset_path, 'n000002/pulid_id/uncond.npy')

            # Final
            self.src_img_list = src_img_list # Source (ID embed용)
            self.id_embed_paths = id_embed_list
            self.controlnet_paths = controlnet_list # Condition : swapped image
            self.image_paths = trg_list # GT target
            self.uncond_id_embed = torch.Tensor(np.load(uncond_id_embed_path))
            self.id_embed_candidates_cache = id_embed_candidates_cache
            
            assert len(self.image_paths) == len(self.id_embed_paths)
            assert len(self.image_paths) == len(self.controlnet_paths)
            assert len(self.image_paths) == len(self.src_img_list)

            
            if mode == 'train':
                self.image_paths = self.image_paths[:-num_validation]
                self.id_embed_paths = self.id_embed_paths[:-num_validation]
                self.controlnet_paths = self.controlnet_paths[:-num_validation]
                self.src_img_list = self.src_img_list[:-num_validation]
                print(f"[INFO] Pseudo dataset size: {len(self.image_paths)} images.")
            else:
                self.image_paths = self.image_paths[-num_validation:]
                self.id_embed_paths = self.id_embed_paths[-num_validation:]
                self.controlnet_paths = self.controlnet_paths[-num_validation:]
                self.src_img_list = self.src_img_list[-num_validation:]
                print(f"[INFO] Pseudo dataset size (validation): {len(self.image_paths)} images.")

            



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        while True:
            try:
                # Processing
                src_img = Image.open(self.src_img_list[idx]).convert('RGB')
                src_img_basename = os.path.basename(self.src_img_list[idx]).split('.')[0] # e.g. 0001_01
                img = Image.open(self.image_paths[idx]).convert('RGB') # GT target
                controlnet_img = Image.open(self.controlnet_paths[idx]).convert('RGB')
                if self.get_random_id_embed_every_step:
                    # 동일한 ID 중에서 매번 랜덤 선택
                    id = self.id_embed_paths[idx].split('/')[-3] # e.g. n000002
                    id_candidates = self.id_embed_candidates_cache[id]
                    # 자기 자신밖에 없을수도 있으니, 제외 처리 안하고 랜덤 선택
                    selected_id_embed = random.choice(id_candidates)
                    face_id_embed = torch.Tensor(np.load(selected_id_embed))
                else:
                    face_id_embed = torch.Tensor(np.load(self.id_embed_paths[idx]))
                gaze_embed = torch.Tensor(np.load(self.gaze_paths[idx])) if self.train_gaze else None
                

                return {
                    "src_img": src_img,
                    "img": img,
                    # "original_size": (original_height, original_width),
                    # "prompt_embeds": prompt_embeds,
                    # "pooled_prompt_embeds": pooled_prompt_embeds,
                    "face_id_embed": face_id_embed,
                    "uncond_id_embed": self.uncond_id_embed,
                    # "drop_image_embed": drop_image_embed,
                    'controlnet_img': controlnet_img,
                    "gaze_embed": gaze_embed ,
                }

            
            except Exception as e:
                print(f"[WARN] idx {idx} failed with error: {e}, retrying...")
                idx = random.randint(0, self.__len__() - 1)


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type: str = "deblurring",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        drop_id_prob: float = 0.1,
        drop_gaze_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.drop_id_prob = drop_id_prob
        self.drop_gaze_prob = drop_gaze_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale


        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __get_condition__(self, image, condition_type):
        condition_size = self.condition_size
        position_delta = np.array([0, 0])
        if condition_type in ["canny", "coloring", "deblurring", "depth"]:
            image, kwargs = image.resize(condition_size), {}
            if condition_type == "deblurring":
                blur_radius = random.randint(1, 10)
                kwargs["blur_radius"] = blur_radius
            condition_img = convert_to_condition(condition_type, image, **kwargs)
        elif condition_type == "depth_pred":
            depth_img = convert_to_condition("depth", image)
            condition_img = image.resize(condition_size)
            image = depth_img.resize(condition_size)
        elif condition_type == "fill":
            condition_img = image.resize(condition_size).convert("RGB")
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
        elif condition_type == "sr":
            condition_img = image.resize(condition_size)
            position_delta = np.array([0, -condition_size[0] // 16])
        else:
            raise ValueError(f"Condition type {condition_type} is not  implemented.")
        return condition_img, position_delta

    def __getitem__(self, idx):
        try :
            base_dataset_item = self.base_dataset[idx]
            image = base_dataset_item["img"]
            src_img = base_dataset_item.get('src_img', base_dataset_item['img']) # src_img이 없으면 그냥 img 사용
            image = image.resize(self.target_size).convert("RGB")
            gaze_embed = base_dataset_item.get('gaze_embed', None)
            gaze_embed = gaze_embed.squeeze() if gaze_embed is not None else None
            # description = base_dataset_item["json"]["prompt"] 
            description = 'a photo of human face' # using fixed prompt for face swap

            condition_size = self.condition_size
            position_scale = self.position_scale

            _, position_delta = self.__get_condition__(
                image, self.condition_type
            )
            condition_img = base_dataset_item["controlnet_img"]
            condition_img = condition_img.resize(condition_size).convert("RGB")

            id_embed = base_dataset_item["face_id_embed"].squeeze()
            uncond_id_embed = base_dataset_item["uncond_id_embed"].squeeze()

            # Randomly drop text or image (for training)
            drop_text = random.random() < self.drop_text_prob
            drop_image = random.random() < self.drop_image_prob
            drop_id = random.random() < self.drop_id_prob
            drop_gaze = random.random() < self.drop_gaze_prob

            if drop_text:
                description = ""
            if drop_image:
                condition_img = Image.new("RGB", condition_size, (0, 0, 0))
            if drop_id:
                id_embed = uncond_id_embed
            if drop_gaze and gaze_embed is not None:
                gaze_embed = torch.zeros_like(gaze_embed)
        except Exception as e:
            # In case of error, return a random item
            print(f"[WARN] idx {idx} failed with error: {e}, retrying...")
            return self.__getitem__(random.randint(0, self.__len__() - 1))



        return {
            "src_img": self.to_tensor(src_img),
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": position_delta,
            "description": description,
            "id_embed": id_embed,
            'uncond_id_embed': uncond_id_embed,
            "drop_gaze" : drop_gaze, # type : bool
            **({"gaze_embed": gaze_embed} if gaze_embed is not None else {}),
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }


# TODO : for faceswap
@torch.no_grad()
def test_function(model, save_path, file_name, test_dataset):
    # model : OminiModel
    print(f"[DEBUG] In test_function, LoRA weight mean: {model.transformer.transformer_blocks[0].attn.to_q.lora_A['default'].weight.data.mean()}")

    os.makedirs(save_path, exist_ok=True)

    trg_imgs = []
    condition_imgs = []
    result_imgs = []
    src_imgs = []

    for i in range(len(test_dataset)):
        each = test_dataset[i]
        src_img = each.get('src_img', each['image']) # torch.Tensor [0, 1]
        img = each['image'] # torch.Tensor [0, 1] 
        prompt = each["description"]
        condition_img = each["condition_0"] # torch.Tensor [0, 1]
        condition_type = each["condition_type_0"]
        position_delta = each.get('position_delta', [0,0])
        position_scale = each.get('position_scale', 1.0)
        id_embed = each['id_embed']
        if len(id_embed.shape) == 2: # Make batch size 1
            id_embed = id_embed.unsqueeze(0)
        uncond_id_embed = each['uncond_id_embed']
        condition = Condition(condition_img, model.adapter_names[2], position_delta, position_scale)
        target_size = model.training_config["dataset"]["target_size"]
        gaze_embed = each.get('gaze_embed', None) # (gaze_dim,) or None
        print(f"[DEBUG] Test sample {i}, gaze_embed.shape : {gaze_embed.shape if gaze_embed is not None else None}")
        if gaze_embed is not None and len(gaze_embed.shape) == 1:
            gaze_embed = gaze_embed.unsqueeze(0)
            

        generator = torch.Generator('cpu').manual_seed(0)

        res = generate_ca(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
            id_embed=id_embed,
            uncond_id_embed=uncond_id_embed,
            gaze_embed=gaze_embed,
        )
        file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        res.images[0].save(file_path)
        result_imgs.append(res.images[0])

        # save condition image
        condition_image_pil = T.ToPILImage()(condition_img)
        condition_imgs.append(condition_image_pil)
        condition_image_pil.save(os.path.join(save_path, f"{file_name}_{condition_type}_{i}_condition.jpg"))

        img_pil = T.ToPILImage()(img)
        trg_imgs.append(img_pil)
        img_pil.save(os.path.join(save_path, f"{file_name}_{condition_type}_{i}_original.jpg"))

        src_img_pil = T.ToPILImage()(src_img)
        src_img_pil.save(os.path.join(save_path, f"{file_name}_{condition_type}_{i}_src.jpg"))
        src_imgs.append(src_img_pil)

        # result, trg의 ID Loss 계산
        val_recon_loss = 0.0
        val_id_loss = 0.0
        for result_img, trg_img, src_img in zip(result_imgs, trg_imgs, src_imgs):
            # Src와 Result의 ID Loss 계산
            val_id_loss += model.id_loss_func_from_pil(result_img, src_img, model.flux_pipe.transformer.netarc).mean().item() # (1,)
            # Result와 Trg의 Reconstruction Loss 계산
            result_img_pt = pil_to_tensor(result_img).float() / 255.0
            trg_img_pt = pil_to_tensor(trg_img).float() / 255.0
            val_recon_loss += F.mse_loss(result_img_pt, trg_img_pt).mean().item() # (1,)

        val_recon_loss = val_recon_loss / len(result_imgs)
        val_id_loss = val_id_loss / len(result_imgs)
        val_loss = val_recon_loss + val_id_loss

    return trg_imgs, condition_imgs, result_imgs, src_imgs, val_loss, val_recon_loss, val_id_loss


def main():
    # Set seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # 1) PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU/프로세스

    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    if config.get("debug", False):
        print("[DEBUG] Debug mode is ON: accumulate_grad_batches set to 1, num_validation set to 1")
        config['accumulate_grad_batches'] = 1
        training_config['dataset']['num_validation'] = 1
        training_config['run_name'] = '[DEBUG]' + training_config['run_name']

    # Load dataset text-to-image-2M
    # dataset = load_dataset(
    #     "webdataset",
    #     data_files={"train": training_config["dataset"]["urls"]},
    #     split="train",
    #     cache_dir="cache/t2i2m",
    #     num_proc=32,
    # )

    dataset_type = training_config["dataset"].get("name", "ffhq")
    if dataset_type == "ffhq":
        dataset_class = FFHQDataset
    elif dataset_type == "vgg":
        dataset_class = VGGDataset
    else:
        print(f"[ERROR] Dataset type {dataset_type} is not implemented.")
        raise NotImplementedError
    print(f"[INFO] Using dataset type: {dataset_type}, with class {dataset_class}")

    cache_vgg = training_config["dataset"].get("cache_vgg", True)
    if cache_vgg :
        vgg_dataset_path = training_config["dataset"]["dataset_path"] # '/mnt/data2/dataset/VGGface2_None_norm_512_true_bygfpgan'
        cache_vgg_path = os.path.join(vgg_dataset_path, 'id_embed_cache.pt')
        if os.path.exists(cache_vgg_path):
            print(f"[INFO] Loading cached VGG ID embed candidates from {cache_vgg_path}")
            id_embed_candidates_cache = torch.load(cache_vgg_path)
        else:
            id_from = training_config["dataset"].get("id_from", "original")
            if id_from == 'original':
                id_dirname = 'pulid_id'
            elif id_from == 'masked':
                id_dirname = 'masked_pulid_id'
            ids = natsorted(os.listdir(vgg_dataset_path))
            id_embed_candidates_cache = {} # 캐시
            for id in tqdm(set(ids)): # 캐싱
                id_embed_candidates = glob(os.path.join(vgg_dataset_path, id, id_dirname, '*.npy'))
                id_embed_candidates_cache[id] = natsorted(list(id_embed_candidates))
            torch.save(id_embed_candidates_cache, cache_vgg_path)
            print(f"[INFO] Cached VGG ID embed candidates saved to {cache_vgg_path}")

    train_dataset = dataset_class(
        dataset_path=training_config["dataset"]["dataset_path"],
        mode='train',
        num_validation = training_config["dataset"].get("num_validation", 5),
        condition_type=training_config["dataset"].get("condition_type", 'condition_blended_image_blurdownsample8_segGlass_landmark'),
        gaze_type=training_config.get("gaze_type", 'unigaze'),
        pseudo=training_config["dataset"].get("pseudo", False),
        swapped_path=training_config["dataset"].get("swapped_path", None),
        id_from = training_config["dataset"].get("id_from", "original"),
        swapped_condition_type=training_config["dataset"].get("swapped_condition_type", None),
        id_embed_candidates_cache=id_embed_candidates_cache if cache_vgg and dataset_type == "vgg" else None,
        get_random_id_embed_every_step=training_config["dataset"].get("get_random_id_embed_every_step", False),
        validation_with_other_src_id_embed = training_config["dataset"].get("validation_with_other_src_id_embed", False)

    )
    test_dataset = dataset_class(
        dataset_path=training_config["dataset"]["dataset_path"],
        mode='test',
        num_validation = training_config["dataset"].get("num_validation", 5),
        condition_type= training_config["dataset"].get("condition_type", 'condition_blended_image_blurdownsample8_segGlass_landmark'),
        gaze_type=training_config.get("gaze_type", 'unigaze'),
        pseudo=training_config["dataset"].get("pseudo", False),
        swapped_path=training_config["dataset"].get("swapped_path", None),
        id_from = training_config["dataset"].get("id_from", "original"),
        swapped_condition_type=training_config["dataset"].get("swapped_condition_type", None),
        id_embed_candidates_cache=id_embed_candidates_cache if cache_vgg and dataset_type == "vgg" else None,
        get_random_id_embed_every_step= False, # no need for testing
        validation_with_other_src_id_embed = training_config["dataset"].get("validation_with_other_src_id_embed", False)
    )

    # Initialize custom dataset
    dataset = ImageConditionDataset(
        train_dataset,
        condition_size=training_config["dataset"]["condition_size"],
        target_size=training_config["dataset"]["target_size"],
        condition_type=training_config["condition_type"],
        drop_text_prob=training_config["dataset"]["drop_text_prob"],
        drop_image_prob=training_config["dataset"]["drop_image_prob"],
        drop_id_prob=training_config["dataset"].get("drop_id_prob", 0.1),
        drop_gaze_prob=training_config["dataset"].get("drop_gaze_prob", 0.1),
        position_scale=training_config["dataset"].get("position_scale", 1.0),
    )

    test_dataset = ImageConditionDataset(
        test_dataset,
        condition_size=training_config["dataset"]["condition_size"],
        target_size=training_config["dataset"]["target_size"],
        condition_type=training_config["condition_type"],
        drop_text_prob=0.0,
        drop_image_prob=0.0,
        drop_id_prob=0.0,
        drop_gaze_prob=0.0,
        position_scale=training_config["dataset"].get("position_scale", 1.0),
    )
    # with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
    
    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_path=training_config.get("lora_path", None),
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        use_netarc=training_config.get("use_netarc", False),
        netarc_path=training_config.get("netarc_path", None),
        use_irse50=training_config.get("use_irse50", False),
        id_loss_thres=training_config.get("id_loss_thres", 0.5),
        train_omini=training_config.get("train_omini", False),
        train_pulid_enc=training_config.get("train_pulid_enc", False),
        train_pulid_ca=training_config.get("train_pulid_ca", False),
        gaze_type=training_config.get("gaze_type", 'unigaze'),
        train_gaze=training_config.get("train_gaze", False),
        train_gaze_type=training_config.get("train_gaze_type", 'CA'),
        train_gaze_loss=training_config.get("train_gaze_loss", False),
        train_gaze_loss_type=training_config.get("train_gaze_loss_type", 'feature'),
        train_lpips_loss=training_config.get("train_lpips_loss", False),
        lpips_weight=training_config.get("lpips_weight", 1.0),
        lpips_loss_thres=training_config.get("lpips_loss_thres", 0.5),

    )
    train(dataset, trainable_model, config, test_function=lambda *args : test_function(*args, test_dataset=test_dataset))


if __name__ == "__main__":
    main()
