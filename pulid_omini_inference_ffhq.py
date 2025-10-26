import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformer_flux_ca import FluxTransformer2DModelCA
# from diffusers import FluxTransformer2DModel
import torch

guidance_scale=1.0
image_guidance_scale = 1.0
id_guidance_scale = 1.0
use_gaze = False


weight_dtype = torch.bfloat16
device = 'cuda'
# transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer')
# transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None)
transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None, use_gaze=use_gaze)
# no grad
torch.set_grad_enabled(False)

from pulid.utils import resize_numpy_image_long
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_txt
from diffusers import FluxPipeline
from pipeline_flux_ca import FluxPipelineCA
from omini.pipeline.flux_omini import generate_ca, Condition, convert_to_condition


device = 'cuda'
flux = FluxPipelineCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", transformer=transformer, torch_dtype=weight_dtype).to(device)


# ckpt = 43000
# lora_file_path = f'/mnt/data3/jiwon/OminiControl/runs/faceswap_scratch_lora64_20251004-005548/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}/ffhq_eval'

# ckpt = 60000
# lora_file_path = f'/mnt/data3/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_20251010-115546/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 8000
# lora_file_path = f'/mnt/data3/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_20251014-014645/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_idLoss_t<=0.33_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 8000
# lora_file_path = f'/mnt/data3/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_t<=0.5_20251014-021510/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/pulid_omini_vgg_idLoss_t<=0.5_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 80000
# lora_file_path = f'/mnt/data3/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_20251014-162532/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'

# ckpt = 8000
# lora_file_path = f'/mnt/data3/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt60000_gaze_20251018-024629/ckpt/{ckpt}/default.safetensors'
# output_dir = f'./results/faceswap_vgg_lora64Pretrained_idLoss_irse50_t<=0.5_ckpt60000_gaze_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval'


ckpt = 20000
lora_file_path = f'/mnt/data3/jiwon/fluxswap/runs/pretrained[ffhq43k]_dataset[vgg]_loss[maskid_netarc_t0.5]_train[omini]/ckpt/{ckpt}/default.safetensors'
output_dir = f'./results/pretrained[ffhq43k]_dataset[vgg]_loss[maskid_netarc_t0.5]_train[omini]_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval/base'

adapter_name = 'default'
print(f"Loading LoRA for adapter '{adapter_name}' from {lora_file_path}")
flux.load_lora_weights(lora_file_path, adapter_name=adapter_name)
if use_gaze:
    flux.transformer.gaze_ca.load_state_dict(torch.load(
        os.path.join(os.path.dirname(lora_file_path), 'gaze_ca.pth')    
    ))
    print(f"[INFO] Loaded gaze CA weights from {os.path.join(os.path.dirname(lora_file_path), 'gaze_ca.pth')}")

flux.transformer.components_to_device(device, weight_dtype)

# Compile the model with torch.compile for faster inference
flux.transformer = torch.compile(flux.transformer, mode="reduce-overhead", fullgraph=True)



import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image

# output_dir = './results_pulid_omini/vgg_src/clsUncond_hiddenUncond'
os.makedirs(output_dir, exist_ok=True)

src_img_path_list = sorted(glob('/mnt/data2/dataset/ffhq_eval/src/*.jpg'))
trg_img_path_base = '/mnt/data2/dataset/ffhq_eval/trg'

# for id_guidance_scale in [1.0]:
# for id_guidance_scale in [1.5]:
# for id_guidance_scale in [1.0, 1.5, 2.0]:
# for id_guidance_scale in [1.0, 1.5, 2.0]:
for id_guidance_scale in [1.0]:
    for src_img_path in tqdm(src_img_path_list, desc='Processing Image'):
        prompt="a photo of human face",
        neg_prompt = ""
        true_cfg = 1.0
        use_true_cfg = True if true_cfg > 1.0 else False

        src_num = os.path.basename(src_img_path).split('.')[0]
        trg_img_path = os.path.join (trg_img_path_base, 'condition_blended_image_blurdownsample8_segGlass_landmark', f"{src_num}.png")
        condition_img = Image.open(trg_img_path).convert('RGB')
        
        if use_gaze:
            import numpy as np
            gaze_path = os.path.join (trg_img_path_base, 'gaze', f"{src_num}.npy")
            gaze_embed = torch.from_numpy(  np.load(gaze_path) ).unsqueeze(0).to(device, dtype=weight_dtype) # (1, gaze_dim)
        else:
            gaze_embed = None
        
        # Resize
        condition_img = condition_img.resize((512,512))
        condition_type = 'deblurring'
        position_delta = [0,0]
        position_scale = 1.0
        condition = Condition(condition_img, 'default', position_delta, position_scale)


        id_image = cv2.imread(src_img_path)
        print(f"id_image shape: {id_image.shape}")
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
            raise RuntimeError('id embedding is nan')

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

        img_save_fname = f"{output_dir}/{src_num}.png"
        if os.path.exists(img_save_fname):
            print(f"Image {img_save_fname} already exists. Skipping...")
            continue

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
        image = img.images[0] if isinstance(img.images, list) else img.images
        image.save(f"{output_dir}/{src_num}.png")
        # condition_img.save(f"{output_dir}/{src_num}_cond.png")

        grid = make_image_grid([id_image_pil, condition_img, image], rows=1, cols=3)
        os.makedirs(f"{output_dir}/grid", exist_ok=True)
        grid.save(f"{output_dir}/grid/{src_num}_grid.png")

