import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformer_flux_ca import FluxTransformer2DModelCA
# from diffusers import FluxTransformer2DModel
import torch

weight_dtype = torch.bfloat16
device = 'cuda'
# transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer')
# transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None)
transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None)

from pulid.utils import resize_numpy_image_long
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_txt
from diffusers import FluxPipeline
from pipeline_flux_ca import FluxPipelineCA
from omini.pipeline.flux_omini import generate_ca, Condition, convert_to_condition




device = 'cuda'
flux = FluxPipelineCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", transformer=transformer, torch_dtype=weight_dtype).to(device)

ckpt = 43000
lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_scratch_lora64_20251004-005548/ckpt/{ckpt}/default.safetensors'
# ckpt = 60000
# lora_file_path = f'/mnt/data2/jiwon/OminiControl/runs/faceswap_vgg_lora64Pretrained_20251010-115546/ckpt/{ckpt}/default.safetensors'
adapter_name = 'default'
print(f"Loading LoRA for adapter '{adapter_name}' from {lora_file_path}")
flux.load_lora_weights(lora_file_path, adapter_name=adapter_name)

flux.transformer.components_to_device(device, weight_dtype)



import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image

output_dir = f'./results/pulid_omini_ckpt{ckpt}/vgg_data_eval'
# output_dir = f'./results/pulid_omini_vgg_ckpt{ckpt}/vgg_data_eval'
# output_dir = './results_pulid_omini/vgg_src/clsUncond_hiddenUncond'
os.makedirs(output_dir, exist_ok=True)

src_img_path_list = sorted(glob('/mnt/data2/dataset/vgg_data_eval/src/*.jpg'))
trg_img_path_base = '/mnt/data2/dataset/vgg_data_eval/trg'

for src_img_path in tqdm(src_img_path_list, desc='Processing Image'):
    prompt="a photo of human face",
    neg_prompt = ""
    true_cfg = 1.0
    use_true_cfg = True if true_cfg > 1.0 else False

    src_num = os.path.basename(src_img_path).split('.')[0]
    trg_img_path = os.path.join (trg_img_path_base, 'condition_blended_image_blurdownsample8_segGlass_landmark', f"{src_num}.png")
    condition_img = Image.open(trg_img_path).convert('RGB')
    condition_type = 'deblurring'
    position_delta = [0,0]
    position_scale = 1.0
    condition = Condition(condition_img, 'default', position_delta, position_scale)


    id_image = cv2.imread(src_img_path)
    print(f"id_image shape: {id_image.shape}")
    id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
    id_image = resize_numpy_image_long(id_image, 1024)
    id_image_pil = Image.fromarray(id_image)
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

