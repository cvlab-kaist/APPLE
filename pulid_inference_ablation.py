# %%
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from transformer_flux_ca import FluxTransformer2DModelCA
# from diffusers import FluxTransformer2DModel
import torch

weight_dtype = torch.bfloat16
device = 'cuda'
# transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer')
# transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None)
transformer = FluxTransformer2DModelCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=weight_dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None)

# %%
from pulid.utils import resize_numpy_image_long
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_txt
from diffusers import FluxPipeline
from pipeline_flux_ca import FluxPipelineCA

device = 'cuda'
flux = FluxPipelineCA.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", transformer=transformer, torch_dtype=weight_dtype).to(device)


# %%
flux.transformer.components_to_device(device, weight_dtype)


from diffusers.utils import load_image
import numpy as np
from pulid.utils import img2tensor, tensor2img
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize



import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image

output_dir = './ablation_results/pulid_id_clip_swap_ablation_w_vgg_data'
os.makedirs(output_dir, exist_ok=True)

seed = 0
g=3.5

src_img_path_list = sorted(glob('/mnt/data2/dataset/vgg_data/src/*.jpg'))
for src_img_path, trg_img_path in tqdm(zip(src_img_path_list[:-1], src_img_path_list[1:]), desc='Processing Image'):
    prompt="a photo of human face",
    neg_prompt = ""
    true_cfg = 1.0
    use_true_cfg = True if true_cfg > 1.0 else False

    src_num = os.path.basename(src_img_path).split('.')[0]
    id_image = cv2.imread(src_img_path)
    print(f"id_image shape: {id_image.shape}")
    id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
    id_image = resize_numpy_image_long(id_image, 1024)
    id_image_pil = Image.fromarray(id_image)
    id_image_pil.save(f"{output_dir}/{src_num}_id.png")

    clip_image = cv2.imread(trg_img_path)
    clip_image = cv2.cvtColor(clip_image, cv2.COLOR_BGR2RGB)
    clip_image = resize_numpy_image_long(clip_image, 1024)
    clip_image_pil = Image.fromarray(clip_image)
    clip_image_pil.save(f"{output_dir}/{src_num}_clip.png")

    # flux.t5, flux.clip = flux.t5.to(device), flux.clip.to(device)
    # flux.pulid_model.components_to_device(device)

    id_embeddings, uncond_id_embeddings = flux.transformer.get_id_embedding(id_image, cal_uncond=use_true_cfg)
    id_embeddings = id_embeddings.to(device, dtype=weight_dtype)
    if use_true_cfg:
        uncond_id_embeddings = uncond_id_embeddings.to(device, dtype=weight_dtype)
    
    id_clip_swap_embeddings = flux.transformer.get_id_embedding_from_id_and_clip(id_image, clip_image)
    id_clip_swap_embeddings = id_clip_swap_embeddings.to(device, dtype=weight_dtype)

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

    img = flux(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=g,
        generator=torch.Generator('cpu').manual_seed(seed),
        id_embed=id_embeddings,
        uncond_id_embed=uncond_id_embeddings,
    )
    image_default = img.images[0] if isinstance(img.images, list) else img.images
    image_default.save(f"{output_dir}/{src_num}.png")

    img = flux(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=g,
        generator=torch.Generator('cpu').manual_seed(seed),
        id_embed=id_clip_swap_embeddings,
        uncond_id_embed=uncond_id_embeddings,
    )
    image_swapped = img.images[0] if isinstance(img.images, list) else img.images
    image_swapped.save(f"{output_dir}/{src_num}_swapped.png")

    grid = make_image_grid([id_image_pil, clip_image_pil, image_default, image_swapped], rows=1, cols=4)
    os.makedirs(f"{output_dir}/grid", exist_ok=True)
    grid.save(f"{output_dir}/grid/{src_num}_grid.png")




