# %%
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change the value to select a different GPU
import torch
from diffusers import FluxPipeline

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

prompt = "A frog holding a sign that says hello world"
image = pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=4.5,
).images[0]
image.save("flux-krea-dev.png")



