# *[Specify the config file path and the GPU devices to use]
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# *[Specify the config file path]
export OMINI_CONFIG='./train/config/faceswap_vgg_maskidLoss.yaml'
export WANDB_API_KEY='f9831e23517e27f7ecac9b54bc2cdcabb3af8c33'

# *[Specify the WANDB API key]
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

# python -m pdb -m omini.train_flux.train_faceswap
# python -m omini.train_flux.train_faceswap

accelerate launch --main_process_port 29502 -m omini.train_flux.train_faceswap