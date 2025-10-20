# *[Specify the config file path and the GPU devices to use]
export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# *[Specify the config file path]
export OMINI_CONFIG=./train/config/spatial_alignment.yaml

# *[Specify the WANDB API key]
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

# python -m omini.train_flux.train_spatial_alignment

accelerate launch -m omini.train_flux.train_spatial_alignment