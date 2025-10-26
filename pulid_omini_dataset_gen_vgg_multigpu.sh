export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 pulid_omini_dataset_gen_vgg_multigpu.py
