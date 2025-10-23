export CUDA_VISIBLE_DEVICES=0,1,5,6
torchrun --nproc_per_node=4 pulid_omini_dataset_gen_vgg_multigpu.py
