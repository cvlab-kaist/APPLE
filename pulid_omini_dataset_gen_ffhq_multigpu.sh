export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 pulid_omini_dataset_gen_ffhq_multigpu.py