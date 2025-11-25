export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3,4,5,6
torchrun --standalone --nproc_per_node=5 inference_ffhq_eval_torchrun.py