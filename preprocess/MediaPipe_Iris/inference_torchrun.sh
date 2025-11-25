export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
torchrun --standalone --nproc_per_node=6 inference_torchrun.py