CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_inversion_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step60000_global15000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris' \
    --second_order