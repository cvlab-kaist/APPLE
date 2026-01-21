CUDA_VISIBLE_DEVICES=0,2,3,5,6,7  torchrun --standalone --nproc_per_node=6 pulid_omini_inference_ffhq_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data3/jiwon/fluxswap/reface_dataset/dataset \
    --run_name 'stage2_flux[dev]_dataset[ours50kV2_ckpt50k]_loss[maskid_netarc_t0.5]_loss[lpips_t0.5]_train[omini]_init[50k]' \
    --ckpt step59991_global15000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 