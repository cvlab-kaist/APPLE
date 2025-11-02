# CUDA_VISIBLE_DEVICES=0 python pulid_omini_inference_ffhq_args.py \
#     --base_path /home/work/.project/jiwon/fluxswap \
#     --ffhq_base_path /home/work/.project/jiwon/dataset/ffhq_eval \
#     --run_name '[DEBUG_randomidfalse]pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.4]_train[omini]' \
#     --ckpt step20000_global5000 \
#     --gpu_id 0 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 

# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_args_multigpu.py \
#     --base_path /home/work/.project/jiwon/fluxswap \
#     --ffhq_base_path /home/work/.project/jiwon/dataset/ffhq_eval \
#     --run_name '[DEBUG_randomidfalse]pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.3]_train[omini]' \
#     --ckpt step20000_global5000 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pulid_omini_inference_ffhq_args_multigpu.py \
#     --base_path /mnt/data3/jiwon/fluxswap \
#     --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
#     --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
#     --ckpt step60000_global15000 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 \
#     --condition_type 'blur_landmark_iris'

# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 torchrun --standalone --nproc_per_node=5 pulid_omini_inference_ffhq_inversion_args_multigpu.py \
#     --base_path /mnt/data3/jiwon/fluxswap \
#     --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
#     --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
#     --ckpt step60000_global15000 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 \
#     --condition_type 'blur_landmark_iris'

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_inversion_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name '[DEBUG_randomidfalse]pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step16000_global4000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step20000_global5000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step40000_global10000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_inversion_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step20000_global5000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_inversion_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step40000_global10000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step100000_global25000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 pulid_omini_inference_ffhq_inversion_args_multigpu.py \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval \
    --run_name 'baseline_dataset[vgg_aes5.1]_loss[maskid_netarc_t0.35]_loss[lpips_t0.35]_train[omini]' \
    --ckpt step100000_global25000 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --condition_type 'blur_landmark_iris'