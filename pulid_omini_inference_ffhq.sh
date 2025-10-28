# python pulid_omini_inference_ffhq_args.py \
#     --run_name pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.3]_loss[lpips_t0.3]_train[omini]_globalresume2K \
#     --ckpt step32000_global8000 \
#     --gpu_id 0 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 \
#     --base_path /home/work/.project/jiwon/fluxswap \
#     --ffhq_base_path /home/work/.project/jiwon/dataset/ffhq_eval


# python pulid_omini_inference_ffhq_args.py \
#     --run_name pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.5]_train[omini]_ver2_globalresume2K\
#     --ckpt step41000_global8000 \
#     --gpu_id 0 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 \
#     --base_path /mnt/data3/jiwon/fluxswap \
#     --ffhq_base_path /mnt/data2/dataset/ffhq_eval

CUDA_VISIBLE_DEVICES=1 python pulid_omini_inference_ffhq_args.py \
    --run_name pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.5]_loss[lpips_t0.5]_train[omini]_globalresume2K \
    --ckpt step32000_global8000 \
    --gpu_id 0 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 \
    --base_path /mnt/data3/jiwon/fluxswap \
    --ffhq_base_path /mnt/data2/dataset/ffhq_eval