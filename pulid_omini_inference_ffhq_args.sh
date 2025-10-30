# CUDA_VISIBLE_DEVICES=0 python pulid_omini_inference_ffhq_args.py \
#     --base_path /home/work/.project/jiwon/fluxswap \
#     --ffhq_base_path /home/work/.project/jiwon/dataset/ffhq_eval \
#     --run_name '[DEBUG_randomidfalse]pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.4]_train[omini]' \
#     --ckpt step20000_global5000 \
#     --gpu_id 0 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 

# CUDA_VISIBLE_DEVICES=1 python pulid_omini_inference_ffhq_args.py \
#     --base_path /home/work/.project/jiwon/fluxswap \
#     --ffhq_base_path /home/work/.project/jiwon/dataset/ffhq_eval \
#     --run_name '[DEBUG_randomidfalse]pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.5]_train[omini]' \
#     --ckpt step20000_global5000 \
#     --gpu_id 0 \
#     --guidance_scale 1.0 \
#     --image_guidance_scale 1.0 \
#     --id_guidance_scale 1.0 

CUDA_VISIBLE_DEVICES=0 python pulid_omini_inference_ffhq_args.py \
    --base_path /home/work/.project/jiwon/fluxswap \
    --ffhq_base_path /home/work/.project/jiwon/dataset/ffhq_eval \
    --run_name '[DEBUG_randomidfalse]pretrained[ffhq43K]_dataset[vgg]_loss[maskid_netarc_t0.3]_train[omini]' \
    --ckpt step20000_global5000 \
    --gpu_id 0 \
    --guidance_scale 1.0 \
    --image_guidance_scale 1.0 \
    --id_guidance_scale 1.0 