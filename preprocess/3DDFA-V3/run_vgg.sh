export CUDA_VISIBLE_DEVICES=0

python  demo_from_folder_jiwon_vgg.py --inputpath None \
    --savepath None \
    --device cuda --iscrop 1 --detector retinaface \
    --ldm68 1 --useTex 1 --extractTex 1 \
    --backbone resnet50


# python demo.py --inputpath examples/ --savepath examples/results --device cuda --iscrop 1 --detector retinaface --ldm68 1 --ldm106 1 --ldm106_2d 1 --ldm134 1 --seg_visible 1 --seg 1 --useTex 1 --extractTex 1 --backbone resnet50
