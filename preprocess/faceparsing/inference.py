import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from PIL import Image
from models.parsing import BiSeNet
from glob import glob
from utils.module import SpecificNorm, cosin_metric
from torchvision import transforms
from torchvision.transforms import Resize
from torchvision.transforms import functional as TF
import numpy as np
from tqdm import tqdm


imgs_list = []
from glob import glob
img_dir_path = '/mnt/data2/jiwon/AnimateDiff/SamsungDataset/firstReport/trg_imgs'
img_files = glob(f'{img_dir_path}/*.png')
img_files = [img_path for img_path in img_files if '_mask' not in img_path and '_segmap' not in img_path]
# sort
# img_files = sorted(img_files, key=lambda x: int(os.path.basename(os.path.dirname(x)).split('_')[-1]))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

netSeg = BiSeNet(n_classes=19).to(device).eval()
spNorm = SpecificNorm().to(device).eval()
netSeg.load_state_dict(torch.load('./checkpoints/FaceParser.pth', weights_only=False))

# Attributes = [0, 'background', 1 'skin', 2 'r_brow', 3 'l_brow', 4 'r_eye', 5 'l_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
color_list = [[0, 0, 0], [255, 0, 0], [0, 204, 204], [0, 0, 204], [255, 153, 51], [204, 0, 204], [0, 0, 0],
                [204, 0, 0], [102, 51, 0], [0, 0, 0], [76, 153, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153],
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]


target_hw_list = [None, 256, 512, 1024]

for target_hw in target_hw_list : 
    for img_path in tqdm(img_files, total=len(img_files)):
        save_dir = os.path.dirname(img_path)
        fname = os.path.basename(img_path).split('.')[0]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size  

        if target_hw is None :
            hw = min(width, height)
        else :
            hw = target_hw

        print(f'Processing {img_path} with size {hw}x{hw}')
        
        transform = transforms.Compose([
            transforms.Resize((hw, hw)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor(),])
        
        img_input = transform(img).to(device)
        if len(img_input.shape) == 3:
            img_input = img_input.unsqueeze(0)
        targ_mask = netSeg(spNorm(img_input))[0]
        # targ_mask = transforms.Resize((height, width))(targ_mask)
        parsing   = targ_mask.squeeze(0).detach().cpu().numpy().argmax(0)
        targ_base = np.zeros((hw, hw, 3))
        targ_mask = np.zeros((hw, hw, 3))
        for idx, color in enumerate(color_list):
            targ_base[parsing == idx] = color
            if idx != 0 :
                if color != [0, 0, 0]:
                    targ_mask[parsing == idx] = [255, 255, 255]

        targ_base_PIL = Image.fromarray(targ_base.astype(np.uint8))
        targ_base_PIL.save(os.path.join(save_dir, f'{fname}_segmap{target_hw}.png'))
        targ_mask_PIL = Image.fromarray(targ_mask.astype(np.uint8))
        targ_mask_PIL.save(os.path.join(save_dir, f'{fname}_mask{target_hw}.png'))
