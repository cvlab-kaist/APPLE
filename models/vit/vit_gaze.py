
from os import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32



class ViTGaze(nn.Module):
	def __init__(self, 
				vit_type="b_16", 
				pretrained=True,
				custom_pretrained_path=None,
				**kwargs
				):
		super().__init__()
		if vit_type == "b_16":
			"""
			patch_size=16,
			num_layers=12,
			num_heads=12,
			hidden_dim=768,
			mlp_dim=3072,
			"""
			self.vit = vit_b_16(pretrained=pretrained )


			self.vit.heads = nn.Sequential(
				nn.Linear(768,2)
			)
		elif vit_type == "b_32":
			self.vit = vit_b_32(pretrained=pretrained)
			self.vit.heads = nn.Sequential(
				nn.Linear(768,2)
			)
		elif vit_type == "l_16":
			self.vit = vit_l_16(pretrained=pretrained)
			self.vit.heads = nn.Sequential(
				nn.Linear(1024,2)
			)
		elif vit_type == "l_32":
			self.vit = vit_l_32(pretrained=pretrained)
			self.vit.heads = nn.Sequential(
				nn.Linear(1024,2)
			)
		if custom_pretrained_path is not None:
			ckpt = torch.load(custom_pretrained_path)
			print('Loading custom pretrained weights from: ', custom_pretrained_path)
			# self.vit.load_state_dict( ckpt['model'], strict=True)
			keys_in_ckpt = ckpt.keys()
			print('Keys in ckpt: ', keys_in_ckpt)
			self.vit.load_state_dict( ckpt, strict=True)

	def forward(self, x_in):
		out_dict = {}
		gaze = self.vit(x_in)
		out_dict['pred_gaze'] = gaze
		return out_dict


	
from models.vit.mae import interpolate_pos_embed, vit_huge_patch14
class CustomViT_H14(nn.Module):
	def __init__(self, global_pool=False, drop_path_rate=0.1,
			  custom_pretrained_path=None):
		super().__init__()
		self.vit = vit_huge_patch14( global_pool=global_pool, drop_path_rate=drop_path_rate)
		
		if custom_pretrained_path is not None:
			checkpoint_model = torch.load(custom_pretrained_path, map_location='cpu')

			state_dict = self.vit.state_dict()
			for k in  ['head.weight', 'head.bias']:
				if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
					print(f"Removing key {k} from pretrained checkpoint")
					del checkpoint_model[k]
			
			# interpolate position embedding
			interpolate_pos_embed(self.vit, checkpoint_model)
		
			self.vit.load_state_dict( checkpoint_model, strict=False )
			print('Loaded custom pretrained weights from {}'.format(custom_pretrained_path))

		embed_dim = self.vit.embed_dim
		self.gaze_fc = nn.Linear(embed_dim, 2)


	def forward(self, input):
		features = self.vit.forward_features(input)

		pred_gaze = self.gaze_fc(features)
		output_dict = {}
		output_dict['pred_gaze'] = pred_gaze
		return output_dict
	


		