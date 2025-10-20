
from os import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from models.vit.mae import interpolate_pos_embed, MaskedAutoencoderViT, vit_base_patch16, vit_large_patch16, vit_huge_patch14




class MAE_Gaze(nn.Module):

	def __init__(self, model_type='vit_b_16', global_pool=False, drop_path_rate=0.1,
			  custom_pretrained_path=None):
		
		super().__init__()
		if model_type == "vit_b_16":
			self.vit = vit_base_patch16( global_pool=global_pool, drop_path_rate=drop_path_rate)
		elif model_type == "vit_l_16":
			self.vit = vit_large_patch16( global_pool=global_pool, drop_path_rate=drop_path_rate)
		elif model_type == "vit_h_14":
			self.vit = vit_huge_patch14( global_pool=global_pool, drop_path_rate=drop_path_rate)
		else:
			raise ValueError('model_type not supported')

		if custom_pretrained_path is not None:
			checkpoint_model = torch.load(custom_pretrained_path, map_location='cpu')['model']
			state_dict = self.vit.state_dict()
			for k in  ['head.weight', 'head.bias']:
				if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
					print(f"Removing key {k} from pretrained checkpoint")
					del checkpoint_model[k]
			


			# interpolate position embedding
			interpolate_pos_embed(self.vit, checkpoint_model)
		
			# keys_in_ckpt = checkpoint_model.keys()
			# print('Keys in ckpt: ', keys_in_ckpt)
			self.vit.load_state_dict( checkpoint_model, strict=False)
			print('Loaded custom pretrained weights from {}'.format(custom_pretrained_path))

		# del self.decoder_embed
		# del self.mask_token
		# del self.decoder_pos_embed
		# del self.decoder_blocks
		# del self.decoder_norm
		# del self.decoder_pred

		embed_dim = self.vit.embed_dim
		self.gaze_fc = nn.Linear(embed_dim, 2)


	def forward(self, input):
		features = self.vit.forward_features(input)

		pred_gaze = self.gaze_fc(features)
		output_dict = {}
		output_dict['pred_gaze'] = pred_gaze
		output_dict['features'] = features
		return output_dict
	