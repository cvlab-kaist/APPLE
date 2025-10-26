import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time

from typing import List

import prodigyopt

from ..pipeline.flux_omini import transformer_forward, encode_images, transformer_forward_ca
from transformer_flux_ca import FluxTransformer2DModelCA

import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
import numpy as np


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"], # txt_n + img_n
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_netarc: bool = False,
        netarc_path: str = None,
        use_irse50: bool = False,
        id_loss_thres: float = 0.5,
        train_omini: bool = False,
        train_pulid_enc : bool = False,
        train_pulid_ca: bool = False,
        gaze_type : str = 'unigaze',
        train_gaze : bool = False,
        train_gaze_type : str = 'CA', # 'CA' for CrossAttn, 'temb' for time embedding, 'omini' for OminiControl
        train_gaze_loss: bool = False,
        train_gaze_loss_type: str = 'feature', # 'feature' for feature L2 loss, 'pred' for (yaw, pitch) prediction loss
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config


        # Load the Flux pipeline
        self.transformer: FluxTransformer2DModelCA = FluxTransformer2DModelCA.from_pretrained(flux_pipe_id, torch_dtype=dtype, subfolder='transformer', low_cpu_mem_usage=False, device_map=None, use_netarc=use_netarc, netarc_path=netarc_path, use_irse50=use_irse50, use_gaze=train_gaze, gaze_conditioning_type=train_gaze_type ,gaze_type=gaze_type, local_rank=get_rank())
        # self.transformer#.to(device)

        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype, transformer=self.transformer
        )# .to(device)

        # self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.train_omini = train_omini
        if not train_omini:
            # for p in self.lora_layers:
            #     p.requires_grad_(False)
            self.lora_layers = []

        # ID Loss settings
        self.id_loss_thres = id_loss_thres
        print(f"[INFO] ID loss threshold : t <= {self.id_loss_thres}")

        # add pulid training
        self.train_pulid_enc = train_pulid_enc
        self.train_pulid_ca = train_pulid_ca
        self.train_gaze = train_gaze
        self.train_gaze_type = train_gaze_type
        self.train_gaze_loss = train_gaze_loss
        self.train_gaze_loss_type = train_gaze_loss_type
        if train_pulid_enc:
            # self.transformer.pulid_encoder
            self.lora_layers += list(self.transformer.pulid_encoder.parameters())
            print(f"[INFO] Training PULID encoder")
        if train_pulid_ca:
            # self.transformer.pulid_cross_attention
            self.lora_layers += list(self.transformer.pulid_ca.parameters())
            print(f"[INFO] Training PULID cross-attention")
        if train_gaze:
            import torch.nn as nn
            if train_gaze_type == 'CA':
                self.lora_layers += list(self.transformer.gaze_ca.parameters())
                # for block in self.transformer.gaze_ca:
                #     for module in block.modules():
                #         # Initialize the parameters
                #         if isinstance(module, nn.LayerNorm):
                #             nn.init.constant_(module.weight, 1.0)
                #             nn.init.constant_(module.bias, 0.0)
                #         elif isinstance(module, nn.Linear):
                #             nn.init.xavier_uniform_(module.weight)
                #             if module.bias is not None:
                #                 nn.init.constant_(module.bias, 0.0)
                #         else:
                #             print(f"[WARNING] {module} not initialized.")
                #             raise NotImplementedError("Gaze CA init not implemented for module:", module)
                def init_gaze_module(m):
                    if isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(m.bias, 0.0)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)

                for block in self.transformer.gaze_ca:
                    block.apply(init_gaze_module)
            elif train_gaze_type == 'temb' or train_gaze_type == 'omini':
                self.lora_layers += list(self.transformer.gaze_temb_proj.parameters())
                def init_timestep_embedding(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)

                self.transformer.gaze_temb_proj.apply(init_timestep_embedding)
                print(f"[INFO] Training Gaze time embedding projection for train_gaze_type {train_gaze_type}")
            else:
                raise NotImplementedError("train_gaze_type not implemented:", train_gaze_type)


            print(f"[INFO] Using gaze conditioning type {train_gaze_type} in training")

            # Load GAZE model
            if train_gaze_loss:
                if gaze_type == 'gaze':
                    from model import Model 
                    self.GazeTR = Model().to(dtype)
                    self.GazeTR.load_state_dict(torch.load('/mnt/data2/jiwon/GazeTR/GazeTR-H-ETH.pt'))
                    self.GazeTR.eval()
                    self.GazeTR.requires_grad_(False)
                    print(f"[INFO] Loaded GazeTR model for gaze embedding extraction.")
                elif gaze_type == 'unigaze':
                    # --- Load Model ---
                    # "Main function to predict gaze from a single image."
                    from models.vit.mae_gaze import MAE_Gaze
                    self.mae_gaze = MAE_Gaze(model_type='vit_h_14') ## custom_pretrained_path does not matter because it will be overwritten by the UniGaze weight
                    weight = torch.load('./unigaze_h14_cross_X.pth.tar', map_location='cpu')['model_state']
                    self.mae_gaze.load_state_dict(weight, strict=True)
                    self.mae_gaze.eval().to(dtype)

        print(f"[INFO] Trainable parameters: {len(self.lora_layers)}")



    def to(self, *args):
        self.flux_pipe.to(*args)
        self.transformer.to(*args)
        if self.train_gaze_loss :
            self.GazeTR.to(*args)
        # self.lora_layers = [p.to(*args) for p in self.lora_layers]
        return self

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            for adapter_name in self.adapter_set:
                lora_file_path = os.path.join(lora_path)
                print(f"[INFO] Loading LoRA for adapter '{adapter_name}' from {lora_file_path}")
                self.transformer.load_lora_adapter(lora_file_path, adapter_name=adapter_name)
        else:
            print(f"[INFO] Initializing new LoRA adapters: {self.adapter_set}")
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
        
        lora_layers = filter(
            lambda p: p.requires_grad, self.transformer.parameters()
        )
        # named_lora_layers = [n for n, p in self.transformer.named_parameters() if p.requires_grad]
        # from pprint import pprint
        # print("[INFO] Trainable LoRA layers:")
        # pprint(named_lora_layers)
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )   

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        if self.train_gaze_loss:
            self.GazeTR.requires_grad_(False)
        if isinstance(self.transformer, FluxTransformer2DModelCA):
            self.transformer.components_requires_grad(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer


    def id_loss_func(self, x_in, targ, embedder, transforms=None):
        embedder.eval()

        def cosin_metric(x1, x2):
            return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

        id_loss = torch.tensor(0)

        # masked_input = x_in * self.mask
        masked_input = x_in

        # Add batch dimension if not exist
        if len(masked_input.shape) == 3:
            masked_input = masked_input.unsqueeze(0)
        if len(targ.shape) == 3:
            targ = targ.unsqueeze(0)

        # Resize    
        masked_input = F.interpolate(masked_input, (112, 112))
        targ         = F.interpolate(targ, (112, 112))

        # transform
        assert transforms is not None, "Please provide transforms for id loss"
        masked_input = transforms(masked_input) if transforms is not None else masked_input
        targ         = transforms(targ) if transforms is not None else targ

        # masked_input = self.image_augmentations(masked_input)

        src_id  = embedder(masked_input)
        src_id = F.normalize(src_id, p=2, dim=1) # (B, 512)

        targ_id = embedder(targ.to(x_in.device, x_in.dtype))
        targ_id = F.normalize(targ_id, p=2, dim=1) # (B, 512)

        dists   = 1 - cosin_metric(src_id, targ_id) # (B,)

        # # We want to sum over the averages
        # batch_size = x_in.shape[0]
        # for i in range(batch_size):
        #     id_loss = id_loss + dists[i:: batch_size].mean()

        return dists

    def training_step(self, batch, batch_idx):
        if self.global_step % 10 == 0:
            print(f"[DEBUG] Training step {self.global_step}, LoRA weight mean: {self.transformer.transformer_blocks[0].attn.to_q.lora_A['default'].weight.data.mean()}")

        # print ("start training step")
        imgs, prompts = batch["image"], batch["description"]
        height, width = imgs.shape[2], imgs.shape[3] # torch.Tensor (B, C, H, W), [0, 1]
        id_embed = batch['id_embed']
        image_latent_mask = batch.get("image_latent_mask", None)
        gaze_embed = batch.get("gaze_embed", None)
        # print(f"gaze_embed : {gaze_embed.shape if gaze_embed is not None else None}")
        if self.train_gaze :
            assert gaze_embed is not None, "[ERROR] Gaze embed is required for gaze training."

        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000): # 실제로는 0번만 사용
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"]) # torch.Tensor, [0,1]
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        # Prepare inputs
        with torch.no_grad():

            # Prepare image input
            # print(f"self.flux_pipe.device : {self.flux_pipe.device} self.flux_pipe.dtype : {self.flux_pipe.dtype}")
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # print("finish : encode image")

            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

            # print("finish : encode prompt")

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.flux_pipe.device))
            x_1 = torch.randn_like(x_0).to(self.flux_pipe.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.flux_pipe.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                # Prepare conditions
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                # Scale the position (see OminiConrtol2)
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                # Add position delta (see OminiControl)
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                if len(p_delta) > 1:
                    print("Warning: only the first position delta is used.")
                # Append to the list
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.flux_pipe.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # print("finish : prepare cond")

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.flux_pipe.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # Forward pass
        # check dtype of input
        # print (f"x_t dtype : {x_t.dtype}, condition_latents dtype : {[each.dtype for each in condition_latents]})")
        transformer_out = transformer_forward_ca(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            # There are three timesteps for the three branches
            # (text, image, and the condition)
            timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
            # Same as above
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            # The LoRA adapter names of each branch
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
            id_embed=id_embed.to(self.flux_pipe.dtype),
            id_weight=1.0,
            gaze_embed=gaze_embed.to(self.flux_pipe.dtype) if gaze_embed is not None else None,
            gaze_weight=1.0,
        )
        pred = transformer_out[0]

        # print("finish : forward")

        # Compute loss
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        recon_loss = step_loss

        # ID Loss
        if self.flux_pipe.transformer.netarc is not None or self.train_gaze_loss:

            # pred_x0 계산
            pred_x0 = x_t - t_.to(self.flux_pipe.dtype) * pred

            # VAE decode : pred_x0 -> 이미지
            pred_x0 = self.flux_pipe._unpack_latents(pred_x0, height, width, self.flux_pipe.vae_scale_factor)
            pred_x0 = (pred_x0 / self.flux_pipe.vae.config.scaling_factor) + self.flux_pipe.vae.config.shift_factor
            image = self.flux_pipe.vae.decode(pred_x0, return_dict=False)[0]
            x_in = self.flux_pipe.image_processor.postprocess(image, output_type='pt', do_denormalize=[False]) # [-1, 1]

            # DEBUG
            # image_pil = self.flux_pipe.image_processor.postprocess(image.detach(), output_type='pil') # List[PIL.Image]
            # breakpoint()
            
            if self.flux_pipe.transformer.netarc is not None:
                # id loss 계산
                src_imgs = batch['src_img'] # [0, 1]
                arc_src = src_imgs.detach().clone().to(x_in.device, x_in.dtype) # [0, 1]
                arc_pred   = ((x_in + 1) / 2).clamp(0,1) # [-1, 1] -> [0, 1]
                if self.flux_pipe.transformer.use_netarc:
                    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    id_loss_per_sample   = self.id_loss_func(arc_src, arc_pred, self.flux_pipe.transformer.netarc, transforms=transform) # (B,)
                elif self.flux_pipe.transformer.use_irse50:
                    transform = TF.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    id_loss_per_sample   = self.id_loss_func(arc_src, arc_pred, self.flux_pipe.transformer.netarc, transforms=transform) # (B,)
                else:
                    raise NotImplementedError("ID loss model not implemented.")
                
                # ID Loss를 적용할 timestep  : t에 따라서 weight 조절 -- t <= id_loss_thres 일때만 loss 적용
                t_mask = (t <= self.id_loss_thres).to(self.flux_pipe.dtype)
                id_loss = (id_loss_per_sample * t_mask).mean() 
                id_loss /= (t_mask.mean() + 1e-5) # Scaling by the number of valid samples


                step_loss = step_loss + id_loss

            if self.train_gaze_loss:
                if self.gaze_type == 'gaze':
                    if self.train_gaze_loss_type == 'feature':
                        x_in = ((x_in + 1) / 2).clamp(0,1) # [-1, 1] -> [0, 1], shape : (B, C, H, W)
                        # Resize to 224x224
                        x_in = F.interpolate(x_in, size=(224, 224), mode='bilinear',)
                        # RGB -> BGR
                        x_in_bgr = x_in[:, [2, 1, 0], :, :]
                        
                        img = {'face': x_in_bgr}

                        gaze_pred = self.GazeTR.forward_feature(img) # (B, 32)
                        gaze_target = gaze_embed.to(gaze_pred.device, gaze_pred.dtype).detach().clone()
                        gaze_loss_per_sample = F.mse_loss(gaze_pred, gaze_target, reduction='none').mean(dim=1) # (B,)

                        # timestep t에 따라서 weight 조절 : t<=0.5일때만 loss 적용
                        t_mask = (t <= 0.5).to(self.flux_pipe.dtype)

                        # drop_gaze가 True인 경우 Mask
                        drop_gaze = batch.get("drop_gaze")
                        drop_mask = (~drop_gaze).to(self.flux_pipe.dtype)

                        # Combine masks
                        gaze_mask = t_mask * drop_mask # (B,)
                        print(f"gaze_mask.shape : {gaze_mask.shape}, ")

                        gaze_loss = (gaze_loss_per_sample * gaze_mask).mean()
                        gaze_loss /= (gaze_mask.mean() + 1e-5) # Scaling by the number of valid samples
                    elif self.train_gaze_loss_type == 'pred':
                        trg_imgs = imgs.clone().to(x_in.device, x_in.dtype) # [0, 1]
                        pred_imgs = x_in.clone().to(x_in.device, x_in.dtype)
                        pred_imgs = ((pred_imgs + 1) / 2).clamp(0,1) # [-1, 1] -> [0, 1], shape : (B, C, H, W)

                        # Resize to 224x224
                        trg_imgs = F.interpolate(trg_imgs, size=(224, 224), mode='bilinear',)
                        pred_imgs = F.interpolate(pred_imgs, size=(224, 224), mode='bilinear',)

                        # RGB -> BGR
                        trg_imgs_bgr = trg_imgs[:, [2, 1, 0], :, :]
                        pred_imgs_bgr = pred_imgs[:, [2, 1, 0], :, :]
                        trg_img = {'face': trg_imgs_bgr}
                        pred_img = {'face': pred_imgs_bgr}

                        gaze_trg = self.GazeTR(trg_img) # (B, 2)
                        gaze_pred = self.GazeTR(pred_img) # (B, 2)

                        gaze_loss_per_sample = F.mse_loss(gaze_pred, gaze_trg, reduction='none').mean(dim=1) # (B,)

                        # timestep t에 따라서 weight 조절 : t<=0.5일때만 loss 적용
                        t_mask = (t <= 0.25).to(self.flux_pipe.dtype)
                        gaze_loss = (gaze_loss_per_sample * t_mask).mean()
                        gaze_loss /= (t_mask.mean() + 1e-5) # Scaling by the number of valid samples
                    else:
                        raise NotImplementedError("train_gaze_loss_type not implemented:", self.train_gaze_loss_type)
                elif self.gaze_type == 'unigaze':
                    if self.train_gaze_loss_type == 'feature':
                        trg_imgs = imgs.clone().to(x_in.device, x_in.dtype) # [0, 1]
                        pred_imgs = x_in.clone().to(x_in.device, x_in.dtype)
                        pred_imgs = ((pred_imgs + 1) / 2).clamp(0,1) # [-1, 1] -> [0, 1], shape : (B, C, H, W)

                        # Resize to 224x224
                        trg_imgs = F.interpolate(trg_imgs, size=(224, 224), mode='bilinear',)
                        pred_imgs = F.interpolate(pred_imgs, size=(224, 224), mode='bilinear',)

                        # RGB -> BGR
                        trg_imgs_bgr = trg_imgs[:, [2, 1, 0], :, :]
                        pred_imgs_bgr = pred_imgs[:, [2, 1, 0], :, :]
                        gaze_target = gaze_embed.to(gaze_pred.device, gaze_pred.dtype).detach().clone()
                        gaze_loss_per_sample = F.mse_loss(gaze_pred, gaze_target, reduction='none').mean(dim=1) # (B,)

                        # timestep t에 따라서 weight 조절 : t <=0.5일때만 loss 적용
                        t_mask = (t <= 0.5).to(self.flux_pipe.dtype)

                        # drop_gaze가 True인 경우 Mask
                        drop_gaze = batch.get("drop_gaze")
                        drop_mask = (~drop_gaze).to(self.flux_pipe.dtype)

                        # Combine masks
                        gaze_mask = t_mask * drop_mask # (B,)

                        gaze_loss = (gaze_loss_per_sample * gaze_mask).mean()
                        gaze_loss /= (gaze_mask.mean() + 1e-5) # Scaling by the number of valid samples
                    else:
                        raise NotImplementedError("train_gaze_loss_type not implemented for unigaze:", self.train_gaze_loss_type)


                step_loss = step_loss + gaze_loss


    

        self.last_t = t.mean().item()
        self.last_id_loss = id_loss.item() if self.flux_pipe.transformer.netarc is not None else 0.0
        self.last_recon_loss = recon_loss.item()
        self.last_gaze_loss = gaze_loss.item() if self.train_gaze_loss else 0.0

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )
        print("[INFO] Use WanDB:", self.use_wandb)

        self.total_steps = 0
        self.test_function = test_function

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "global_steps": pl_module.global_step,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            report_dict["recon_loss"] = pl_module.last_recon_loss
            if pl_module.last_id_loss > 0.0 :
                report_dict["id_loss"] = pl_module.last_id_loss
            if pl_module.last_gaze_loss > 0.0 :
                report_dict["gaze_loss"] = pl_module.last_gaze_loss
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Global Steps: {pl_module.global_step}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f} Recon loss: {pl_module.last_recon_loss:.4f}, ID loss: {pl_module.last_id_loss:.4f}, Gaze loss: {pl_module.last_gaze_loss:.4f}"
            )

        # Save LoRA weights at specified intervals
        if ((pl_module.global_step % self.save_interval == 0 and pl_module.global_step != 0) or self.total_steps == 1):
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Global Steps: {pl_module.global_step}, - Saving LoRA weights"
            )
            ckpt_path = f"{self.save_path}/{self.run_name}/ckpt/step{self.total_steps}_global{pl_module.global_step}"
            os.makedirs(ckpt_path, exist_ok=True)
            if pl_module.train_omini:
                pl_module.save_lora(
                    ckpt_path
                )
            if pl_module.train_pulid_enc:
                torch.save(pl_module.transformer.pulid_encoder.state_dict(), f"{ckpt_path}/pulid_encoder.pth")
            if pl_module.train_pulid_ca:
                torch.save(pl_module.transformer.pulid_ca.state_dict(), f"{ckpt_path}/pulid_ca.pth")
            if pl_module.train_gaze:
                if pl_module.train_gaze_type == 'temb' or pl_module.train_gaze_type == 'omini':
                    torch.save(pl_module.transformer.gaze_temb_proj.state_dict(), f"{ckpt_path}/gaze_temb_proj.pth")
                elif pl_module.train_gaze_type == 'CA':
                    torch.save(pl_module.transformer.gaze_ca.state_dict(), f"{ckpt_path}/gaze_ca.pth")
                else:
                    raise NotImplementedError("train_gaze_type not implemented:", pl_module.train_gaze_type)

        # Generate and save a sample image at specified intervals
        if ((pl_module.global_step % self.save_interval == 0 and pl_module.global_step != 0) or self.total_steps == 1) and self.test_function:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Global Steps: {pl_module.global_step} - Generating a sample"
            )
            pl_module.eval()
            trg_imgs, condition_imgs, result_imgs, src_imgs = self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}_global_{pl_module.global_step}.png",
            )

            import matplotlib.pyplot as plt
            from PIL import Image
            import io  # to handle in-memory binary streams

            grid_images = []
            for trg, cond, res, src in zip(trg_imgs, condition_imgs, result_imgs, src_imgs):
                fig, axs = plt.subplots(1, 4, figsize=(12, 3))  # 4-column subplot
                for ax, img, title in zip(axs, [src, cond, res, trg], ['Source', 'Condition', 'Result', 'Target']):
                    ax.imshow(img)
                    ax.set_title(title)
                    ax.axis('off')
                fig.tight_layout()

                # Convert the figure to PIL Image
                buf = io.BytesIO()
                fig.savefig(buf, format='png')  # save figure to in-memory buffer
                buf.seek(0)  # rewind to beginning
                pil_img = Image.open(buf).convert("RGB")  # open as PIL Image and ensure RGB
                grid_images.append(pil_img)

                plt.close(fig)  # close to free memory


            if self.use_wandb :
                # wandb.log({
                #     'original' : [wandb.Image(image) for image in trg_imgs],
                #     'condition' : [wandb.Image(image) for image in condition_imgs],
                #     'result' : [wandb.Image(image) for image in result_imgs],
                # })
                wandb.log({
                    'comparison': [wandb.Image(image) for image in grid_images],
                })
            pl_module.train()


def train(dataset, trainable_model, config, test_function):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = training_config.get("run_name", None)
    # run_name = run_name + '_' + time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)
