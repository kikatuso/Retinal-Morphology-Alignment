

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from functools import partial
from .lpips import LossTerm
import numpy as np
from collections import namedtuple


RETFOUND_CHECKPOINT = '/well/papiez/users/zwk579/Analysis/DiffusionModels/models/RETFound_MAE/weights/RETFound_mae_natureCFP.pth'

imagenet_mean =  torch.from_numpy(np.array([0.485, 0.456, 0.406])).view(1, 3, 1, 1).float()
imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).view(1, 3, 1, 1).float()


def RETFound_LPIPS(normalize,checkpoint=RETFOUND_CHECKPOINT,**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_backbone = RETFound_encoder(global_pool=False)
    mae_load_weights_from_pth(model_backbone, checkpoint,ignore_keys=['decoder','mask_token'])
    model = TIM_VIT_EXTRACTOR(model_backbone)
    return VitLPIPS(model=model,normalize=normalize, **kwargs).to(device)



def RETFound_encoder(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.head = nn.Identity()
    return model

class TIM_VIT_EXTRACTOR(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model  = model # has to be based on Timm VIT 

        #  retfound is a ViT large with 24 transformer blocks
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.N_slices = 4
        for x in range(4):
            self.slice1.add_module(str(x), self.model.blocks[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.model.blocks[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.model.blocks[x])
        for x in range(16, 24):
            self.slice4.add_module(str(x), self.model.blocks[x])

    
    def patch_embedder(self, x):
        B = x.shape[0]
        x = self.model.patch_embed(x)
        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed # type: ignore
        if hasattr(self.model, 'pos_drop'):
            x = self.model.pos_drop(x)
        return x

    def forward(self, X):
        h = self.patch_embedder(X)
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VITOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
    
class VitLPIPS(LossTerm):
    def __init__(self,model,retfound_weight=1.0,retfound_factor=1.0,retfound_start_iter=None,retfound_start_epoch=None,eval_mode=True,**kwargs):
        super().__init__(retfound_weight,retfound_factor,retfound_start_iter,retfound_start_epoch)
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        if eval_mode:
            self.eval()
        
    def resize(self, img):
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        return img

    def forward(self, input, target):
        # Resize images to 224x224
        if input.shape[-1] != 224:
            input = self.resize(input)
        if target.shape[-1] != 224:
            target = self.resize(target)
        

        outs0, outs1 = self.model(input), self.model(target)

        feats0, feats1 = {}, {}
        loss = 0.0 
        
        # Compare the feature maps at each layer
        for kk in range(self.model.N_slices):
            # normalize the feature maps, so that each token has unit norm
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk],dim=-1), normalize_tensor(outs1[kk],dim=-1)
            val_k  = (feats0[kk] - feats1[kk]).pow(2).sum(dim=-1) # mean over the channels, like in perceptual loss (which uses a pretrained 1x1 conv)
            loss += val_k.mean(dim=-1) # mean over the spatial dimensions

        return loss

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)


        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
def mae_load_weights_from_pth(model, checkpoint,ignore_keys=['decoder','mask_token'],strict=True):
    saved_checkpoint = torch.load(checkpoint, map_location='cpu',weights_only=True)
    try:
        checkpoint_model = saved_checkpoint['model']
    except KeyError:
        checkpoint_model = saved_checkpoint
    interpolate_pos_embed(model, checkpoint_model)  # interpolate position embedding to current size
    weights = {key: value for key, value in checkpoint_model.items() if not any(ik in key for ik in ignore_keys)}
    model.load_state_dict(weights, strict=strict)
    model = model
    model.eval()


def normalize_tensor(x,eps=1e-10,dim=1):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=dim,keepdim=True))
    return x/(norm_factor+eps)


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

