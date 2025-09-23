# source: https://github.com/CompVis/taming-transformers/blob/master/taming/models/cond_transformer.py

import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import sys 
from utils.utils import instantiate_from_config
from .modules.utils import SOSProvider, top_k_top_p_filtering
from einops import repeat

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 base_learning_rate,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 edge_detector_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 conditioning_weight = 1.0,
                 conditioning_type='image',
                 cond_stage_trainable=False,
                 downsample_cond_size=-1,
                 conditional_learning_rate=1.e-8,
                 pkeep=1.0, # how many tokens to keep for partial noising - 1.0 means no noising
                 sos_token=0,
                 conditional=False,
                 ):
        super().__init__()
        self.sos_token = sos_token
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_weight = conditioning_weight
        self.automatic_optimization = True
        assert conditioning_type in ['image', 'label','edges',None], "Conditioning type must be either 'image' or 'label', 'edges' or None."
        self.conditioning_type = conditioning_type
        self.learning_rate = base_learning_rate
        self.conditional = conditional
        
        if self.conditional and self.cond_stage_trainable:
            assert conditional_learning_rate is not None, "For conditional models, a separate conditional learning rate must be provided."
            self.cond_learning_rate = conditional_learning_rate
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if edge_detector_config is not None:
            self.init_edge_detector_from_ckpt(edge_detector_config)
        if permuter_config is None:
            permuter_config = {"target": "models.stable_diffusion.modules.mintgpt.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
    
    def save_configs(self,config_path, output_dir, experiment_name):
        """
        Save the configuration file to the specified output directory and experiment name.
        If the configuration file already exists in the target directory, it will not be copied.
        """
        output_path = os.path.join(output_dir, experiment_name)
        target_file = os.path.join(output_path, os.path.basename(config_path))
        
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print(f"Created directory: {output_path}")
        
        if not os.path.isfile(target_file):
            print(f"Saving config to {target_file}")
            os.system(f"cp {config_path} {target_file}")
        else:
            print(f"Config file already exists: {target_file}, skipping copy.")

    def init_edge_detector_from_ckpt(self,config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.edge_filter = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or not self.conditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.conditional = False
            self.conditioning_type = None
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            if not self.cond_stage_trainable:
                print("Setting cond stage model to eval mode.")
                model = model.eval()
                model.train = disabled_train
            self.cond_stage_model = model
        
    def _get_batch_w_edges(self, batch):
        x, _ = batch
        c = self.edge_filter(x)
        if self.cond_stage_model.include_edges:
            if c.shape[1] != 4:
                c = c.repeat(1, 4, 1, 1)
        else:
            if c.shape[1] != 3:
                c = c.repeat(1, 3, 1, 1)
        return x, c
    
    def _get_batch_w_label(self, batch):
        x, y = batch
        return x, y 

    def _get_batch(self, batch):
        if self.conditioning_type == 'edges':
            return self._get_batch_w_edges(batch)
        elif self.conditioning_type == 'label' or self.conditioning_type == 'image':
            return self._get_batch_w_label(batch)
        elif self.conditioning_type is None:
            x, _ = batch
            return x,None 
        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning_type}")
        
    def masking(self,z_indices):
        mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                        device=z_indices.device))
        mask = mask.round().to(dtype=torch.int64)
        r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
        a_indices = mask*z_indices+(1-mask)*r_indices
        return a_indices
    
    def loss(self, logits, target):
        """
        Computes the cross-entropy loss between the logits and the target.
        Args:
            logits: The output logits from the transformer model.
            target: The target indices for the loss computation.
        Returns:
            The computed loss value.
        """
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        
    def forward(self, x, c=None):
        # one step to produce the logits
        _,z_indices = self.encode_to_z(x) # B x (H*W = 1024)
        if self.training and self.pkeep < 1.0:
            r_indices = self.masking(z_indices)
        else:
            r_indices = z_indices

        if self.conditional:
            _,c_loss,c_indices = self.encode_to_c(c) # c_token:  B x 1 
        else:
            c_indices = repeat(torch.tensor([self.sos_token]), '1 -> b 1', b=x.shape[0]).to(x.device)  # sos token
            c_loss = None 

        cr_indices = torch.cat((c_indices, r_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices # predict entire image sequence

        # make the prediction
        logits, _ = self.transformer(cr_indices[:, :-1]) # predicting with all but last token as input
    
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        k = c_indices.shape[1]-1
        logits = logits[:,k:,:]   # keep all except first k condtioning tokens
        return logits, target, c_loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def sample(self,x,c=None,sample=True,temperature=1.0):

        B = x.shape[0]
        z_code, z_indices =  self.encode_to_z(x)
        if self.conditional:
            _,_,c_indices = self.encode_to_c(c)
        num_patches = z_indices.shape[1]

        if temperature is None:
            temperature = 1.0

        x_start = repeat(torch.tensor([self.sos_token]), '1 -> b 1', b=B).to(x.device)  # sos token

        #x_start = z_indices[:, :1]  # start from first token

        sampled_indices = self.sample_with_past(
            x = c_indices if self.conditional else x_start,
            steps=num_patches,
            temperature=temperature,
            sample_logits=sample,
            top_k=None,
            top_p=None,
            callback=None
        )

        # sampled_indices is of shape (B, num_patches)
        # we need to decode it to an image
        x_sample = self.decode_to_img(sampled_indices, z_code.shape)

        out = {
                "x_sample": x_sample
            }
        if self.conditional:
            out["c"] = c

        return out 


    @torch.no_grad()
    def sample_next_token(self, x,steps,c=None,temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        if self.conditional:
            x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size # make sure model can see conditioning
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            logits, _ = self.transformer(x_cond)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        if self.conditional:
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x 
    
    @torch.no_grad()
    def sample_with_past(self,x, steps, temperature=1., sample_logits=True,
                        top_k=None, top_p=None, callback=None):
        # x is conditioning
        sample = x
        cond_len = x.shape[1]
        past = None
        for n in range(steps):
            if callback is not None:
                callback(n)
            logits, _, present = self.transformer.forward_with_past(x, past=past, past_length=(n+cond_len-1))
            if past is None:
                past = [present]
            else:
                past.append(present)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            if not sample_logits:
                _, x = torch.topk(probs, k=1, dim=-1)
            else:
                x = torch.multinomial(probs, num_samples=1)
            # append to the sequence and continue
            sample = torch.cat((sample, x), dim=1)
        del past
        sample = sample[:, cond_len:]  # cut conditioning off
        return sample

    ### note this is broken
    @torch.no_grad()
    def sample_patches(self, x, c, temperature=1.0, top_k=None):
            """
            Performs patch-wise sampling for a transformer model, generating an image progressively 
            by iterating over patches of size `patch_size x patch_size`.

            Args:
                x: Input image tensor of shape (B, 3, H, W).
                c: Conditioning image tensor of shape (B, 3, H, W).
                temperature: Scaling factor for logits (higher = more random).
                top_k: If specified, restricts sampling to top k logits.
                update_every: Number of steps before updating the visualized image.

            Returns:
                Generated image tensor of shape (B, 3, H, W).
            """
            # Encode conditioning image

            if self.conditional:
                if self.cond_stage_model.include_edges:
                    if c.shape[1] != 4:
                        c = c.repeat(1, 4, 1, 1)
                else:
                    if c.shape[1] != 3:
                        c = c.repeat(1, 3, 1, 1)
                if self.multiplication_factor != 1.0:
                    x = torch.nn.functional.interpolate(x, scale_factor=self.multiplication_factor, mode="bicubic")
                    c = torch.nn.functional.interpolate(c, scale_factor=self.multiplication_factor, mode="bicubic")
                z_code, z_indices =  self.encode_to_z(x)
                c_code, c_indices = self.encode_to_c(c)
            else:
                if self.multiplication_factor != 1.0:
                    x = torch.nn.functional.interpolate(x, scale_factor=self.multiplication_factor, mode="bicubic")
                        
                z_code, z_indices =  self.encode_to_z(x)


            z_code_shape = z_code.shape
            B, C, H, W = z_code.shape

            print(f"z_code shape: {z_code_shape}, z_indices shape: {z_indices.shape}")

            assert z_code.shape[2]*z_code.shape[3] == z_indices.shape[1]

            x_rec = self.first_stage_model.decode(z_code)
            if self.conditional:
                x_mask_rec = self.first_stage_model.decode(c_code)

            # Extract shape details
            patch_size = z_code_shape[-1] // self.multiplication_factor # Patch size
            
            # Reshape indices to match spatial dimensions
            z_indices_rand = torch.randint(1,size=z_indices.shape).to(z_code.device)
            idx = z_indices_rand.reshape(B,z_code.shape[2],z_code.shape[3]) # Dim [B, H, W]
            if self.conditional:
                cidx = c_indices.reshape(B,z_code_shape[2],z_code_shape[3])

            for i in range(z_code.shape[2]): # Iterate over height indices e.g. (1 to 32)
                if i <= patch_size//2:
                    local_i = i
                elif z_code.shape[2]-i < patch_size//2:
                    local_i = patch_size-(z_code.shape[2]-i)
                else:
                    local_i = patch_size//2
                for j in range(z_code.shape[3]): # Iterate over width indices e.g. (1 to 32)
                    if j <= patch_size//2:
                        local_j = j
                    elif z_code.shape[3]-j < patch_size//2:
                        local_j = patch_size-(z_code.shape[3]-j)
                    else:
                        local_j = patch_size//2

                    # Define patch boundaries
                    i_start, i_end = i - local_i, i - local_i + patch_size
                    j_start, j_end = j - local_j, j - local_j + patch_size

                    # Extract patches
                    patch = idx[:, i_start:i_end, j_start:j_end].reshape(B, -1)

                    if self.conditional:
                        cpatch = cidx[:, i_start:i_end, j_start:j_end].reshape(B, -1)
                        # Concatenate conditioning
                        patch = torch.cat((cpatch, patch), dim=1)

                    # Generate logits
                    logits, _ = self.transformer(patch[:,:-1])  # Exclude last token
                    last_token_logits = logits[:, -1, :]  # Get logits for the last token dim [B, patch_size**2, vocab_size]
                    # Reshape logits to match patch size
                    last_token = last_token_logits.reshape(B, patch_size, patch_size, -1)

                    # Apply temperature scaling
                    last_token= last_token / temperature

                    # Apply top-k filtering
                    if top_k is not None:
                        last_token = self.top_k_logits(last_token, top_k)

                    # Convert to probabilities and sample
                    probs = torch.nn.functional.softmax(last_token, dim=-1)
                    idx[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    # Update visualization periodically
                    step = i * z_code.shape[3] + j
                    if step == z_code.shape[2] * z_code.shape[3] - 1:
                        x_sample = self.decode_to_img(idx,z_code_shape)  # Generate batch of images
                        break

            x = x[:,:3,:,:]
            x_rec = x_rec[:,:3,:,:]
            x_sample = x_sample[:,:3,:,:]

            out = {
                "x": x,
                "c": c,
                "x_rec": x_rec,
                "x_sample": x_sample
            }

            if self.conditional:
                c = c[:,0,:,:].unsqueeze(1)
                out["c"] = c
            if self.first_stage_model.include_edges:
                x_mask_rec = x_mask_rec[:,-1,:,:].unsqueeze(1)
                x_sample_mask = x_sample[:,-1,:,:].unsqueeze(1)
                out["x_mask_rec"] = x_mask_rec
                out["x_sample_mask"] = x_sample_mask
            
            return out

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.conditioning_type =='image':
            if self.downsample_cond_size > -1:
                c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
            quant_c,c_loss, [_,_,indices] = self.cond_stage_model.encode(c)
            #if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        else:
            quant_c, c_loss, indices = self.cond_stage_model.forward(c) # assumes BxM shape where M is number of features
        if indices.dim() == 1:
            indices = indices.unsqueeze(-1) # make sure Bx1
        return quant_c,c_loss,indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    def training_step(self, batch, batch_idx):
        x,c = self._get_batch(batch)
        if self.first_stage_model.include_edges:
            x_edges = self.edge_filter(x)
            x = torch.cat([x,x_edges],dim=1)
        logits, target, closs = self(x, c)
        loss = self.loss(logits, target)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.conditional and self.cond_stage_trainable:
            assert closs is not None
            loss = loss + self.cond_weight * closs
            self.log("train/closs", closs, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,c = self._get_batch(batch)
        if self.first_stage_model.include_edges:
            x_edges = self.edge_filter(x)
            x = torch.cat([x,x_edges],dim=1)
        logits, target, closs = self(x, c)
        loss = self.loss(logits, target)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.conditional and self.cond_stage_trainable:
            assert closs is not None
            loss = loss + self.cond_weight * closs
            self.log("val/closs", closs, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01,'lr': self.learning_rate},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0,'lr': self.learning_rate},
        ]

        if self.cond_stage_trainable:
            optim_groups.append({"params": list(self.cond_stage_model.parameters()), "lr": self.cond_learning_rate})

        optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95))
        return optimizer