
from torch.optim.lr_scheduler import LambdaLR
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import os
from tqdm import tqdm

from .dpms import DDPM
from . import ddim_timesteps, extract_into_tensor, noise_like, default
from utils.utils import instantiate_from_config
from models.autoencoders.modules.distributions import DiagonalGaussianDistribution
from models.autoencoders.autoencoder import VQModelInterface

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 use_snr_scaling=True,
                 cond_stage_trainable=False,
                 concat_mode=True,
                 clip_denoised=False,
                 classifier_free_guidance=0.0, # probability of discarding conditioning during training
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.use_snr_scaling = use_snr_scaling
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = clip_denoised
        self.classifier_free_guidance = classifier_free_guidance

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def on_train_start(self):
        print("Using saved scaling factor:", self.scale_factor)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)  
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            self.scale_factor.copy_(1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule='scaled_linear', timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3,gamma=5.0):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        # betas in range [0,1] - amount of noise
        # alphas in range [1,0] - amount of signal 
        if self.use_snr_scaling:
            snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
            if self.parameterization == 'eps':
                snr = 1/(snr+1e-15)
            self.register_buffer('snr_weight', snr.clamp(max=gamma,min=0.1))
        else:
            self.register_buffer('snr_weight', torch.ones_like(self.alphas_cumprod))

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.sample()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

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

    @torch.no_grad()
    def get_input(self, batch, cfg=False, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch,self.first_stage_key)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None and not cfg:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}
        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        z = (1. / self.scale_factor) * z
        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        cfg = torch.rand(1).item() < self.classifier_free_guidance
        x, c = self.get_input(batch, cfg=cfg)
        loss = self(x, c,cfg=cfg)
        return loss

    def forward(self, x, c, cfg, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            if self.cond_stage_trainable and not cfg:
                assert c is not None
                c = self.get_learned_conditioning(c)
        return self.p_losses(x, c, t, cfg=cfg, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, cfg=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        output = self.model(x_noisy, t, **cond, cfg=cfg)
        return output

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def get_loss(self,pred,target,t):
        loss = super().get_loss(pred,target,mean=False)
        return (loss * extract_into_tensor(self.snr_weight,t,pred.shape)).mean([1, 2, 3])

    def p_losses(self, x_start, cond, t,cfg=False,noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t, cond,cfg=cfg) 

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target,t)
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t # no change if logvar is 0
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if self.original_elbo_weight != 0:
            loss_vlb = self.get_loss(model_output, target,t).mean(dim=(1, 2, 3))
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
            loss += (self.original_elbo_weight * loss_vlb)

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool,use_ema_model=False,guidance_scale=1.0):
        t_in = t
        model_out_uncond = None
        if use_ema_model:
            with self.ema_scope():
                model_out = self.apply_model(x, t_in, c)
                if self.is_conditional: # classifier free guidance
                    model_out_uncond = self.apply_model(x, t_in, c, cfg=True)
        else:
            model_out = self.apply_model(x, t_in, c)
            if self.is_conditional: # classifier free guidance
                model_out_uncond = self.apply_model(x, t_in, c, cfg=True)

        if self.is_conditional and guidance_scale != 1.0:
            model_out = self._classifier_free_guidance(model_out,model_out_uncond,weight=guidance_scale)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == 'v':
            x_recon = self.predict_start_from_v(x,t,model_out)
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
  
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    def _classifier_free_guidance(self,pred_cond,pred_uncond,weight=7.5,rescale=0.7,eps=1e-12):
        # from https://arxiv.org/abs/2305.08891
        cfg_pred = pred_uncond + weight * (pred_cond - pred_uncond)

        # per-sample std over non-batch dims
        dims = list(range(1, pred_cond.dim()))
        std_pos = pred_cond.std(dim=dims, keepdim=True)
        std_cfg = cfg_pred.std(dim=dims, keepdim=True).clamp_min(eps)

        # rescale (rho=rescale in the paper)
        factor = std_pos / std_cfg
        factor = rescale * factor + (1.0 - rescale)

        return cfg_pred * factor

    @torch.no_grad()
    def ddim_sample(self, cond,num_steps=150, sigma_t=0.0,use_ema_model=True,return_intermediates=False,guidance_scale=1.0):

        if sigma_t is None:
            sigma_t = 0.0

        model_out_uncond = None
        c = self.get_learned_conditioning(cond)
        batch_size = cond.shape[0]
        x = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=cond.device)

        alpha_bar_prev = self.alphas_cumprod_prev
        alpha_bar = self.alphas_cumprod

        intermediates = [x]

        log_every_t = int(self.log_every_t * num_steps / self.num_timesteps)

        timesteps = ddim_timesteps(num_steps, self.num_timesteps,method='quadratic')

        for t in tqdm(reversed(timesteps), desc='Sampling t', total=num_steps):

            t_vector = torch.full((batch_size,), t, device=cond.device, dtype=torch.long)

            if use_ema_model:
                with self.ema_scope():
                    model_out = self.apply_model(x, t_vector, c)
                    if self.is_conditional and guidance_scale != 1.0: # classifier free guidance
                        model_out_uncond = self.apply_model(x, t_vector, c, cfg=True)
            else:
                model_out = self.apply_model(x, t_vector, c)
                if self.is_conditional and guidance_scale != 1.0: # classifier free guidance
                    model_out_uncond = self.apply_model(x, t_vector, c, cfg=True)
            
            if self.is_conditional and guidance_scale != 1.0:
                model_out = self._classifier_free_guidance(model_out,model_out_uncond,weight=guidance_scale)
            
            if self.parameterization == "eps":
                eps = model_out
                x0 = self.predict_start_from_noise(x, t, eps)
            elif self.parameterization == "x0":
                x0 = model_out
                eps = self._predict_eps_from_xstart(x, t, x0)
            elif self.parameterization == 'v':
                v = model_out
                x0 = self.predict_start_from_v(x, t, v)
                eps = self._predict_eps_from_xstart(x, t, x0)
            else:
                raise ValueError(f"Unknown parameterization {self.parameterization}")

            if self.clip_denoised:
                x0 = x0.clamp(-1, 1)

            x0_scaled = alpha_bar_prev[t].sqrt() * x0
            dir_eps = (1-alpha_bar_prev[t]-sigma_t**2).sqrt() * eps 
            random_noise = sigma_t * torch.randn_like(x)
            x_prev = x0_scaled + dir_eps + random_noise
            x = x_prev

            if t %  log_every_t == 0 or t == num_steps - 1:
                intermediates.append(x.detach().cpu())

        latents = x
        decoded_images = self.decode_first_stage(latents)
        if return_intermediates:
            return decoded_images, intermediates
        else:
            return decoded_images

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised, repeat_noise=False, temperature=1.,
                  noise_dropout=0., use_ema_model=False,guidance_scale=1.0):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, c=c, t=t,
                                            clip_denoised=clip_denoised, use_ema_model=use_ema_model,guidance_scale=guidance_scale)
        
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None, use_ema_model=True, temperature=1.0,guidance_scale=1.0):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                use_ema_model=use_ema_model,temperature=temperature,guidance_scale=guidance_scale)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img
    
    @torch.no_grad()
    def sample(self, cond, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None,
               mask=None, x0=None, shape=None, temperature=1.0,guidance_scale=7.5):
        
        cond = self.get_learned_conditioning(cond)
        batch_size = cond.shape[0]

        if temperature is None:
            temperature = 1.0

        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        latents = self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, mask=mask, x0=x0,temperature=temperature,guidance_scale=guidance_scale)
        decoded_images = self.decode_first_stage(latents)
        if return_intermediates:
            return decoded_images, latents
        else:
            return decoded_images
    
    @torch.no_grad()
    def sample_intermediates(self,cond=None,shape=None,ddim=True):
        """
        Sample intermediates during the diffusion process.
        Returns:
            A list of tensors representing the sampled intermediates.
        Intermediate steps run from x_T to x_0.
        """
        if ddim:
            _, intermediates = self.ddim_sample(cond,num_steps=150, return_intermediates=True)
        else:
            cond = self.get_learned_conditioning(cond)
            if shape is None:
                shape = (1, self.channels, self.image_size, self.image_size)
            _, intermediates = self.p_sample_loop(cond, shape, return_intermediates=True)
            # Decode the latent code to image space

        intermediates = [self.decode_first_stage(latent.to(self.device)) for latent in intermediates]
        return torch.stack(intermediates, dim=0).squeeze(1)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
