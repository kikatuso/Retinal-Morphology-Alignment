import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from utils.utils import instantiate_from_config

from .modules.blocks import Encoder, Decoder
from .modules.distributions import DiagonalGaussianDistribution
from .modules.quantize import VectorQuantizer2


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 loss,
                 n_embed,
                 embed_dim,
                 optim_lr=4.5e-6,
                 include_edges = False,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 batch_resize_range=None,
                 edge_detector_config=False,
                 quantize=False,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.learning_rate = optim_lr
        self.image_key = image_key
        self.include_edges = include_edges
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        if edge_detector_config is not False:
            self.edge_filter = instantiate_from_config(edge_detector_config)
        else:
            self.include_edges = False
        self.loss = instantiate_from_config(loss)
        if quantize is not False:
            self.quantize = instantiate_from_config(quantize)
        else:
            self.quantize =  VectorQuantizer2(n_embed, embed_dim, beta=0.25,legacy=True)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu",weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False,)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

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

    def _edges(self,x):
        if self.include_edges:
            input_edges = self.edge_filter(x)
            x = torch.cat([x, input_edges], dim=1)
        return x

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, (perplexity,_,ind) = self.encode(input)
        dec = self.decode(quant)
        return input,dec, diff,perplexity
    
    def training_step(self, batch, batch_idx):
        x,_ = batch

        x = self._edges(x)

        x, xrec, qloss,quant_perp = self(x)

        opt_ae, opt_disc = self.optimizers() # type: ignore

        # Step 1: Autoencoder Optimization
        opt_ae.zero_grad()
        aeloss, log_dict_ae,log_loss_data = self.loss(qloss, x,xrec, 0, self.current_epoch,self.global_step,last_layer=self.get_last_layer(),
                                         split="train")
        self.manual_backward(aeloss)  # Manually backpropagate loss
        opt_ae.step()

        # Log autoencoder metrics
        self.log("train/Autoencoder Loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log('train/Perplexity',quant_perp, prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True,sync_dist=True)

        # Step 2: Discriminator Optimization
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(qloss, x,xrec, 1, self.current_epoch,self.global_step, last_layer=self.get_last_layer(),split="train")
        self.manual_backward(discloss)
        opt_disc.step()

        # Log discriminator metrics
        self.log("train/Discriminator Loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        out = {'reconstructions': xrec}
        for k,i in log_loss_data.items():
            out[k] = i
        return out

    def validation_step(self,batch,batch_idx):
        x,_ = batch

        x = self._edges(x)

        x, xrec, qloss,quant_perp = self(x)

        aeloss, log_dict_ae,log_loss_data = self.loss(qloss, x, xrec, 0, self.current_epoch,self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.current_epoch,self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val")

        self.log("val/Autencoder Loss", aeloss,prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log('val/Perplexity',quant_perp,prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log_dict(log_dict_ae,sync_dist=True)
        self.log_dict(log_dict_disc,sync_dist=True)
        out = {}
        for k,i in log_loss_data.items():
            out[k] = i
        return out

    def configure_optimizers(self):
        lr = self.learning_rate

        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    def get_last_encoder_layer(self):
        return self.encoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 loss,
                 embed_dim,
                 learning_rate=4.5e-6,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(loss)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
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

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu",weights_only=True)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def _edges(self,x):
        return x

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, inputs, sample_posterior=True):
        posterior = self.encode(inputs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return inputs,dec, posterior,None

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        inputs,_ = batch
        inputs,reconstructions, posterior,_ = self(inputs)

        opt_ae, opt_disc = self.optimizers()  # type: ignore

        # ---------------------------
        # Autoencoder Optimization
        # ---------------------------
        opt_ae.zero_grad()
        aeloss, log_dict_ae,log_loss_data = self.loss(posterior,inputs, reconstructions,0, self.current_epoch,
            self.global_step,last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False,sync_dist=True)


        # ---------------------------
        # Discriminator Optimization
        # ---------------------------
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(posterior,inputs, reconstructions,1, self.current_epoch,
            self.global_step,last_layer=self.get_last_layer(), split="train"
        )
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False,sync_dist=True)

        return aeloss


    def validation_step(self, batch, batch_idx):
        inputs,_ = batch
        inputs,reconstructions, posterior,_ = self(inputs)
        aeloss, log_dict_ae,log_loss_data = self.loss(posterior,inputs, reconstructions, 0, self.current_epoch,
                                        self.global_step,last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(posterior,inputs, reconstructions, 1, self.current_epoch,
                            self.global_step,last_layer=self.get_last_layer(), split="val")

        self.log_dict(log_dict_ae,sync_dist=True)
        self.log_dict(log_dict_disc,sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

