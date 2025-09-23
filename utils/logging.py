#!/usr/bin/env python3
import os
import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import copy
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import numpy as np 
import torch.distributed as dist
from torchvision.utils import make_grid



class PlotSamplesDiffusion(Callback):

    def __init__(self,output_dir,name,device,num_steps=None,num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.name = name
        self.device = device
        self.num_steps = num_steps 
        self.num_samples = num_samples

        os.makedirs(os.path.join(self.output_dir, f'{self.name}/inference_samples'), exist_ok=True)


    def _plot_dist(self,images,title):
        if dist.is_initialized():
            print('rank', dist.get_rank())
            if dist.get_rank() == 0:
                self._plot(images, title)
            if dist.get_world_size() > 1:
                dist.barrier()
        else:
            self._plot(images, title)

    def _plot(self,images,title):
        """
        Plots a batch of images as a grid.

        Args:
            images (torch.Tensor): Tensor of shape (B, C, H, W).
            num_samples (int): Number of images to display (should be a perfect square for grid).
            save_path (str, optional): If provided, saves the plot to this path.
            title (str, optional): Title for the plot.
        """

        # Select first num_samples images
        images = images[:self.num_samples]
        row = int(np.sqrt(self.num_samples))
        grid = make_grid(images, nrow=row)

        # Convert to numpy
        npimg = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(8, 8))
        if title:
            plt.title(title)
        plt.imshow(npimg)
        plt.axis('off')

        save_dir = os.path.join(self.output_dir, f'{self.name}/inference_samples')
        path = os.path.join(save_dir, f'samples_{title}.pdf')
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(path, bbox_inches='tight')
        plt.close()
    
    def _generate_samples_autoencoder(self,trainer, pl_module):
        # Ensure the model is in evaluation mode
        pl_module.eval()

        # Get the validation dataloader
        # Deepcopy the val dataset
        val_dataloader = trainer.val_dataloaders
        dataset = copy.deepcopy(val_dataloader.dataset)
        
        # Now create a new DataLoader with the copied dataset
        val_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=val_dataloader.num_workers,
            pin_memory=True)

        # Get a batch of data
        data_iter = iter(val_dataloader)
        # Containers for model outputs
        x_list = []
        context_list = []
        xrec_list = []

        num_collected = 0

        with torch.no_grad():
            while num_collected < self.num_samples:
                # Get next batch
                inputs_i, context_i = next(data_iter)
                batch_size_i = inputs_i.size(0)

                # Move to device
                inputs_i = inputs_i.to(pl_module.device)
                context_i = context_i.to(pl_module.device)

                # If input has 5 dimensions (e.g., batch x frames x channels x height x width), flatten
                if inputs_i.ndim == 5:
                    x_i = inputs_i.reshape(-1, inputs_i.shape[-3], inputs_i.shape[-2], inputs_i.shape[-1])
                else:
                    x_i = inputs_i

                # Pass through model
                #xrec_out = pl_module.first_stage_model(x_i)
                _,xrec_out,_,_ = pl_module.first_stage_model(x_i)


                x_list.append(x_i)  # Append original images
                context_list.append(context_i)  # Append context
                xrec_list.append(xrec_out)

                num_collected += batch_size_i

            # Concatenate outputs and truncate to exactly num_samples
            x = torch.cat(x_list, dim=0)[:self.num_samples]
            xrec = torch.cat(xrec_list, dim=0)[:self.num_samples]

            x_both = torch.cat([x, xrec], dim=0)  # Concatenate real and reconstructed images
            xc = torch.cat(context_list, dim=0)[:self.num_samples]  # Concatenate contexts

            return x_both,xc

    def on_validation_epoch_end(self,trainer, pl_module):

        pl_module.eval()
        images_au, context_au = self._generate_samples_autoencoder(trainer, pl_module)
        images_real = images_au[:self.num_samples]

        if pl_module.is_conditional:
            images = pl_module.sample(context_au)
            xc_single = context_au[0].unsqueeze(0)  # Take the first context for unconditional sampling
            images_intermediates = pl_module.sample_intermediates(cond=xc_single, ddim=False)
        else:
            images = pl_module.sample()
            images_intermediates = pl_module.sample_intermediates(ddim=False)

        self._plot_dist(images, title=f'epoch_{trainer.current_epoch}')

        self._plot_dist(images_intermediates, title=f'epoch_{trainer.current_epoch}_intermediates')

        #images_intermediates_forward = pl_module.sample_intermediates_forward(x = images_real)
        #self._plot_dist(images_intermediates_forward, title=f'epoch_{trainer.current_epoch}_intermediates_forward')


class PlotSamplesCallback(Callback):
    def __init__(self, output_dir: str, num_samples: int = 10,name='',
                 on_train_batch_end=False,on_validation_epoch_end=True,mode='autoencoder'):
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.enable_on_train_batch_end = on_train_batch_end
        self.enable_on_validation_epoch_end = on_validation_epoch_end
        self.valid_last_iter_cache = None
        self.mode = mode
        self.savepath = os.path.join(output_dir, f'{name}/inference_samples')
        self.edge_keys = ['val/edge_loss/input_edges', 'val/edge_loss/recon_edges']
        os.makedirs(self.savepath, exist_ok=True)

    def _generate_samples_transformer(self, trainer, pl_module):
        pl_module.eval()

        val_dataloader = trainer.val_dataloaders
        dataset = copy.deepcopy(val_dataloader.dataset)

        batch_size = 10

        val_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_dataloader.num_workers,
            pin_memory=True)

        data_iter = iter(val_dataloader)

        x, x_c = [], []
        x_rec = []

        num_collected = 0

        desc = 'Generating samples'
        total = self.num_samples

        with torch.no_grad():
            pbar = tqdm(total=total, desc=desc)
            while num_collected < total:
                try:
                    x_i, y_i = next(data_iter)
                except StopIteration:
                    break

                x_i = x_i.to(pl_module.device)
                y_i = y_i.to(pl_module.device)

                if pl_module.conditional:
                    if pl_module.conditioning_type == 'edges':
                        c_i = pl_module.edge_filter(x_i)
                    else:
                        c_i = y_i
                else:
                    c_i = None

                if pl_module.first_stage_model.include_edges:
                    x_i_mask = pl_module.first_stage_model.edge_filter(x_i)
                    x_i = torch.cat([x_i_mask,x_i], dim=1)

                out = pl_module.sample(x_i, c=c_i)

                x.append(x_i)
                x_rec.append(out['x_sample'])

                if pl_module.conditional:
                    x_c.append(c_i)


                batch_size = x_i.size(0)
                num_collected += batch_size
                pbar.update(batch_size)

            pbar.close()

        x = torch.cat(x, 0)[:self.num_samples].to(pl_module.device)
        x_rec = torch.cat(x_rec, 0)[:self.num_samples].to(pl_module.device)

        out = {'x_rec':x_rec[:,:3,:,:]}
        return out

    def _generate_samples_autoencoder(self,trainer, pl_module):
        # Ensure the model is in evaluation mode
        pl_module.eval()

        # Get the validation dataloader
        # Deepcopy the val dataset
        val_dataloader = trainer.val_dataloaders
        dataset = copy.deepcopy(val_dataloader.dataset)
        
        # Now create a new DataLoader with the copied dataset
        val_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=val_dataloader.num_workers,
            pin_memory=True)

        num_samples = self.num_samples

        # Get a batch of data
        data_iter = iter(val_dataloader)
        # Containers for model outputs
        x_list = []
        xrec_list = []

        num_collected = 0

        with torch.no_grad():
            while num_collected < num_samples:
                # Get next batch
                inputs_i, _ = next(data_iter)
                batch_size_i = inputs_i.size(0)

                # Move to device
                inputs_i = inputs_i.to(pl_module.device)

                # If input has 5 dimensions (e.g., batch x frames x channels x height x width), flatten
                if inputs_i.ndim == 5:
                    x_i = inputs_i.reshape(-1, inputs_i.shape[-3], inputs_i.shape[-2], inputs_i.shape[-1])
                else:
                    x_i = inputs_i

                # Pass through model
                if hasattr(pl_module, "_edges"):
                    x_i = pl_module._edges(x_i)  # add edges if necessary
                x_out, xrec_out, _, _ = pl_module(x_i)
                

                x_out = x_out[:,:3,:,:]  # Keep only RGB channels
                xrec_out = xrec_out[:,:3,:,:]  # Keep only RGB channels

               
                # Append outputs
                x_list.append(x_out)
                xrec_list.append(xrec_out)

                num_collected += batch_size_i


        # Concatenate outputs and truncate to exactly num_samples
        x = torch.cat(x_list, dim=0)[:num_samples]
        xrec = torch.cat(xrec_list, dim=0)[:num_samples]
        out = {
            'x':x[:,:3,:,:],
            'x_rec': xrec[:,:3,:,:]
        }
        return out
        
    def _plot_dist(self, inputs, iteration_type, idx, callback_type, name='',titles=None):
        if dist.is_initialized() and dist.get_rank() != 0:
            return  # Skip saving on non-zero ranks            
        self._plot(inputs, iteration_type, idx, callback_type, name, titles)

    def _plot(self, inputs, iteration_type, idx, callback_type, name='',titles=None):
        """
        Plots a batch of images from multiple sources (tuple of images).
        
        Args:
            inputs (tuple of torch.Tensor): Tuple of images, each of shape (B, C, H, W).
            titles (list of str): List of titles for each column, must match length of `inputs`.
            iteration_type (str): Type of iteration (e.g., 'train' or 'val').
            idx (int): Index of the iteration.
            callback_type (str): Callback type.
            name (str, optional): Additional identifier for saving. Default is ''.
        """

        # Create output directory
        figpath = os.path.join(self.output_dir, f'{self.name}/inference_samples/{callback_type}')
        os.makedirs(figpath, exist_ok=True)

        if isinstance(inputs, dict):
            titles = list(inputs.keys())
            inputs = list(inputs.values())
        
        if titles is None:
            titles = [f"Image {i+1}" for i in range(len(inputs))]
        assert len(inputs) == len(titles), "Length of inputs and titles must match"


        # Determine batch size (assume all inputs have the same batch size)
        batch_size = inputs[0].shape[0]
        num_columns = len(inputs)
                
        # Set figure size
        fig_width = num_columns * 3  # Width scales with number of images in tuple
        fig_height = batch_size * 3  # Height scales with batch size

        # Create figure with a grid of (B x num_columns)
        fig, axes = plt.subplots(batch_size, num_columns, figsize=(fig_width, fig_height))

        # Ensure axes is always a 2D array
        if batch_size == 1:
            axes = [axes]
        if num_columns == 1:
            axes = [[ax] for ax in axes]

        for row in range(batch_size):  # Iterate over batch
            for col, (img_batch, title) in enumerate(zip(inputs, titles)):  # Iterate over different image types
                img = img_batch[row]  # Select image from batch (C, H, W)
                
                # Determine colormap
                cmap = 'gray' if img.shape[0] == 1 else None

                # Prepare image for plotting
                img = img.permute(1, 2, 0)  # Change to (H, W, C)
                img = img.cpu().numpy()

                # Plot image
                axes[row][col].imshow(img, cmap=cmap)
                if row == 0:  # Add title only to the first row
                    axes[row][col].set_title(title)
                axes[row][col].axis("off")

        # Save figure
        plt.tight_layout()
        filename = f"{iteration_type}_{idx}_{name}.pdf" if name else f"{iteration_type}_{idx}.pdf"
        plt.savefig(os.path.join(figpath, filename))
        plt.close(fig)

        print(f"Saved validation samples for {iteration_type} {idx} to {figpath}")


    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enable_on_validation_epoch_end:
            return

        # Generate samples based on mode
        if self.mode == 'autoencoder':
            out = self._generate_samples_autoencoder(trainer, pl_module)
        elif self.mode == 'transformer':
            out = self._generate_samples_transformer(trainer, pl_module)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'autoencoder' or 'transformer'.")
        
        epoch = trainer.current_epoch

        # If no samples were generated, exit early
        if out is None:
            return

        # Validate output format
        if not isinstance(out, dict) or len(out) not in [1, 2]:
            raise ValueError(f"Expected output to be a dict with 1 or 2 items, but got {type(out)} with len={len(out)}")

        # Determine titles
        titles = []

        if self.mode == 'autoencoder':
            titles = ['Real', 'Synthetic']
        else:  # transformer
            titles = ['Synthetic']

        # Plot the generated images
        self._plot_dist(out, iteration_type='epoch', idx=epoch,
                        callback_type='on_validation_epoch_end', titles=titles)

        # Optionally plot cached edge reconstructions
        if self.valid_last_iter_cache is not None:
            self._plot_dist(self.valid_last_iter_cache, iteration_type='epoch', idx=epoch,
                            callback_type='on_validation_epoch_end', name='edges')
            self.valid_last_iter_cache = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enable_on_train_batch_end:
            return
        if batch_idx % 100 != 0:
            return
        inputs, _ = batch
        reconstructions = outputs['reconstructions']
        out = (inputs,reconstructions)
        self._plot_dist(out,iteration_type='batch',idx=batch_idx,callback_type='on_train_batch_end')
        if 'train/edge_loss/input_edges' in outputs and 'train/edge_loss/recon_edges' in outputs:
            batch_pair = (outputs['train/edge_loss/input_edges'],outputs['train/edge_loss/recon_edges'])
            self._plot_dist(batch_pair,iteration_type='batch',idx=batch_idx,callback_type='on_train_batch_end',name='edges')
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None or not isinstance(outputs, dict) or not all(key in outputs for key in self.edge_keys):
            return
        batch_pair = (outputs[self.edge_keys[0]],outputs[self.edge_keys[1]])
        self.valid_last_iter_cache = batch_pair


from models.diffusion_models.simple_diffusion import ClassConditionedUnet, sample_random_conditioning

def generate_samples(args, device, net, noise_scheduler,modelname='', epoch=-1,is_conditional=False,num_samples=10):
    image_size = args['image_size']
    class_embed_size = args['class_embed_size']

    ncol = 2
    nrow = num_samples // ncol
    figsize = (ncol * 5, nrow * 5)  # Make each image ~5 inches wide

    # Sample noise and conditioning
    x = torch.randn(num_samples, 3, image_size, image_size).to(device)
    if is_conditional:
        cond = sample_random_conditioning(num_samples=num_samples, device=device)
    else:
        cond = None 

    # Load model if needed
    if net is None:
        modelname = modelname or 'checkpoint.pth'
        model_path = os.path.join(args['output_dir'], modelname)

        state_dict = torch.load(model_path, map_location='cpu')['net']
        net = ClassConditionedUnet(class_embed_size=class_embed_size, image_size=image_size).to(device)
        net.load_state_dict(state_dict, strict=False)
    net.eval()

    # Sampling loop
    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = net(x, t, cond)
        x = noise_scheduler.step(residual, t, x).prev_sample
        #x = noise_scheduler.step(residual,t,x)

    # Plot the generated images
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.flatten()

    x = x.detach().cpu().clip(0, 1)

    for i, ax in enumerate(axes):
        if i < num_samples:
            img = x[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis("off")

    fig.suptitle(f'Samples at epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save the figure
    figpath = os.path.join(args['output_dir'], 'inference_samples_{modelname}'.format(modelname=modelname))
    os.makedirs(figpath, exist_ok=True)
    fig.savefig(os.path.join(figpath, f'samples_at_epoch_{epoch}.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    