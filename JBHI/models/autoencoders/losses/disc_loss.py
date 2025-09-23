import torch 
import torch.nn as nn
import torch.nn.functional as F 
from .discriminator import weights_init, NLayerDiscriminator
from utils.utils import instantiate_from_config


def adopt_weight(weight,current_val, threshold=0, value=0.):
    if current_val < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 percept_loss = None,
                 edge_loss = None,
                 disc_start_iter=None,
                 disc_start_epoch=None,
                 codebook_term=True,
                 codebook_weight=1.0,
                 kll_weight=0.000001, # The weight of the codebook loss in the overall loss function.
                 pixelloss_weight=1.0,
                 disc_num_layers=3, 
                 disc_in_channels=3,
                 disc_factor=1.0, 
                 disc_weight=1.0,
                 disc_conditional=False,
                 disc_ndf=64,
                 disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        
        self.codebook_term = codebook_term

        self.codebook_weight = codebook_weight if codebook_term else kll_weight
        self.pixel_weight = pixelloss_weight

        ## Edge loss
        if edge_loss is not None:
            self.edge_loss = instantiate_from_config(edge_loss)
            self.edge_weight = getattr(self.edge_loss, "weight", 1.0)
            self.edge_start_epoch = getattr(self.edge_loss, "start_epoch", None)
            self.edge_start_iter =  getattr(self.edge_loss, "start_iter", None)
            self.edge_factor =  getattr(self.edge_loss, "factor", 1.0)
            assert self.edge_start_iter is not None or self.edge_start_epoch is not None, "Edge start be defined."
            print(f'Edge loss started with {self.edge_weight} weight, {self.edge_factor} factor, {self.edge_start_epoch} start_epoch, {self.edge_start_iter} start_iter')
        else:
            self.edge_loss = None

        ## RetFound/Perceptual loss
        if percept_loss is not None:
            self.percept_loss = instantiate_from_config(percept_loss)
            self.percept_weight = getattr(self.percept_loss, "weight", 1.0)
            self.percept_start_epoch = getattr(self.percept_loss, "start_epoch", None)
            self.percept_start_iter = getattr(self.percept_loss, "start_iter", None)
            self.percept_factor = getattr(self.percept_loss, "factor", 1.0)
            assert self.percept_start_iter is not None or self.percept_start_epoch is not None, "Perceptual loss start be defined."
            print(f'RetFound/Perceptual loss started with {self.percept_weight} weight, {self.percept_factor} factor, {self.percept_start_epoch} start_epoch, {self.percept_start_iter} start_iter')
        else:
            self.percept_loss = None



        ## Discriminator

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        
        self.discriminator_start_epoch = disc_start_epoch if disc_start_epoch is None else int(disc_start_epoch)
        self.discriminator_start_iter = disc_start_iter if disc_start_iter is None else int(disc_start_iter)
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
    
    def calculate_adaptive_weight(self, nll_loss, g_loss,weight, last_layer=None):
        """
        Calculate adaptive weight for the g_loss based on the gradients of the nll_loss and g_loss.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * weight
        return d_weight

    def check_for_nan(self,loss_dict):
        for loss_name,value in loss_dict.items():
            if torch.isnan( value.detach() ).any():
                raise RuntimeError(f"NaN detected in {loss_name}")
        
    def forward(self, q_loss, inputs,reconstructions, optimizer_idx,
                current_epoch,global_iter, last_layer=None, cond=None, split="train"):

        ## Autoencoder part = pixel loss + codebook loss
        
        nll_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        if self.codebook_term: # for VQ-GAN
            quantizer_term = self.codebook_weight * q_loss.mean()
        else: # for VAE with KL divergence term
            kl_loss = q_loss.kl()
            quantizer_term = self.codebook_weight*(torch.sum(kl_loss) / kl_loss.shape[0])

        loss = nll_loss + quantizer_term

        log = {}
        log_data = {} ## DATA TO BE USED IN LOGGING

        log["{}/autencoder/reconstruction_loss".format(split)] = nll_loss.detach().mean()
        log["{}/autoencoder/codebook_loss".format(split)] = quantizer_term.detach()

        ############## any additional losses here ##############

        if optimizer_idx == 0:

            ## RetFound/Perceptual loss

            if self.percept_loss is not None:
                if self.percept_start_epoch:
                    percept_factor = adopt_weight(self.percept_factor,current_epoch, threshold=self.percept_start_epoch)
                else:
                    percept_factor = adopt_weight(self.percept_factor,global_iter, threshold=self.percept_start_iter)
                if percept_factor != 0:
                    percept_loss = self.percept_loss(inputs.contiguous(),reconstructions.contiguous()).mean()
                    try:
                        percept_weight = self.calculate_adaptive_weight(nll_loss,percept_loss,weight=self.percept_weight,last_layer=last_layer)
                    except RuntimeError:
                        assert not self.training
                        percept_weight = torch.tensor(0.0,device=inputs.device)
                    loss = loss + percept_loss * percept_factor * percept_weight
                else:
                    percept_weight = torch.tensor(0.0,device=inputs.device)
                    percept_loss = torch.tensor(0.0,device=inputs.device)
                log["{}/autoencoder/percept_loss".format(split)] = percept_loss.detach().mean()
                log["{}/autoencoder/percept_weight".format(split)] = percept_weight.detach()
                log["{}/autoencoder/percept_factor".format(split)] = torch.tensor(percept_factor,device=inputs.device)

            ## Edge loss
            if self.edge_loss is not None:
                
                if self.edge_start_epoch:
                    e_factor = adopt_weight(self.edge_factor,current_epoch, threshold=self.edge_start_epoch)
                else:
                    e_factor = adopt_weight(self.edge_factor,global_iter, threshold=self.edge_start_iter)

                if e_factor != 0:
                    input_edges,recon_edges,e_loss = self.edge_loss(inputs,reconstructions)

                    log_data['{}/edge_loss/input_edges'.format(split)] = input_edges.detach()
                    log_data['{}/edge_loss/recon_edges'.format(split)] = recon_edges.detach()

                    try:
                        e_weight = self.calculate_adaptive_weight(nll_loss,e_loss,weight=self.edge_weight,last_layer=last_layer)
                    except RuntimeError:
                        assert not self.training
                        e_weight = torch.tensor(0.0,device=inputs.device)
                    
                    loss = loss + e_weight * e_factor * e_loss
                else:
                    e_weight = torch.tensor(0.0,device=inputs.device)
                    e_loss = torch.tensor(0.0,device=inputs.device)
                
                log['{}/autoencoder/edge_loss'.format(split)] = e_loss.detach()
                log['{}/autoencoder/edge_weight'.format(split)] = e_weight.detach()
                log['{}/autoencoder/edge_factor'.format(split)] = torch.tensor(e_factor,device=inputs.device)

        # now the GAN part
        if optimizer_idx == 0:

            if self.discriminator_start_epoch:
                disc_factor = adopt_weight(self.disc_factor,current_epoch, threshold=self.discriminator_start_epoch)
            else:
                disc_factor = adopt_weight(self.disc_factor,global_iter, threshold=self.discriminator_start_iter)

            if disc_factor != 0:
                # generator update
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(reconstructions[:,:3,:,:].contiguous())
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(torch.cat((reconstructions[:,:3,:,:].contiguous(), cond), dim=1))
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss,weight=self.discriminator_weight, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0,device=inputs.device)
                
                loss = loss + disc_factor * d_weight * g_loss
            else:
                d_weight = torch.tensor(0.0,device=inputs.device)
                g_loss = torch.tensor(0.0,device=inputs.device)
            
            log['{}/autoencoder/generator_loss'.format(split)] = g_loss.detach().mean()
            log['{}/autoencoder/GAN_weight'.format(split)] = d_weight.detach()
            log['{}/autoencoder/disc_factor'.format(split)] = torch.tensor(disc_factor,device=inputs.device)


            return loss, log, log_data

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs[:,:3,:,:].contiguous().detach())
                logits_fake = self.discriminator(reconstructions[:,:3,:,:].contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs[:,:3,:,:].contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions[:,:3,:,:].contiguous().detach(), cond), dim=1))

            if self.discriminator_start_epoch:
                disc_factor = adopt_weight(self.disc_factor,current_epoch, threshold=self.discriminator_start_epoch)
            else:
                disc_factor = adopt_weight(self.disc_factor,global_iter, threshold=self.discriminator_start_iter)
            
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            with torch.no_grad():
                real_acc = (logits_real > 0).float().mean()   # fraction of real samples correctly classified
                fake_acc = (logits_fake < 0).float().mean()   # fraction of fake samples correctly classified

            log = {"{}/discriminator/logits_real".format(split): logits_real.detach().mean(),
                   "{}/discriminator/logits_fake".format(split): logits_fake.detach().mean(),
                     "{}/discriminator/real_accuracy".format(split): real_acc.detach(),
                     "{}/discriminator/fake_accuracy".format(split): fake_acc.detach(),
                   }
            return d_loss, log