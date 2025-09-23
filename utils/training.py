import os
import re
import glob
import math
import sys
import torch
from torch.amp import  GradScaler
from utils.dist import MetricLogger, trigger_job_requeue, get_rank
import numpy as np


def restart_from_checkpoint(ckp_path, remove_module_subpath=False, run_variables=None, **kwargs):
    """
    Re-start from checkpoint saved as:
    {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
        "loss": loss_fn.state_dict()
    }
    """
    if not os.path.isfile(ckp_path):
        print(f"=> No checkpoint found at '{ckp_path}'")
        return

    print(f"=> Found checkpoint at '{ckp_path}'")

    checkpoint = torch.load(ckp_path, map_location="cpu")

    # Optionally remove 'module.' from keys in net state_dict
    if remove_module_subpath and "net" in checkpoint:
        net_params = {
            k.replace("module.", ""): v
            for k, v in checkpoint["net"].items()
        }
        checkpoint["net"] = net_params

    # Load items passed as kwargs
    # example: {'net': model, 'optimizer': opt}
    for key, obj in kwargs.items():
        if key in checkpoint and obj is not None:
            try:
                if key == "optimizer":
                    msg = init_optimizer(obj, checkpoint["net"], checkpoint[key])
                else:
                    msg = obj.load_state_dict(checkpoint[key], strict=False)
                print(f"=> loaded '{key}' from checkpoint with msg: {msg}")
            except TypeError:
                msg = obj.load_state_dict(checkpoint[key])
                print(f"=> loaded '{key}' from checkpoint (no strict option used)")
        else:
            print(f"=> key '{key}' not found in checkpoint")

    # Load run variables like epoch and args
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
                print(f"=> restored '{var_name}' from checkpoint")


def find_last_checkpoint(log_dir, experiment_name,return_params=False,return_last=False):
    # Path to the experiment's logs
    experiment_path = os.path.join(log_dir, experiment_name)

    if not os.path.exists(experiment_path):
        return None
    
    # List all versions in the experiment directory
    versions = [d for d in os.listdir(experiment_path) if d.startswith("version_")]
    if not versions:
        return None 
    # Sort versions by version number (last version is the most recent)
    versions.sort(key=lambda x: int(x.split("_")[-1]))

    for version in versions[::-1]:
        # Path to the checkpoints directory of the last version
        version_dir = os.path.join(experiment_path, version)
        checkpoints_dir = os.path.join(version_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            continue
        # Find all checkpoint files in the directory
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.ckpt"))
        if not checkpoint_files:
            continue
        else:
            print(f'Found checkpoint in version {version}')

            if return_last:
                last_checkpoint = os.path.join(checkpoints_dir, "last.ckpt")
            else:
                last_checkpoint = max(checkpoint_files, key=lambda x: int(re.search(r"epoch=(\d+)", x).group(1)))
            if return_params:
                hparams = os.path.join(version_dir, "hparams.yaml") 
                assert os.path.exists(hparams), f"Hyperparameters file not found at {hparams}"
                return last_checkpoint, hparams
            else:
                return last_checkpoint
    print(f"No checkpoints found.")
    return None

def init_optimizer(optimizer,model_params,checkpoint):
    """
    Synchronize optimizer with the checkpoint state.
    """
    if checkpoint is not None:
        current_state_dict = optimizer.state_dict()
        saved_state_dict = checkpoint

        # Check and add missing parameter groups
        current_groups = len(current_state_dict["param_groups"])
        saved_groups = len(saved_state_dict["param_groups"])

        param_id_to_tensor = {id(param): param for param in model_params}


        if current_groups < saved_groups:
            print(f"=> Adding {saved_groups - current_groups} parameter group(s) to the optimizer")
            for i in range(current_groups, saved_groups):
                new_group = saved_state_dict["param_groups"][i]
                
                new_group["params"] = [
                    param_id_to_tensor[param_id]
                    for param_id in new_group["params"]
                    if param_id in param_id_to_tensor]
                
                optimizer.add_param_group(new_group)

        # Load the saved optimizer state
        try:
            optimizer.load_state_dict(saved_state_dict)
            print("=> Optimizer state loaded and synchronized with checkpoint")
        except Exception as e:
            print(f"Failed to load optimizer state: {e}")
    else:
        print("No optimizer state found in checkpoint")
    return "Optimizer initialized successfully"


def train_one_epoch(model, data_loader, noise_scheduler,
                    loss_fn, optimizer,
                    epoch, args, use_amp=False):   # renamed arg to avoid shadowing

    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args['n_epochs'])

    model.train(True)
    scaler = GradScaler(enabled=use_amp)

    for it, (images, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, args['num_timesteps'] - 1, (images.shape[0],)).long()

        images = images.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        noise = noise.cuda(non_blocking=True)

        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        optimizer.zero_grad()

        # autocast only active if use_amp=True
        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(noisy_images, timesteps, label)
            if args['prediction_type'] == 'epsilon':
                loss = loss_fn(pred, noise)
            else:
                loss = loss_fn(pred, images)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args['DDP']:
            torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())

        if args['DDP']:
            if get_rank() == 0 and os.environ.get("SIGNAL_RECEIVED", "False") == "True":
                trigger_job_requeue()

    if args['DDP']:
        metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_one_epoch(model, data_loader, noise_scheduler,
                       loss_fn, epoch, args, use_amp=False):

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: [{}/{}]".format(epoch, args['n_epochs'])

    model.eval()

    for it, (images, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, args['num_timesteps'] - 1, (images.shape[0],)).long()

        images = images.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        noise = noise.cuda(non_blocking=True)

        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        # autocast only active if use_amp=True
        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(noisy_images, timesteps, label)
            if args['prediction_type'] == 'epsilon':
                loss = loss_fn(pred, noise)
            else:
                loss = loss_fn(pred, images)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping evaluation")
            sys.exit(1)

        if args['DDP']:
            torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())

    if args['DDP']:
        metric_logger.synchronize_between_processes()

    print("Evaluation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

