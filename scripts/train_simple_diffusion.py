#!/usr/bin/env python3
import sys 
import os 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch import nn
import datetime
import time
import json
from pathlib import Path 


sys.path.append(os.getcwd()) 

from models.diffusion_models.simple_diffusion import ClassConditionedUnet

from utils.logging import generate_samples
from diffusers import DDPMScheduler
from utils.dataset import GenericDatasetClass
from utils.dist import init_DDP, is_main_process, save_on_master
from utils.training import train_one_epoch, evaluate_one_epoch, restart_from_checkpoint



def main(args,device):

    torch.set_float32_matmul_precision('medium') 

    output_dir = args['output_dir']
    n_epochs = args['n_epochs']
    image_size = args['image_size']
    num_timesteps = args['num_timesteps']
    DDP = args['DDP']
    class_embed_size = args['class_embed_size']
    modelname = args['modelname']
    batch_size = args['batch_size']
    conditional = args['conditional']
    block_out_channels = args['block_out_channels']
    layers_per_block = args['layers_per_block'] 
    down_block_types =  args['down_block_types']
    up_block_types =  args['up_block_types']
    autocast = args['autocast']
    prediction_type = args['prediction_type']
    transform = args['transform']
    val_transform = args['val_transform']

    # save args
    save_path = Path(output_dir) / "config.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(args, f, indent=4)


    best_val_loss = float("inf")
    patience_counter = 0

    if DDP:
        init_DDP(args)

    # Create a scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_timesteps,
        beta_start=0.002,
        beta_end=0.02,
        beta_schedule="scaled_linear",
        prediction_type=prediction_type,
    )

    data = GenericDatasetClass(transform=transform,val_transform=val_transform)
    train_loader, test_loader = data.get_dataloaders(batch_size=batch_size)


    net = ClassConditionedUnet(class_embed_size=class_embed_size,image_size=image_size,
            down_block_types = down_block_types, up_block_types = up_block_types,
            is_conditional=conditional,block_out_channels=block_out_channels,layers_per_block=layers_per_block).to(device)

    if DDP:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    last_checkpoint_path = os.path.join(args['output_dir'], "checkpoint_{modelname}.pth".format(modelname=modelname))
    if os.path.exists(last_checkpoint_path):
        print('Restarting from the last checkpoint...')
        restart_from_checkpoint(
            os.path.join(args['output_dir'], "checkpoint_{modelname}.pth".format(modelname=modelname)),
            run_variables=to_restore,
            net=net,
            optimizer=optimizer,
            loss_fn=loss_fn)
        
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch,n_epochs):

        train_stats = train_one_epoch(model=net,data_loader=train_loader,noise_scheduler=noise_scheduler,
                                      loss_fn=loss_fn,optimizer=optimizer,epoch=epoch,args=args,use_amp=autocast)

        test_stats = evaluate_one_epoch(model=net,data_loader=test_loader,noise_scheduler=noise_scheduler,
                                         loss_fn=loss_fn,epoch=epoch,args=args,use_amp=autocast)

        # ============ writing logs ... ============
        save_dict = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "loss": loss_fn.state_dict()
        }

        save_on_master(save_dict, os.path.join(args['output_dir'],f"checkpoint_{modelname}.pth"))
            
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            **{f"test_{k}": v for k, v in test_stats.items()},
        }
        
        if is_main_process():
            with (Path(args['output_dir']) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        print('Inference...')
        generate_samples(args=args,device=device,net=net,noise_scheduler=noise_scheduler,epoch=epoch,
                        modelname = modelname,is_conditional=conditional)
        
        val_loss = test_stats["loss"]

        # Early stopping check
        if val_loss < best_val_loss - args['early_stopping_min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            save_on_master(
                {
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "args": args,
                    "loss": loss_fn.state_dict()
                },
                os.path.join(args['output_dir'], f"best_model_{modelname}.pth")
            )
            print(f"Epoch {epoch}: Validation loss improved to {val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch}: No improvement. Patience {patience_counter}/{args['early_stopping_patience']}")

        if patience_counter >= args['early_stopping_patience']:
            print(f"Early stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
            break


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    timestamp = datetime.datetime.now().strftime('%m_%d_%H_%M')
    save_on_master(save_dict, os.path.join(args['output_dir'], f"final_instance_{modelname}_{timestamp}.pth"))


if __name__ == '__main__':


    device =  "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args_saved_path = '/well/papiez/users/zwk579/SubmissionCode/JBHI/configs/simple_diffusion.json'
    if os.path.exists(args_saved_path):
        print(f'Loading args from {args_saved_path}')
        with open(args_saved_path, "r") as f:
            args = json.load(f)
     

    main(args=args,device=device)

