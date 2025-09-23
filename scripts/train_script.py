
#!/usr/bin/env python3
import torch
import os, sys
import yaml

import sys 
import os 


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.getcwd()) 

from utils.utils import instantiate_from_config
from utils.training import find_last_checkpoint
from utils.dataset import GenericDatasetClass
from utils.dist import init_DDP


def train(args):

    torch.set_float32_matmul_precision('medium') 
    name = args['experiment_name']
    n_epochs = args['n_epochs']
    batch_size = args['batch_size']
    DDP = args['DDP']
    output_dir = args['output_dir']
    config_path = args['config']
    strategy = args['strategy']
    transform = args['transform']
    val_transform = args['val_transform']
    plot_callback = args['plot_callback']
    num_devices = torch.cuda.device_count()


    config = yaml.safe_load(open(config_path, "r"))

    if DDP:
        init_DDP(args)


    data = GenericDatasetClass(transform=transform,val_transform=val_transform)

    train_loader, test_loader = data.get_dataloaders(batch_size=batch_size)
    
    model = instantiate_from_config(config['model'])
    model.save_configs(config_path=config_path,output_dir=output_dir,experiment_name=name)

    
    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1, save_on_train_epoch_end=True)

    logger = TensorBoardLogger("lightning_logs", name=name)

    checkpoint = find_last_checkpoint(log_dir='lightning_logs',experiment_name=name)

    print('Found checkpoint:',checkpoint)

    trainer = Trainer(max_epochs=n_epochs,accelerator='gpu',devices=num_devices,
                    log_every_n_steps=100,enable_checkpointing=True,
                      strategy=strategy,callbacks=[checkpoint_callback,plot_callback],logger=logger,use_distributed_sampler=False)
    
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=test_loader,ckpt_path=checkpoint)
