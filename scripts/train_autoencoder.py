from .train_script import train 
from torchvision import transforms
from utils.logging import PlotSamplesCallback
import torch

torch.set_float32_matmul_precision('medium')  # or 'high'



if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    val_transform = transform

    config = '/well/papiez/users/zwk579/SubmissionCode/JBHI/configs/autoencoders/vqgan_perceptual.yaml'
    name = 'VQGAN_perceptual'
    output_dir = f'/logs/autoencoders/{name}/'

    plot_callback = PlotSamplesCallback(output_dir=output_dir, num_samples=10,name=name,on_train_batch_end=False)

    args = {'n_epochs':1000,
            'experiment_name':f'{name}/',
            'image_size':256,
            'batch_size':16,
            'strategy': 'auto',
            'DDP':True,
            'transform':transform,
            'val_transform':val_transform,
            'config':config,
            'output_dir':output_dir,
            'plot_callback':plot_callback,
            'master_port': '12346'}
    
    train(args)
