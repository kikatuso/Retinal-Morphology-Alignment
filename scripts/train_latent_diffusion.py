from .train_script import train 
from torchvision import transforms
from utils.logging import PlotSamplesDiffusion
import torch

torch.set_float32_matmul_precision('medium')  # or 'high'



if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    val_transform = transform

    config = '/well/papiez/users/zwk579/SubmissionCode/JBHI/configs/stable_diffusion/v_prediction.yaml'
    name = 'latent_diffusion_v_prediction'
    output_dir = f'logs/latent_diffusion/{name}/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plot_callback = PlotSamplesDiffusion(num_samples=16, device=device, output_dir=output_dir, name=name)


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
