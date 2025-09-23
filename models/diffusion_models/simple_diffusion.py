import torch.nn.functional as F
from diffusers import UNet2DConditionModel,UNet2DModel
from torch import nn
import torch
import sys 

# source: https://huggingface.co/learn/diffusion-course/en/unit2/3

def sample_random_conditioning(
    num_samples: int,
    device='cpu',
    age_mean: float = 50.0,
    age_std: float = 5.0,
    age_min: float = 40.0,
    age_max: float = 69.0,
    p_hypertension: float = 0.3,
    p_female: float = 0.5,
    use_zscore: bool = False
) -> torch.Tensor:
    """
    Generates random conditioning vectors for diffusion model inference.

    Returns:
        torch.Tensor: (num_samples, 8) with:
            [hypertension, IsFemale, Age, Ethnicity_Asian, Black, White, Other, Mixed]
    """
    hypertension = torch.bernoulli(torch.full((num_samples, 1), p_hypertension))
    is_female = torch.bernoulli(torch.full((num_samples, 1), p_female))

    age = torch.normal(mean=age_mean, std=age_std, size=(num_samples, 1))

    if use_zscore:
        age_feat = (age - age_mean) / age_std
    else:
        age_feat = (age - age_min) / (age_max - age_min)

    ethnicity_idx = torch.randint(0, 5, (num_samples,))
    ethnicity_onehot = F.one_hot(ethnicity_idx, num_classes=5).float()

    cond = torch.cat([hypertension, is_female, age_feat, ethnicity_onehot], dim=1).to(device)
    return cond


class ConditioningEncoder(nn.Module):
    def __init__(self, class_emb_size=32,num_variables=8):
        super().__init__()
        self.emb_size = class_emb_size

        self.embed = nn.Linear(num_variables, class_emb_size)

        self.combiner = nn.Sequential(
            nn.ReLU(),
            nn.Linear(class_emb_size, class_emb_size),
            nn.ReLU(),
        )

    def forward(self, cond):
        """
        Args:
            cond (torch.Tensor): shape (batch_size, 8), numeric features (float or long),
                                 already preprocessed (e.g., one-hot ethnicity, normalized age)
        Returns:
            torch.Tensor: shape (batch_size, class_emb_size)
        """
        cond = self.embed(cond.float())  # ensure float
        cond_emb = self.combiner(cond)
        return cond_emb.unsqueeze(1)  # Add sequence dimension for compatibility with UNet2DConditionModel
        
    def encode(self,cond):
        return self.forward(cond)
    
class ConditioningEncoder2(nn.Module):
    def __init__(self, class_emb_size=32, num_variables=8, hidden_mult=2, dropout=0.05):
        super().__init__()
        hidden_size = class_emb_size * hidden_mult

        self.embed = nn.Linear(num_variables, hidden_size)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, class_emb_size)
        )

    def forward(self, cond):
        x = self.embed(cond.float())
        cond_emb = self.mlp(x)
        return cond_emb.unsqueeze(1)  # Add sequence dimension for UNet

    def encode(self, cond):
        return self.forward(cond)


class ClassConditionedUnet(nn.Module):
    def __init__(self,num_channels=3, image_size=(256, 256), norm_num_groups=32,
                 layers_per_block=1,  is_conditional=True,
                 block_out_channels=(64,64,128,128,192,256),  # Match channel_mult × model_channels
                 class_embed_size=32,num_variables=8,
                 down_block_types=('DownBlock2D','DownBlock2D',"DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                 up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D","UpBlock2D", "UpBlock2D")):  # Mirror structure
        super().__init__()

        self.is_conditional = is_conditional 

        if is_conditional:
            print("Using conditional UNet with class embeddings.")
            self.condition_encoder = ConditioningEncoder(class_emb_size=class_embed_size,num_variables=num_variables)
            self.model = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=num_channels,
            out_channels=num_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=class_embed_size)
        else:
            print("Using unconditional UNet, no class embeddings.")
            down_block_types = list(down_block_types)
            up_block_types = list(up_block_types)
            down_block_types = [block.replace("CrossAttnDownBlock2D", "DownBlock2D") for block in down_block_types]
            up_block_types = [block.replace("CrossAttnUpBlock2D", "UpBlock2D") for block in up_block_types]

            self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=num_channels,
            out_channels=num_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            norm_num_groups=norm_num_groups)

        params_info = self.read_params_size()
        for submodule, count in params_info.items():
            print(f"{submodule}: {count:,} params")

    def read_params_size(self):
        """
        Returns a dict with the number of trainable parameters for each top-level submodule,
        as well as the total trainable parameters count.
        """
        param_counts = {}
        
        for name, module in self.named_children():
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_counts[name] = num_params
        
        # Total params in the entire model
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_counts['total'] = total_params
        
        return param_counts

    def forward(self, x, t,cond=None):

        bs, _, w, h = x.shape
        if self.is_conditional and cond is None:
            raise ValueError("Conditioning vector 'cond' must be provided for conditional UNet.")
        if self.is_conditional:
            cond_emb = self.condition_encoder(cond)
            out = self.model(x, timestep=t, encoder_hidden_states=cond_emb).sample
        else:
            out = self.model(x, timestep=t).sample
        return out
