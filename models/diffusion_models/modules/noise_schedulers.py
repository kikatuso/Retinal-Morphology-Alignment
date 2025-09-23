
import torch
import numpy as np 



def make_beta_schedule(schedule, n_timestep, beta_start=1e-4, beta_end=2e-2, cosine_s=0.008):

    if schedule == "linear":
        betas = (
                torch.linspace(beta_start, beta_end, n_timestep, dtype=torch.float64)
        )
    elif schedule == "scaled_linear":
        betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timestep)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)

        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(beta_start, beta_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(beta_start, beta_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def ddim_timesteps(num_ddim_steps=150, total_steps=1000, method='last', prediction_type='x0'):
    if method == 'linspace':
        timesteps = np.linspace(0, total_steps - 1, num_ddim_steps).round().astype(np.int64)
    elif method == 'leading':
        step_ratio = total_steps / num_ddim_steps
        timesteps = ((np.arange(0, num_ddim_steps) * step_ratio).round().astype(np.int64))
    elif method == 'trailing':
        step_ratio = total_steps / num_ddim_steps
        timesteps = (np.round(np.arange(total_steps, 0, -step_ratio)).astype(np.int64) - 1)[::-1]
    elif method == 'last':
        if prediction_type =='x0':
            timesteps = range(0,num_ddim_steps) # the last 150 steps from the DPMS
        else:
            timesteps = range(total_steps - num_ddim_steps, total_steps)  # the last 150 steps from the DPMS
        timesteps = np.array(list(timesteps)).astype(np.int64)
    elif method == 'quadratic':
        timesteps = ((np.linspace(0, np.sqrt(total_steps * 0.8), num_ddim_steps)) ** 2).astype(int)
        timesteps = make_unique(timesteps)
    else:
        raise ValueError(f"Unknown method {method}")

    # 🔑 Ensure correct endpoint
    timesteps[0] = 0  
    timesteps[-1] = total_steps - 1 

    return timesteps


def make_unique(arr):
    """
    Ensure all integers in arr are unique by incrementing duplicates by +1
    until uniqueness is achieved. Keeps array length unchanged.
    """
    unique_arr = []
    seen = set()
    for val in arr:
        while val in seen:
            val += 1
        seen.add(val)
        unique_arr.append(val)
    return np.array(unique_arr, dtype=np.int64)