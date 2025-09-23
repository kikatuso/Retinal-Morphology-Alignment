import torch
import pickle
import matplotlib.pyplot as plt

with open('/gpfs3/well/papiez/users/zwk579/.cache/dnnlib/downloads/2d485eb6b15693993325dd7df7afbfcc_last.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = 1                                # class labels 
w = G.mapping(z, c, truncation_psi=1.0)
img = G.synthesis(w, noise_mode='const', force_fp32=True)

print('Conditional dimension:',G.c_dim)  # conditional dimension

img = img.detach().cpu().squeeze(0).numpy()  # convert to numpy

img = img.transpose(1, 2, 0)  # NHWC

fig,ax = plt.subplots(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.savefig('generated_image.png', bbox_inches='tight', pad_inches=0)
