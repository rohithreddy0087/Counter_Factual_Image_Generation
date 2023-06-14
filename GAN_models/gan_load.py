
import torch
from torch import nn

import legacy
import dnnlib

# +
try:
    from GAN_models.StyleGAN2.model import Generator as StyleGAN2Generator
except Exception as e:
    print('StyleGAN2 load fail: {}'.format(e))

class StyleGAN2Wrapper(nn.Module):
    def __init__(self, g):
        super(StyleGAN2Wrapper, self).__init__()
        self.style_gan2 = g
        self.dim_z = 512

    def forward(self, input, input_is_latent=False):
        return self.style_gan2([input], input_is_latent=input_is_latent)[0]

class StyleGAN2ADAWrapper(nn.Module):
    def __init__(self, g):
        super(StyleGAN2ADAWrapper, self).__init__()
        self.style_gan2 = g
        self.dim_z = self.style_gan2.z_dim

    def forward(self, input, input_is_latent=False):
        if input_is_latent:
            return self.style_gan2.synthesis(input)
        else:
            return self.style_gan2(input,c=0)


def make_style_gan2(size, weights):
    if weights.split('.')[-1] == 'pkl':
        with dnnlib.util.open_url(weights) as f:
            G = legacy.load_network_pkl(f)['G_ema']
            G.cuda().eval()
        return StyleGAN2ADAWrapper(G)
    else:
        G = StyleGAN2Generator(size, 512, 8)
        G.load_state_dict(torch.load(weights, map_location='cpu')['g_ema'])
        G.cuda().eval()
        return StyleGAN2Wrapper(G)
