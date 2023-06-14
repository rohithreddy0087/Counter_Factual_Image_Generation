import torch
from torchvision import transforms
import torch.nn as nn
import collections
from config_parser import ConfigFileparser

from ignite.metrics import FID, InceptionScore
from ignite.engine import Engine

from classifier_models.vgg import VGG, vgg_layers
from classifier_models.resnet import ResNet35

from GAN_models.gan_load import make_style_gan2
from GAN_models.dcgan import Generator

import streamlit as st
import numpy as np

device = torch.device('cuda:0')

def generate_image():
    if config.gan_model == "DCGAN":
        z = torch.randn(1, config.latent_size, 1, 1, device=device).repeat(4, 1, 1, 1)
        z_noise = config.noise_scale * torch.randn(4, config.latent_size, 1, 1, device=device)
        z_perturb = z + z_noise
        
    else:
        z = torch.randn([1,config.latent_size]).repeat(4,1).to(device)
        z_noise = config.noise_scale * torch.randn([4, config.latent_size]).to(device)
        z_perturb = z + z_noise

    if config.gan_model == "DCGAN":
        img_orig = G(z_perturb)
    else:
        w = G.style_gan2.get_latent(z_perturb)
        img_orig = G.style_gan2([w] , input_is_latent=True)[0]
    
    if config.gan_resolution != config.classifier_input_size:
        img_orig = resize_transform(img_orig)

    y_orig = classifier(img_orig)
    y_orig = torch.sigmoid(y_orig[0]) 
    ig = img_orig[0].permute(1, 2, 0).cpu().detach().numpy()
    return ig, y_orig, w, z_perturb

def generate_counterfactual_image(w, z_perturb, y_target):
    if config.gan_model == "DCGAN":
        z_perturb_tmp = z_perturb.view(config.batch_size, config.latent_size)
        dir_pred = shift_model(z_perturb_tmp,y_target)
    else:
        dir_pred = shift_model(w,y_target)
    
    if config.gan_model == "DCGAN":
        dir_pred = dir_pred.view(config.batch_size, config.latent_size, 1, 1)
        img_shift = G(z_perturb + dir_pred)
    else:
        img_shift = G.style_gan2([w + dir_pred] , input_is_latent=True)[0]

    if config.gan_resolution != config.classifier_input_size:
        img_shift = resize_transform(img_shift)
    ig = img_shift[0].permute(1, 2, 0).cpu().detach().numpy()
    return ig

def main():
    if count == 0:
        generated_image = np.zeros((256,256,3))
        counterfactual_image = np.zeros((256,256,3))

    st.title("Counterfactual Image Generator")
    
    # Generate Image button
    if st.button("Generate Image"):
        generated_image, y_orig, latents, z_perturb = generate_image()
        y_target = ~(y_orig[:,config.subset_labels]>0.5)
        y_target = y_target.int()    
        counterfactual_image = generate_counterfactual_image(latents, z_perturb, y_target)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Generated Image")
        st.image(generated_image, use_column_width=True, clamp=True)

    with col2:
        st.subheader("Counterfactual Image")
        st.image(counterfactual_image, use_column_width=True, clamp=True)
    
if __name__ == "__main__":
    configfile = "config.ini"
    config = ConfigFileparser(configfile)
    count = 0

    classifier_weights = torch.load(config.classifier_weight_file)
    if isinstance(classifier_weights, collections.OrderedDict):
        if config.classifier_network == 'VGG':
            classifier = VGG(vgg_layers, config.class_count)
            classifier.load_state_dict(classifier_weights)
        elif config.classifier_network == 'ResNet':
            classifier = ResNet35()
            classifier.resnet.load_state_dict(classifier_weights)
        else:
            raise ValueError('Classifier architecture {} is not defined.'.format(config.classifier_network))
        
    elif isinstance(classifier_weights,dict):
        if 'model' in classifier_weights:
            classifier_weights = classifier_weights['model']
        else:
            config.logger.debug('Classifier weights file includes: %s',classifier_weights.keys())
            raise ValueError('Classifier weight file has an unknown format.')
    else:
        classifier = classifier_weights

    classifier = classifier.eval().to(device)

    if config.gan_model == "DCGAN":
        G = Generator(config.latent_size, 64, config.gan_output_channel)
        G = torch.load(config.gan_weight_file)
    else:
        G = make_style_gan2(config.gan_resolution, config.gan_weight_file)
    G.eval()
    G = G.to(device)

    shift_model = torch.load(config.shift_predictor_weight_file)
    
    shift_model = shift_model.to(device).eval()
    resize_transform = transforms.Resize((config.classifier_input_size, config.classifier_input_size))