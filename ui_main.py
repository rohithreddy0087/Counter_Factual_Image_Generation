import streamlit as st
import numpy as np
from glob import glob
import collections
import random

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

from classifier_networks import VGG, vgg_layers, CNN
from GAN_models.gan_load import make_style_gan2
import logging

GAN_weight_file = 'GAN_models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt'
classifier_weight_file = 'classifier_weights/celebA_MultiLabels_vgg11_classifier.pt'
classifier_network= 'VGG'
class_count= 39
latent_size= 512
gan_resolution= 1024
gan_output_channel= 3
classifier_input_size= 256
classifier_input_channel=3
noise_scale= 0.1
training_name= 'Bald_Male_Smile_Young_Multi'
multi_task = True
subset_labels = [3, 19, 30, 38]
device = torch.device('cuda:0')

class FCShiftPredictor(nn.Module):
    def __init__(self,input_dim,class_dim, inner_dim, output_dim):
        super(FCShiftPredictor, self).__init__()
        self.fc_direction = nn.Sequential(
            nn.Linear(input_dim+class_dim,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(inner_dim,output_dim),
        )
        
    def forward(self, x,c):
        x_c = torch.cat((x,c),1)
        dir_ = self.fc_direction(x_c)
        return dir_

def get_logger():
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger('DD')
    fileHandler = logging.FileHandler("debug.log")
    fileHandler.setFormatter(log_formatter)
    logger .addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(log_formatter)
    logger .addHandler(consoleHandler)

    logger.setLevel(logging.DEBUG)

    return logger

def get_classifer():
    classifier_weights = torch.load(classifier_weight_file)
    if isinstance(classifier_weights, collections.OrderedDict):
        if classifier_network == 'VGG':
            classifier = VGG(vgg_layers,class_count)
        else:
            raise ValueError('Classifier architecture {} is not defined.'.format(classifier_network))
        classifier.load_state_dict(classifier_weights)
    elif isinstance(classifier_weights,dict):
        if 'model' in classifier_weights:
            classifier_weights = classifier_weights['model']
        else:
            logger.debug('Classifier weights file includes:',classifier_weights.keys())
            raise ValueError('Classifier weight file has an unknown format.')
    else:
        classifier = classifier_weights

    classifier = classifier.eval().to(device)
    return classifier

def get_generator():
    G = make_style_gan2(gan_resolution, GAN_weight_file)
    G.eval()
    G = G.to(device)
    return G

def get_shift_predictor():
    weight_file_name = 'trained_weights/shift_in_w_Bald_Male_Smile_Young_Multi_3_19_30_38_0.050.pt'
    # logger.debug("%s", weight_file_name)
    previous_weights = sorted(glob(weight_file_name))
    # logger.debug('Loading previously trained weights ...')
    shift_model = torch.load(previous_weights[-1])
    shift_model.eval()
    shift_model = shift_model.to(device)
    return shift_model


def generate_image():
    z = torch.randn([1,latent_size]).repeat(1,1).to(device)
    z_noise = noise_scale * torch.randn([1,latent_size]).to(device)
    z_perturb = z + z_noise
    w = generator.style_gan2.get_latent(z_perturb)
    img_orig = generator.style_gan2([w] , input_is_latent=True)[0]
    img_orig = resize_transform(img_orig)
    y_orig = classifier(img_orig)
    y_orig = torch.sigmoid(y_orig[0])
    ig = img_orig[0].permute(1, 2, 0).cpu().detach().numpy()
    # print(ig)
    return ig, y_orig, w

def generate_counterfactual_image(w, y_target):
    dir_pred = shift_predictor(w,y_target)
    img_shift = generator.style_gan2([w + dir_pred] , input_is_latent=True)[0]
    ig = img_shift[0].permute(1, 2, 0).cpu().detach().numpy()
    # print(ig)
    return ig

def main():
    # Generate the initial image
    if count == 0:
        generated_image = np.zeros((256,256,3))
        counterfactual_image = np.zeros((256,256,3))

    st.title("Counterfactual Image Generator")
    
    # Generate Image button
    if st.button("Generate Image"):
        generated_image, y_orig, latents = generate_image()
        y_target = ~(y_orig[:,subset_labels]>0.5)
        # logger.debug("Traget variables %s", y_target)
        # index1 = random.randint(0, len(y_target) - 1)
        # index2 = random.randint(0, len(y_target) - 1)
        # y_target[index1] = ~(y_target[index1])
        # y_target[index2] = ~(y_target[index2])
        y_target = y_target.int()
        # logger.debug("Traget variables %s", y_target)
    
        counterfactual_image = generate_counterfactual_image(latents, y_target)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Generated Image")
        st.image(generated_image, use_column_width=True, clamp=True)

    with col2:
        st.subheader("Counterfactual Image")
        st.image(counterfactual_image, use_column_width=True, clamp=True)

if __name__ == '__main__':
    logger = get_logger()
    classifier = get_classifer()
    generator = get_generator()
    shift_predictor = get_shift_predictor()
    resize_transform = transforms.Resize((classifier_input_size,classifier_input_size))
    count = 0
    main()
