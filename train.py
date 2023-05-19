import torch
from torchvision import transforms
import torch.nn as nn
import random
import os
import sys
from glob import glob
import collections

device = torch.device('cuda:0')

import argparse

parser = argparse.ArgumentParser(description='Train shift predictor to detect causal paths in latent space of a GAN.')
parser.add_argument('--gan_weights', type=str,\
                    default = 'GAN_models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt',\
                    help='Path to the GAN model weights.')
parser.add_argument('--classifier_weights', type=str,\
                    default = 'classifier_weights/celebA_MultiLabels_vgg11_classifier.pt',\
                    help='Path to the target classifier weights.')
parser.add_argument('--classifier_network', type=str,\
                    default = 'VGG',\
                    choices=['VGG', 'CNN', 'ResNet50'],
                    help='Classifier network architecture.')
parser.add_argument('--class_count', type=int, default = 39, help='Classification dim.')
parser.add_argument('--latent_dim', type=int, default = 512, help='Input latent code dim.')
parser.add_argument('--faithfulness_ratio', type=float, default = 0.05, help='loss ratio of shift size to shift direction.')

parser.add_argument('--gan_output_dim', type=int, default = 1024, help='The GAN output dim')
parser.add_argument('--gan_output_channel', type=int, default = 3, help='The GAN output channel dim')
parser.add_argument('--classifier_input_dim', type=int, default = 256, help='The dimension of input image for classifier')
parser.add_argument('--classifier_input_channel', type=int, default = 3, help='The channel size of input image for classifier')
parser.add_argument('--batch_size', type=int, default = 8, help='Training batch size.')
parser.add_argument('--n_steps', type=int, default = 100000, help='number of training iterations.')
parser.add_argument('--lr', type=float, default = 1e-4, help='learning rate.')
parser.add_argument('--batch_noise_scale', type=float, default = 0.1, help='noise variance for latent inputs in each batch')
parser.add_argument('--training_name', type=str, default = 'FACE_Multi', help='A label for the training results.')
parser.add_argument('--multi_task', type=bool, default = True, help='Classifier predicts more than one label.')
parser.add_argument('--subset_labels', nargs="+", type=int, default = [25,11,17,28,27,34,16,3], help='Train of a subset of predicted labels in case of multi-task classifier.')

parser.add_argument('--in_w', dest='in_W', action='store_true' ,help='Run in w space of StyleGAN (instead of z)')
parser.add_argument('--in_z', dest='in_W', action='store_false',help='Run in z space of StyleGAN (instead of w)')
parser.set_defaults(in_W=True)

args = parser.parse_args()
# -

print(f'Training in W space of GAN: {args.in_W}')

from classifier_networks import VGG, vgg_layers, CNN
from GAN_models.gan_load import make_style_gan2



class Args:
    def __init__(self):
        self.gan_weights= 'GAN_models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt'
        self.classifier_weights= 'classifier_weights/celebA_MultiLabels_vgg11_classifier.pt'
        self.classifier_network= 'VGG'
        self.class_count= 39
        self.latent_dim= 512
        self.faithfulness_ratio= 0.05
        self.gan_output_dim= 1024
        self.gan_output_channel= 3
        self.classifier_input_dim= 256
        self.classifier_input_channel=3
        self.batch_size= 2
        self.n_steps= 100000
        self.lr= 1e-4
        self.batch_noise_scale= 0.1
        self.training_name= 'FACE_Multi'
        self.multi_task = True
        self.subset_labels = [38, 17, 1, 19, 14, 8, 18, 21, 3]
        self.in_W = True


GAN_weight_file = args.gan_weights
classifier_weight_file = args.classifier_weights
classifier_network = args.classifier_network
class_count = args.class_count
gan_resolution = args.gan_output_dim
batch_size = args.batch_size
latent_size = args.latent_dim
classifier_input_size = args.classifier_input_dim
noise_scale = args.batch_noise_scale

shift_predictor_lr = args.lr
n_steps = args.n_steps

faithfulness_ratio = args.faithfulness_ratio

training_name = args.training_name

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


classifier_weights = torch.load(classifier_weight_file)

if isinstance(classifier_weights, collections.OrderedDict):
    if classifier_network == 'VGG':
        classifier = VGG(vgg_layers,class_count)
    elif classifier_network == 'CNN':
        classifier = CNN()
    elif classifier_network == 'ResNet50':
        raise ValueError('Classifier architecture {} is not implemented.'.format(classifier_network))
    else:
        raise ValueError('Classifier architecture {} is not defined.'.format(classifier_network))
    classifier.load_state_dict(classifier_weights)

elif isinstance(classifier_weights,dict):
    if 'model' in classifier_weights:
        classifier_weights = classifier_weights['model']
    else:
        print('Classifier weights file includes:',classifier_weights.keys())
        raise ValueError('Classifier weight file has an unknown format.')
else:
    classifier = classifier_weights

classifier = classifier.eval().to(device)
# -

G = make_style_gan2(gan_resolution, GAN_weight_file)
G.eval()
G = G.to(device)

if len(args.subset_labels) > 0:
    class_count = len(args.subset_labels)
    lbl_ids = '_'.join([str(i) for i in args.subset_labels])
else:
    lbl_ids = 'all_lbls'

weight_file_name = 'trained_weights/shift_in_{}_{}_{}_{:1.3f}.pt'.format('w' if args.in_W else 'z', training_name,lbl_ids,faithfulness_ratio)

load_pretrained = False
previous_weights = sorted(glob(weight_file_name))

if load_pretrained and len(previous_weights) > 0:
    print('Loading previously trained weights ...')
    shift_model = torch.load(previous_weights[-1])
    previous_steps = int(previous_weights[-1].split('_')[-1].split('.')[0])
    print('Last trained step found: {:08d}'.format(previous_steps))
else:
    shift_model = FCShiftPredictor(latent_size,class_count,1024,latent_size)
    previous_steps = 0

shift_model = shift_model.to(device).train()

shift_model_opt = torch.optim.Adam(shift_model.parameters(), lr=shift_predictor_lr)

resize_transform = transforms.Resize((classifier_input_size,classifier_input_size))

criterion = nn.BCEWithLogitsLoss()

for step in range(n_steps):
    shift_model.zero_grad()
    z = torch.randn([1,latent_size]).repeat(batch_size,1).to(device)
    z_noise = noise_scale * torch.randn([batch_size,latent_size]).to(device)
    z_perturb = z + z_noise
    if args.in_W:
        w = G.style_gan2.get_latent(z_perturb)
        img_orig = G.style_gan2([w] , input_is_latent=True)[0]
    else:
        w = z_perturb
        img_orig = G(w)
    
    if gan_resolution != classifier_input_size:
        img_orig = resize_transform(img_orig)
    y_orig = classifier(img_orig)
    
    y_orig = torch.sigmoid(y_orig[0])  
   
    y_target = torch.randint(-1, 2, y_orig[:,args.subset_labels].shape).to(device)
    
    loss_filter = abs(y_target) > 0 # for which attributes the loss should be applied
    
    while not (loss_filter).any():
        y_target = torch.randint(-1, 2, y_orig[:,args.subset_labels].shape).to(device)
        loss_filter = abs(y_target) > 0
    
    dir_pred = shift_model(w,y_target)
    
    if args.in_W:
        img_shift = G.style_gan2([w + dir_pred] , input_is_latent=True)[0]
    else:
        img_shift = G(w + dir_pred)

    if gan_resolution != classifier_input_size:
        img_shift = resize_transform(img_shift)
    
    if args.gan_output_channel == 1 and args.classifier_input_channel > 1:
        img_shift = img_shift.repeat([1,args.classifier_input_channel,1,1])
    
    y_shift = classifier(img_shift)
    
    if isinstance(y_shift,tuple):
        y_shift = y_shift[0]
    
    y_out = y_shift
    
    y_target = (y_target + 1.0) / 2.0 # map from [-1 0 1] to [0.0 0.5 1.0] for loss computation purposes
    
    if len(args.subset_labels) > 0:
        y_out = y_out[:,args.subset_labels]
    
    dir_loss = criterion(y_out[loss_filter],y_target[loss_filter].float())
    scale_loss = torch.mean(torch.norm(dir_pred,dim=1))
    loss =  dir_loss + faithfulness_ratio * scale_loss
    
    print("STEP {:08d} CLASS LOSS: {:1.8f}  SHIFT SIZE LOSS: {:1.8f}".format(previous_steps + step+1, dir_loss, scale_loss))
    
    loss.backward()
    shift_model_opt.step()
    
    if (step+1) % 10000 == 0:
        torch.save(shift_model, weight_file_name)