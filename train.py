import torch
from torchvision import transforms
import torch.nn as nn
import collections
from config_parser import ConfigFileparser

from classifier_models.vgg import VGG, vgg_layers
from classifier_models.resnet import ResNet35

from GAN_models.gan_load import make_style_gan2
from GAN_models.dcgan import Generator

from shift_predictor_models.models import MLP, MLPDropout, MLPThreeLayer, MLPThreeLayerDropout

device = torch.device('cuda:0')

if __name__ == "__main__":
    configfile = "config.ini"
    config = ConfigFileparser(configfile)

    classifier_weights = torch.load(config.classifier_weight_file)
    if isinstance(classifier_weights, collections.OrderedDict):
        if config.classifier_network == 'VGG':
            classifier = VGG(vgg_layers, config.class_count)
        elif config.classifier_network == 'ResNet':
            classifier = ResNet35()
        else:
            raise ValueError('Classifier architecture {} is not defined.'.format(config.classifier_network))
        classifier.load_state_dict(classifier_weights)
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

    if config.load_pretrained:
        config.logger.debug('Loading previously trained weights ...')
        shift_model = torch.load(config.shift_predictor_weight_file)
    else:
        if config.shift_predictor_model == 1:
            shift_model = MLP(config.latent_size, config.class_count, 1024, config.latent_size)
        elif config.shift_predictor_model == 2:
            shift_model = MLPDropout(config.latent_size, config.class_count, 1024, config.latent_size)
        elif config.shift_predictor_model == 3:
            shift_model = MLPThreeLayer(config.latent_size, config.class_count, 1024, config.latent_size)
        elif config.shift_predictor_model == 4:
            shift_model = MLPThreeLayerDropout(config.latent_size, config.class_count, 1024, config.latent_size)
        else:
            shift_model = MLPDropout(config.latent_size, config.class_count, 1024, config.latent_size)

    shift_model = shift_model.to(device).train()
    shift_model_opt = torch.optim.Adam(shift_model.parameters(), lr = config.shift_predictor_lr)
    resize_transform = transforms.Resize((config.classifier_input_size, config.classifier_input_size))
    criterion = nn.BCEWithLogitsLoss()

    for step in range(config.n_steps):
        shift_model.zero_grad()
        if config.gan_model == "DCGAN":
            z = torch.randn(1, config.latent_size, 1, 1, device=device).repeat(64, 1, 1, 1)
            z_noise = config.noise_scale * torch.randn(64, config.latent_size, 1, 1, device=device)
        else:
            z = torch.randn([1,config.latent_size]).repeat(config.batch_size,1).to(device)
            z_noise = config.noise_scale * torch.randn([config.batch_size, config.latent_size]).to(device)
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
        y_target = ~(y_orig[:,config.subset_labels]>0.5)
        y_target = y_target.int()
        loss_filter = abs(y_target) > 0
        
        while not (loss_filter).any():
            y_target = torch.randint(-1, 2, y_orig[:,config.subset_labels].shape).to(device)
            loss_filter = abs(y_target) > 0

        dir_pred = shift_model(w,y_target)
        
        if config.gan_model == "DCGAN":
            img_shift = G(z_perturb + dir_pred)
        else:
            img_shift = G.style_gan2([w + dir_pred] , input_is_latent=True)[0]

        if config.gan_resolution != config.classifier_input_size:
            img_shift = resize_transform(img_shift)
        
        y_shift = classifier(img_shift)
        
        if isinstance(y_shift,tuple):
            y_shift = y_shift[0]
        
        y_out = torch.sigmoid(y_shift)
                
        if len(config.subset_labels) > 0:
            y_out = y_out[:,config.subset_labels]
        
        dir_loss = criterion(y_out[loss_filter],y_target[loss_filter].float())
        scale_loss = torch.mean(torch.norm(dir_pred,dim=1))
        loss =  dir_loss + config.faithfulness_ratio * scale_loss
        
        config.logger.debug("STEP {:08d} CLASS LOSS: {:1.8f}  SHIFT SIZE LOSS: {:1.8f}".format(step+1, dir_loss, scale_loss))
        
        loss.backward()
        shift_model_opt.step()
        
        if (step+1) % 1000 == 0:
            torch.save(shift_model, config.shift_predictor_weight_file)