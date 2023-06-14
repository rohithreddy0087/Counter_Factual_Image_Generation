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

from metrics import calculate_proximity, calculate_validity

device = torch.device('cuda:0')

def evaluation_step(engine, batch):
    with torch.no_grad():
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
        y_target = ~(y_orig[:,config.subset_labels]>0.5)
        y_target = y_target.int()
        loss_filter = abs(y_target) > 0
        
        while not (loss_filter).any():
            y_target = torch.randint(-1, 2, y_orig[:,config.subset_labels].shape).to(device)
            loss_filter = abs(y_target) > 0

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
        return img_orig, img_shift

if __name__ == "__main__":
    configfile = "config.ini"
    config = ConfigFileparser(configfile)

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

    avg_prox = 0
    avg_val = 0

    for step in range(0):
        if config.gan_model == "DCGAN":
            z = torch.randn(1, config.latent_size, 1, 1, device=device).repeat(config.batch_size, 1, 1, 1)
            z_noise = config.noise_scale * torch.randn(config.batch_size, config.latent_size, 1, 1, device=device)
            z_perturb = z + z_noise
            
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
        
        y_shift = classifier(img_shift)
        
        if isinstance(y_shift,tuple):
            y_shift = y_shift[0]
        
        y_out = torch.sigmoid(y_shift)
                
        if len(config.subset_labels) > 0:
            y_out = y_out[:,config.subset_labels]
        
        prox = calculate_proximity(img_orig, img_shift)
        avg_prox += prox

        val = calculate_validity(y_out, y_target)
        avg_val += val
        
        config.logger.debug("STEP {:08d} Proximity: {:1.8f}  Validity: {:1.8f}".format(step+1, prox, val))
    
    config.logger.debug("Average Proximity: {:1.8f}  Average Validity: {:1.8f}".format(avg_prox, avg_val))


    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])
    evaluator = Engine(evaluation_step)
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")
    evaluator.run(epoch_length=1, max_epochs=1) # use your test data loader, NOT training data loader
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']

    config.logger.debug(f"Metric Scores")
    config.logger.debug(f"*   FID : {fid_score:4f}")
    config.logger.debug(f"*    IS : {is_score:4f}")