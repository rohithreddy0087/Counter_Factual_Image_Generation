# Latent Transformation-Based Counterfactual Image Generation using Neural Networks and GANs

This project focuses on the generation of counterfactual images, which are images that represent alternative realities by altering object appearance or other attributes. By leveraging advancements in deep generative models, our goal is to produce high-quality counterfactual images that are visually similar to the original images while incorporating desired changes. The generated counterfactual images have the potential to enhance applications such as image editing and explainable AI, providing valuable insights and explanations for deep networks.

## Method
![CF flowchart](https://github.com/rohithreddy0087/Counter_Factual_Image_Generation/assets/51110057/23ad575f-7a88-4934-b8ed-b385cd703c42)

The method used to generate counterfactual images involves the following steps:

1. Generate Samples: Utilize a well-trained GAN model, such as StyleGAN or DCGAN, trained on a specific dataset (e.g., CelebA). Randomly sample images by providing Gaussian noise as input to the GAN model. In DCGAN, this noise input corresponds to the latent space, while in StyleGAN, additional processing is required to obtain the latent vectors.
2. Classification on Randomly Sampled Images: Employ a pre-trained classifier like VGG or ResNet, trained on the same dataset, to classify the attributes of the randomly generated images. These attributes are considered the "original attributes."
3. Choosing Target Attributes: Select target attributes either randomly or manually to specify the desired changes in the counterfactual image. These attributes represent the attributes that need to be countered or changed. These attributes are considered the "target attributes."
4. Shift Retainer: Train a shift retainer, typically a MLP model, with regularization or dropout layers for improved performance. The shift predictor takes the original latent representation of the image and the target attributes as input and generates a counterfactual latent representation. It acts as a non-linear function to transform the original latent space to generate the counterfactual latent space.
5. Generating Counterfactual Image: Utilize the generated counterfactual latent representation to generate the counterfactual image. This is done by feeding the counterfactual latent representation into the GAN model, which generates the desired counterfactual image based on the specified target attributes.
6. Binary Cross Entropy Loss: Calculate the Binary Cross Entropy (BCE) loss between the target attributes and the attributes obtained by classifying the counterfactual image. This loss quantifies the dissimilarity between the desired target attributes and the predicted attributes. Backpropagate this loss through the shift retainer model to update its parameters.
    
By optimizing the BCE loss, the shift retainer learns to generate counterfactual latent representations that produce counterfactual images with attributes closer to the target attributes. This iterative training process improves the ability of the shift retainer to generate accurate counterfactual images.

## Prerequisites
- Linux, Windows or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/rohithreddy0087/Counter_Factual_Image_Generation
cd Counter_Factual_Image_Generation
python3 -m venv myenv
source myenv/bin/activate
```
- Install [PyTorch](http://pytorch.org) and other dependencies.
- For pip users, please type the command `pip3 install -r requirements.txt`.

### Pretrained Weights
- Pretrained weights of DCGAN (256x256) on celebA dataset:
- Pretrained weights of StyleGAN on celebA dataset: 
- Pretrained weights of VGG on celebA dataset:
- Pretrained weights of VGG on ResNet dataset:

### Configuration
- Use the config.ini to choose among different GAN, classifier and Shift Retainer Models
- Mention the paths to the trained weights correctly
- Example of a config file:
```
bash

```

### Training and Testing
- Change the config accordingly and run the `train.py` file
```bash
python3 train.py
``` 
### Executing Streamlit
```bash
streamlit run ui_main.py
```
Now, in your browser, open http://localhost:8501/

The test images are present in the test_images folder.

## Demo
A small video demo of the Instance eraser streamlit application


