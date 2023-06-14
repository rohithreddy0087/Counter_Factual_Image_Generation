import torch.nn as nn
import torchvision

class ResNet35(nn.Module):
    def __init__(self, num_classes=39):
        super(ResNet35, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)