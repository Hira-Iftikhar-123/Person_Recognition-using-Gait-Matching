import torch.nn as nn
from torchvision import models

class ResNet101Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)