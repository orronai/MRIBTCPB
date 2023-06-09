import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim.lr_scheduler as lr_scheduler

from datasets import IMAGE_SIZE


# define bottleneck for resnet model. i.e. one block of resnet 50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, drop_rate=0.3):
        super(Bottleneck, self).__init__()
        self.drop_rate = drop_rate
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = F.dropout(out, p=self.drop_rate)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# define resnet model
class Resnet50Model(nn.Module):
    def __init__(self, in_channels=3, block=Bottleneck, num_classes=4, drop_rate=0.3):
        super(Resnet50Model, self).__init__()
        self.in_planes = 64
        self.in_channels = in_channels
        self.patch_size = int(IMAGE_SIZE // np.sqrt(in_channels))

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 3, stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, 4, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, 6, stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, 3, stride=2, drop_rate=drop_rate)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, drop_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def patch_generator(self, image):
        patches = torch.nn.functional.unfold(image, self.patch_size, stride=self.patch_size)
        # Reshape the patches to match the desired output shape [B, C, H, W]
        patches = patches.view(image.shape[0], self.in_channels, self.patch_size, self.patch_size)
        return patches

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


"""import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=True, fine_tune=True, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model"""
