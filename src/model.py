import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from datasets import IMAGE_SIZE


# define bottleneck for resnet model. i.e. one block of resnet 50
"""class Bottleneck(nn.Module):
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
        return out"""


# define resnet model
class Resnet50Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(Resnet50Model, self).__init__()
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.in_planes = 64
        self.in_channels = in_channels
        self.patch_size = int(IMAGE_SIZE // np.sqrt(in_channels))

        self.conv1 = self.increase_channels(model.conv1, in_channels)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer

    def patch_generator(self, image):
        patches = torch.nn.functional.unfold(image, self.patch_size, stride=self.patch_size)
        # Reshape the patches to match the desired output shape [B, C, H, W]
        patches = patches.view(image.shape[0], self.in_channels, self.patch_size, self.patch_size)
        return patches

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def increase_channels(self, m, num_channels=None, copy_weights=0):
        """
        takes as input a Conv2d layer and returns the a Conv2d layer with `num_channels` input channels
        and all the previous weights copied into the new layer.
        """
        # number of input channels the new module should have
        with torch.no_grad():
            new_in_channels = num_channels if num_channels is not None else m.in_channels + 1
            
            bias = False if m.bias is None else True
            
            # Creating new Conv2d layer
            new_m = nn.Conv2d(in_channels=new_in_channels, 
                            out_channels=m.out_channels, 
                            kernel_size=m.kernel_size, 
                            stride=m.stride, 
                            padding=m.padding,
                            bias=bias,
            )
            
            # Copying the weights from the old to the new layer
            new_m.weight[:, :m.in_channels, :, :] = m.weight.clone()
            
            #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            for i in range(new_in_channels - m.in_channels):
                channel = m.in_channels + i
                new_m.weight[:, channel:channel+1, :, :] = m.weight[:, copy_weights:copy_weights + 1, : :].clone()
            new_m.weight = nn.Parameter(new_m.weight)

        return new_m


class ResPatchNet(nn.Module):
    def __init__(self, num_patches, num_classes=4):
        super(ResPatchNet, self).__init__()
        self.num_patches = num_patches
        self.patch_size = int(IMAGE_SIZE // np.sqrt(num_patches))
        # Load pretrained EfficientNet model
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Disable gradient computation for the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Calculate the output feature size of the base model
        base_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Identity()
        self.linear = nn.Linear(base_features * num_patches, base_features)

        # Define classifier layer
        self.classifier = nn.Linear(base_features, num_classes)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # Reshape the input to (batch_size * num_patches, 3, patch_size, patch_size)
        patches = inputs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)

        # Pass the patches through the base model
        patch_features = self.base_model(patches)
        patch_features = patch_features.view(batch_size, self.num_patches, -1)

        # Reshape patch_features to (batch_size, -1)
        patch_features = patch_features.view(batch_size, -1)

        # Pass the patch features through the linear layer
        linear_output = F.relu(self.linear(patch_features))

        # Pass the linear output through the classifier
        output = self.classifier(linear_output)

        return output


class EfficientPatchNet(nn.Module):
    def __init__(self, num_patches, num_classes=4):
        super(EfficientPatchNet, self).__init__()
        self.num_patches = num_patches
        self.patch_size = int(IMAGE_SIZE // np.sqrt(num_patches))
        # Load pretrained EfficientNet model
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        # Disable gradient computation for the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Calculate the output feature size of the base model
        base_features = self.base_model._fc.in_features

        # Define linear layer after concatenation of patch features
        self.linear = nn.Linear(base_features * num_patches, base_features)

        # Define classifier layer
        self.classifier = nn.Linear(base_features, num_classes)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # Reshape the input to (batch_size * num_patches, 3, patch_size, patch_size)
        patches = inputs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)

        # Pass the patches through the base model
        patch_features = self.base_model.extract_features(patches)
        patch_features = self.base_model._avg_pooling(patch_features)
        patch_features = patch_features.view(batch_size, self.num_patches, -1)

        # Reshape patch_features to (batch_size, -1)
        patch_features = patch_features.view(batch_size, -1)

        # Pass the patch features through the linear layer
        linear_output = F.relu(self.linear(patch_features))

        # Pass the linear output through the classifier
        output = self.classifier(linear_output)

        return output



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
