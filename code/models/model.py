import numpy as np
import torch.nn as nn
from torchvision.models import (
    densenet201, resnet50, DenseNet201_Weights, ResNet50_Weights,
)
from efficientnet_pytorch import EfficientNet

from MRIBTCPB.code.utils.datasets import IMAGE_SIZE


class PatchNet(nn.Module):
    resnet_model = 'resnet50'
    efficient_model = 'efficientnet-b4'
    densenet_model = 'densenet201'

    def __init__(self, model_name, num_patches, num_classes=4, fine_tune=False):
        super(PatchNet, self).__init__()
        self.model_name = model_name
        self.num_patches = num_patches
        self.patch_size = int(IMAGE_SIZE // np.sqrt(num_patches))
        # Load pretrained Models
        if model_name == self.resnet_model:
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == self.efficient_model:
            self.base_model = EfficientNet.from_pretrained(self.efficient_model)
        elif model_name == self.densenet_model:
            self.base_model = densenet201(weights=DenseNet201_Weights.DEFAULT)
        else:
            raise NameError('No Model Found')

        if fine_tune:
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Calculate the output feature size of the base model
        if model_name == self.resnet_model:
            base_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == self.densenet_model:
            base_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        else:
            base_features = self.base_model._fc.in_features

        # Define linear layer after concatenation of patch features - classifier
        self.linear = nn.Linear(base_features * num_patches, num_classes)

        # Define classifier layer
        # self.classifier = nn.Linear(base_features, num_classes)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # Reshape the input to (batch_size * num_patches, 3, patch_size, patch_size)
        patches = inputs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)

        # Pass the patches through the base model
        if self.model_name == self.resnet_model or self.model_name == self.densenet_model:
            patch_features = self.base_model(patches)
        else:  # EfficientNet
            patch_features = self.base_model.extract_features(patches)
            patch_features = self.base_model._avg_pooling(patch_features)
        patch_features = patch_features.view(batch_size, self.num_patches, -1)

        # Reshape patch_features to (batch_size, -1)
        patch_features = patch_features.view(batch_size, -1)

        # Pass the patch features through the linear layer
        output = self.linear(patch_features)

        # Pass the linear output through the classifier
        # output = self.classifier(output)

        return output
