import numpy as np
import torch.nn as nn
from torchvision.models import (
    densenet121, densenet161,
    resnet18, resnet50, resnet101,
    DenseNet121_Weights, DenseNet161_Weights,
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
)
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from datasets import IMAGE_SIZE


class PatchNet(nn.Module):
    resnet_models = ['resnet18', 'resnet50', 'resnet101']
    densenet_models = ['densenet121', 'densenet161']

    def __init__(self, model_name, num_patches, num_classes=4):
        super(PatchNet, self).__init__()
        self.model_name = model_name
        self.num_patches = num_patches
        self.patch_size = int(IMAGE_SIZE // np.sqrt(num_patches))
        # Load pretrained Models
        if model_name == 'resnet18':
            self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == 'resnet50':
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet101':
            self.base_model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif model_name == 'efficientnet-b0':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        elif model_name == 'efficientnet-b4':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b4')
        elif model_name == 'efficientnet-b5':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b5')
        elif model_name == 'efficientnet-b6':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b6')
        elif model_name == 'efficientnet-b7':
            self.base_model = EfficientNet.from_pretrained('efficientnet-b7')
        elif model_name == 'densenet121':
            self.base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        elif model_name == 'densenet161':
            self.base_model = densenet161(weights=DenseNet161_Weights.DEFAULT)
        else:
            raise NameError('No Model Found')

        # Disable gradient computation for the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Calculate the output feature size of the base model
        if model_name in self.resnet_models:
            base_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name in self.densenet_models:
            base_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        else:
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
        if self.model_name in self.resnet_models + self.densenet_models:
            patch_features = self.base_model(patches)
        else:
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
