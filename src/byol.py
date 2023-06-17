import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    densenet201, resnet50, DenseNet201_Weights, ResNet50_Weights,
)
from byol_pytorch import BYOL
from efficientnet_pytorch import EfficientNet

from MRIBTCPB.src.datasets import IMAGE_SIZE



class ByolNet:
    def __init__(self, model_name, augment_fn, augment_fn2):
        super().__init__()
        if model_name == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.name = 'ResNet'
        elif model_name == 'efficientnet-b4':
            self.model = EfficientNet.from_pretrained('efficientnet-b4')
            self.model.name = 'EfficientNet'
        elif model_name == 'densenet201':
            self.model = densenet201(weights=DenseNet201_Weights.DEFAULT)
        else:
            raise NameError('No Model Found')

        self.learner = BYOL(
            self.model,
            image_size = IMAGE_SIZE,
            hidden_layer = 'avgpool',
            moving_average_decay = 0.99,
            augment_fn = augment_fn,
            augment_fn2 = augment_fn2,
        )

    def train_byol(self, device, train_loader, epochs=20):
        opt = torch.optim.Adam(self.learner.parameters(), lr=3e-4)
        for _ in range(epochs):
            for _, (data, _) in enumerate(train_loader):
                data = data.to(device)
                loss = self.learner(data)
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.learner.update_moving_average()


class ClassifierByolNet(nn.Module):
    def __init__(self, base_encoder, num_classes, num_patches):
        super(ClassifierByolNet, self).__init__()
        self.num_patches = num_patches
        self.patch_size = int(IMAGE_SIZE // np.sqrt(num_patches))
        self.features = nn.Sequential(*list(base_encoder.children())[:-1])
        for param in self.features.parameters():
                param.requires_grad = False

        if base_encoder.name == 'ResNet':
            in_channels = base_encoder.fc.in_features
        elif base_encoder.name == 'DenseNet':
            in_channels = base_encoder.classifier.in_features
        else:  # EfficientNet
            in_channels = base_encoder._fc.in_features

        self.linear_classifier = nn.Sequential(
            nn.BatchNorm1d(in_channels * self.num_patches, affine=False),
            nn.Linear(in_channels * self.num_patches, num_classes),
        )
    
    def forward(self, x):
        batch_size = x.size(0)

        # Reshape the input to (batch_size * num_patches, 3, patch_size, patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)

        # Pass the patches through the base model
        patch_features = self.base_encoder(patches)
        patch_features = patch_features.view(batch_size, self.num_patches, -1)

        # Reshape patch_features to (batch_size, -1)
        patch_features = patch_features.view(batch_size, -1)
        return self.linear_classifier(patch_features)
