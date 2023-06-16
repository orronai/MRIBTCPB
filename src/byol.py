import torch
import torch.nn as nn
from byol_pytorch import BYOL

from MRIBTCPB.src.datasets import IMAGE_SIZE



class ByolNet:
    def __init__(self, model, augment_fn, augment_fn2):
        super().__init__()
        self.learner = BYOL(
            model,
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
    def __init__(self, base_encoder, num_classes):
        super(ClassifierByolNet, self).__init__()
        self.features = nn.Sequential(*list(base_encoder.children())[:-1])
        for param in self.features.parameters():
                param.requires_grad = False

        if base_encoder.name == 'ResNet':
            in_channels = self.base_model.fc.in_features
        elif base_encoder.name == 'DenseNet':
            in_channels = self.base_model.classifier.in_features
        else:  # EfficientNet
            in_channels = self.base_model._fc.in_features

        self.linear_classifier = nn.Sequential(
            nn.BatchNorm1d(in_channels, affine=False),
            nn.Linear(in_channels, num_classes),
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
