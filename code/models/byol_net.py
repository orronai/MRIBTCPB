import numpy as np
import torch
import torch.quantization
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from MRIBTCPB.code.models.byol import BYOL
from MRIBTCPB.code.utils.datasets import IMAGE_SIZE


class ByolNet:
    def __init__(self, model_name, augment_fn, augment_fn2):
        super().__init__()
        if model_name == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.name = 'ResNet'
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

    def train_byol(self, device, train_loader, epochs=20, use_amp=True):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        opt = torch.optim.Adam(self.learner.parameters(), lr=3e-4)
        scheduler = StepLR(opt, 20, 0.1, verbose=True)
        self.model.to(device)
        self.learner.to(device)
        for epoch in range(epochs):
            counter = 0
            for _, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
                counter += 1
                with torch.cuda.amp.autocast(enabled=use_amp):
                    data = data.to(device)
                    loss = self.learner(data)
                opt.zero_grad(set_to_none=True)
                train_running_loss += loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                self.learner.update_moving_average()
            epoch_loss = train_running_loss / counter
            print(f'Epoch {epoch + 1} loss: {epoch_loss:.4f}')
            scheduler.step()


class ClassifierByolNet(nn.Module):
    def __init__(self, base_encoder, num_classes, num_patches, fine_tune=False):
        super(ClassifierByolNet, self).__init__()
        self.num_patches = num_patches
        self.patch_size = int(IMAGE_SIZE // np.sqrt(num_patches))
        self.base_encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        if fine_tune:
            for param in self.base_encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.base_encoder.parameters():
                param.requires_grad = False

        in_channels = base_encoder.fc.in_features

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
