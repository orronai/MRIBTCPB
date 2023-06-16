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


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.linear_classifier = nn.Sequential(
            nn.BatchNorm1d(in_channels, affine=False),
            nn.Linear(in_channels, num_classes),
        )
    
    def forward(self, x):
        return self.linear_classifier(x)