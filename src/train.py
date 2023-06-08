import torch
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
from tqdm import tqdm


aug_list = AugmentationSequential(
    K.RandomAffine(5, [0.1, 0.1], [0.6, 1], p=0.2),
    K.RandomPerspective(0.5, p=0.1),
    K.RandomHorizontalFlip(p=0.1),
    K.RandomVerticalFlip(p=0.1),
    K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.001, 0.2), p=0.2),
    K.RandomSharpness(p=0.1),
    same_on_batch=False,
)

# Training function.
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    print('\nTraining')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = aug_list(image).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        image = model.patch_generator(image)
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, valid_loader, criterion, device):
    model.eval()
    print('\nValidation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            image = model.patch_generator(image)
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))
    return epoch_loss, epoch_acc
