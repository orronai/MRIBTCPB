import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from MRIBTCPB.src.byol import ByolNet, Classifier
from MRIBTCPB.src.datasets import CLASSES, get_datasets, get_data_loaders
from MRIBTCPB.src.model import PatchNet
from MRIBTCPB.src.utils import calculate_accuracy, save_model, save_plots


aug_list = AugmentationSequential(
    K.RandomAffine(5, [0.05, 0.05], [0.75, 1.05], p=0.1),
    K.RandomPerspective(0.1, p=0.1),
    K.RandomHorizontalFlip(p=0.1),
    K.RandomVerticalFlip(p=0.1),
    K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.0001, 0.5), p=0.4),
    K.RandomSharpness(p=0.1),
    same_on_batch=False,
)

# Training function.
def train(model, train_loader, optimizer, criterion, device, augmentation):
    model.train()
    print('\nTraining')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = aug_list(image).to(device) if augmentation else image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
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

# Train Model.
def train_model(model_name, augmentation, optimizer, scheduler, batch_size, lr, num_patches, fine_tune):
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_test, dataset_classes = get_datasets()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset_train, dataset_valid, dataset_test, batch_size,
    )
    epochs = 30
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    model = PatchNet(
        model_name=model_name, num_patches=num_patches,
        num_classes=4, fine_tune=fine_tune,
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)
    # Scheduler
    scheduler = StepLR(optimizer, 10, 0.1, verbose=True) if scheduler == "StepLR" else CosineAnnealingLR(optimizer, epochs, verbose=True)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader,optimizer, criterion, device, augmentation,
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device,
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)
        time.sleep(0.5)
        scheduler.step()

    # Save the trained model weights.
    save_model(epochs, model, model_name, optimizer, criterion, augmentation)
    # Save the loss and accuracy plots.
    save_plots(model_name, train_acc, valid_acc, train_loss, valid_loss, augmentation)
    print('TRAINING COMPLETE')

    test_accuracy, confusion_matrix = calculate_accuracy(model, test_loader, device)
    print("test accuracy: {:.3f}%".format(test_accuracy))

    matplotlib.style.use('default')
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=CLASSES,
    )
    cm_display.plot(cmap=plt.cm.Blues, values_format='.3f')
    plt.show()


# Training function.
def train_epoch_byol(model, classifier, train_loader, optimizer, criterion, device):
    classifier.train()
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
        outputs = model(image)
        outputs = classifier(outputs)
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
def validate_byol(model, classifier, valid_loader, criterion, device):
    classifier.eval()
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
            outputs = model(image)
            outputs = classifier(outputs)
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


def train_byol(model, batch_size, optimizer, scheduler, lr):
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_test, dataset_classes = get_datasets()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset_train, dataset_valid, dataset_test, batch_size,
    )
    epochs = 30
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    augment_fn = nn.Sequential(
        K.RandomAffine(5, [0.2, 0.2], [0.7, 1], p=0.2),
        K.RandomVerticalFlip(p=0.5),
        K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.001, 1), p=0.5),
        K.RandomSharpness(p=0.2),
        K.RandomBrightness(p=0.3, brightness=(0.95, 1.05)),
    )

    augment_fn2 = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.GaussianBlur2d((3, 3), (1.5, 1.5)),
        K.RandomPerspective(0.3, p=0.5),
        K.RandomRotation(7, p=0.3),
        K.RandomContrast(p=0.2, contrast=(0.95, 1.05)),
    )
    
    Byol = ByolNet(model, augment_fn=augment_fn, augment_fn2=augment_fn2)
    Byol.train_byol(device, train_loader, epochs=30)
    model.eval()
    classifier = Classifier(model.fc.in_features, num_classes=4)
    classifier = classifier.to(device)
    # Optimizer.
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)
    # Scheduler
    scheduler = StepLR(optimizer, 10, 0.2, verbose=True) if scheduler == "StepLR" else CosineAnnealingLR(optimizer, epochs, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train_epoch_byol(
            model, classifier, train_loader, optimizer, criterion, device,
        )
        valid_epoch_loss, valid_epoch_acc = validate_byol(
            model, classifier, valid_loader, criterion, device,
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)
        time.sleep(0.5)

    # Save the trained model weights.
    save_model(epochs, model, 'byol', optimizer, criterion, True)
    # Save the loss and accuracy plots.
    save_plots('byol', train_acc, valid_acc, train_loss, valid_loss, True)
    print('TRAINING COMPLETE')

    test_accuracy, confusion_matrix = calculate_accuracy(model, test_loader, device)
    print("test accuracy: {:.3f}%".format(test_accuracy))

    matplotlib.style.use('default')
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=CLASSES,
    )
    cm_display.plot(cmap=plt.cm.Blues, values_format='.3f')
    plt.show()
