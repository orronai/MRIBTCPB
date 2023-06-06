from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Required constants.
TRAIN_DIR = '../input/training'
TEST_DIR = '../input/test'
VALID_DIR = '../input/validation'
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 32 
NUM_WORKERS = 4 # Number of parallel processes for data preparation.


def get_transform(IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.17340052], std=[0.18875531]),
    ])
    return train_transform

def get_datasets():
    """
    Function to prepare the Datasets.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, transform=(get_transform(IMAGE_SIZE)),
    )
    dataset_valid = datasets.ImageFolder(
        VALID_DIR, transform=(get_transform(IMAGE_SIZE)),
    )
    dataset_test = datasets.ImageFolder(
        TEST_DIR, transform=(get_transform(IMAGE_SIZE)),
    )
    return dataset_train, dataset_valid, dataset_test, dataset_train.classes

def get_data_loaders(dataset_train, dataset_valid, dataset_test):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS,
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS,
    )
    return train_loader, valid_loader, test_loader
