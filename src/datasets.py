from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Required constants.
TRAIN_DIR = '../input/training'
TEST_DIR = '../input/test'
VALID_DIR = '../input/validation'
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
NUM_WORKERS = 2 # Number of parallel processes for data preparation.
CLASSES = ('glioma', 'meningioma', 'notumor', 'pituitary')

def get_transform(IMAGE_SIZE, patch_input=False):
    if patch_input:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Grayscale(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return train_transform

def get_datasets(patch_input=False):
    """
    Function to prepare the Datasets.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, transform=(get_transform(IMAGE_SIZE, patch_input)),
    )
    dataset_valid = datasets.ImageFolder(
        VALID_DIR, transform=(get_transform(IMAGE_SIZE, patch_input)),
    )
    dataset_test = datasets.ImageFolder(
        TEST_DIR, transform=(get_transform(IMAGE_SIZE, patch_input)),
    )
    return dataset_train, dataset_valid, dataset_test, dataset_train.classes

def get_data_loaders(dataset_train, dataset_valid, dataset_test, batch_size=32):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=NUM_WORKERS,
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=True, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, 
        shuffle=False, num_workers=NUM_WORKERS,
    )
    return train_loader, valid_loader, test_loader
