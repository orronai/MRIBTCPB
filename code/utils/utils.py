import torch
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from MRIBTCPB.code.utils.datasets import (
    CLASSES, IMAGE_SIZE, NUM_WORKERS, TEST_DIR, get_transform,
)

matplotlib.style.use('ggplot')

def save_model(epochs, model, model_name, optimizer, criterion, augmentation):
    """
    Function to save the trained model to disk.
    """
    aug = 'augmented' if augmentation else 'not-augmented'
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"MRIBTCPB/outputs/weights/model_{model_name}_{aug}.pth")

def save_plots(model_name, train_acc, valid_acc, train_loss, valid_loss, augmentation):
    """
    Function to save the loss and accuracy plots to disk.
    """
    aug = 'augmented' if augmentation else 'not-augmented'
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='Train Accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='Validataion Accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"MRIBTCPB/outputs/accuracy_{model_name}_{aug}.png")
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='Train Loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='Validataion Loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"MRIBTCPB/outputs/loss_{model_name}_{aug}.png")

# function to calculate accuracy of the model
def calculate_accuracy(model, dataloader, device):
    model.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([4, 4], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    row_sums = np.sum(confusion_matrix, axis=1)
    confusion_matrix = np.divide(confusion_matrix, row_sums[:, np.newaxis])

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

def plot_tsne_representation(classifier, n_points=500, n_components=2, perplexity=30.0):
    if n_components < 2 or n_components > 3:
        print("OH NO :(")
        raise SystemError("2 <= dim <= 3")

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_test = datasets.ImageFolder(
        TEST_DIR, transform=(get_transform(IMAGE_SIZE)),
    )

    test_loader = DataLoader(
        dataset_test, batch_size=n_points,
        shuffle=True, num_workers=NUM_WORKERS,
    )
    n_neighbors = len(CLASSES)

    with torch.no_grad():
        X, labels = next(iter(test_loader))
        latent_X = classifier.get_representation(X.to(device)).view(n_points, -1).data.cpu().numpy()
        labels = labels.data.cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    x_tsne = TSNE(n_components, perplexity=perplexity).fit_transform(latent_X)
    colors = cm.rainbow(np.linspace(0, 1, n_neighbors))

    if n_components == 2:
        ax = fig.add_subplot(1, 1, 1)
        for l in range(n_neighbors):
            ax.scatter(
                x_tsne[labels == l, 0], x_tsne[labels == l, 1], color=colors[l],
                marker='.', label=f'class={CLASSES[l]}',
            )
        ax.set_title("2D t-SNE of the embedded images")
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for l in range(n_neighbors):
            ax.scatter(
                x_tsne[labels == l, 0], x_tsne[labels == l, 1], x_tsne[labels == l, 2],
                color=colors[l], marker='.', label=f'class={CLASSES[l]}',
            )
        ax.set_title("3D t-SNE of the embedded images")

    ax.grid()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"MRIBTCPB/outputs/'t-SNE_BYOL-{n_components}D'.png")
