import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
                }, f"MRIBTCPB/outputs/model_{model_name}_{aug}.pth")

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
