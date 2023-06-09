import optuna
import torch
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets import get_datasets, get_data_loaders
from model import Resnet50Model


def define_model(trial):
    p = trial.suggest_float("drop_rate", 0.1, 0.5)
    in_channels = trial.suggest_categorical('in_channels', [4, 16, 49])
    return Resnet50Model(in_channels=in_channels, num_classes=4, drop_rate=p)


def objective(trial):
    # Generate the model.
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(trial).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, 100)

    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    # Get dataset
    dataset_train, dataset_valid, dataset_test, _ = get_datasets()
    train_loader, valid_loader, _ = get_data_loaders(
        dataset_train, dataset_valid, dataset_test, batch_size,
    )

    epochs = 10
    n_train_examples = batch_size * 80
    n_valid_examples = batch_size * 25

    # Training of the model.
    for epoch in tqdm(range(epochs), total=epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model.patch_generator(data)
            output = model(output)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * batch_size >= n_valid_examples:
                    break

                data = data.to(device)
                target = target.to(device)

                output = model.patch_generator(data)
                output = model(output)
                _, preds = torch.max(output.data, 1)
                correct += (preds == target).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(accuracy, epoch)  
        scheduler.step()

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy
