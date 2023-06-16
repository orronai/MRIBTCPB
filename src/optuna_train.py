import optuna
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from MRIBTCPB.src.datasets import get_datasets, get_data_loaders
from MRIBTCPB.src.model import PatchNet
from MRIBTCPB.src.train import aug_list


def define_model(trial, model_name):
    num_patches = trial.suggest_categorical('num_patches', [4, 16, 49])
    return PatchNet(model_name=model_name, num_patches=num_patches, num_classes=4)


def objective(trial, model_name):
    # Generate the model.
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(trial, model_name).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical('scheduler', ["StepLR", "CosineAnnealingLR"])
    scheduler = StepLR(optimizer, 10, 0.2) if scheduler_name == "StepLR" else CosineAnnealingLR(optimizer, 50)

    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    # Get dataset
    dataset_train, dataset_valid, dataset_test, _ = get_datasets()
    train_loader, valid_loader, _ = get_data_loaders(
        dataset_train, dataset_valid, dataset_test, batch_size,
    )

    epochs = 15
    n_train_examples = batch_size * 40
    n_valid_examples = batch_size * 20

    # Training of the model.
    for epoch in tqdm(range(epochs), total=epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            data = aug_list(data).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
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

                output = model(data)
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

def run_experiments(model_name: str):
    # now we can run the experiment
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name='patch-mri-classification', direction='maximize', sampler=sampler,
    )
    study.optimize(
        lambda trial: objective(trial, model_name=model_name), n_trials=50, timeout=3600,
    )

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return trial.params
