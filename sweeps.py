import numpy as np  # needed for custom train method
import torch  # needed for device and custom train method
import torch.nn.functional as F  # needed for custom train method
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler  # needed for custom train method

from dataset import get_dataset  # needed for custom train method
from model import get_model  # needed for custom train method
from preprocessing import select_preprocessing  # needed for custom train method


def get_sweep_config(metric="loss", goal="minimize", method="random",
                     custom_model=False, early_terminate=None):
    sweep_config = {
        "method": method  # to be specified by user
    }

    metric = {
        "name": metric,  # to be specified by user
        "goal": goal  # to be specified by user
    }
    sweep_config["metric"] = metric

    # parameters to sweep over (dropout not possible atm because models need custom input for dropout)
    parameters_dict = {
        "optimizer": {
            "values": ['sgd', 'adam']  # options: adam, sgd
        },
        "dataset": {
            "values": ["RAF-DB", "AffectNet"]  # options: AffectNet, RAF-DB
        },
        "batch_size": {
            "values": [16, 24, 32, 64, 128]  # defined here since log distribution causes bad comparability
        },
        "model_name": {
            "values": ["model_name"]
        }  # options: EmotionModel_2, CustomEmotionModel_3, LeNet, ResNet18
    }
    sweep_config["parameters"] = parameters_dict

    if sweep_config["method"] == "grid":
        parameters_dict.update({
            'epochs': {
                'value': 1}
        })
    else:
        parameters_dict.update({
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                'distribution': 'uniform',
                'min': 0.0001,
                'max': 0.1
            },
            'epochs': {
                'value': 3  # adjust to your liking (3 gives more accurate results than 1)
            }
        })
    sweep_config["parameters"] = parameters_dict

    if early_terminate is not None:
        sweep_config["early_terminate"] = {"type": "hyperband", "min_iter": 5, "eta": 3}

    return sweep_config


def train_sweep():
    config = get_sweep_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with wandb.init(config=config):
        config = wandb.config

        # get dataset and dataloaders (val_loader not needed for now)
        loader = get_sweep_loader(config)
        # get model
        model = get_model(config["model_name"])
        model.to(device)

        optimizer = get_optimizer_sweep(config["optimizer"], model, config.learning_rate)

        for epoch in range(config["epochs"]):
            avg_loss = train_epoch(model, loader, optimizer, device)
            wandb.log({"loss": avg_loss, "epoch": epoch})


def train_epoch(model, loader, optimizer, device):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        output = model(data)
        log_output = F.log_softmax(output, dim=1)  # Apply log-softmax to the output
        loss = F.nll_loss(log_output, target)  # Compute the loss using the log-softmax output
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)


def get_optimizer_sweep(optimizer, model, learning_rate):
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported.")

    return optimizer


def get_sweep_loader(config):
    preprocessing = select_preprocessing("StandardizeRGB()")
    dataset = get_dataset(config["dataset"], preprocessing=preprocessing, black_and_white=False)

    y = [label for _, label in dataset]
    class_counts = np.bincount(y)
    class_weights = 1. / class_counts
    weights = class_weights[y]
    # Create the sampler
    sampler = WeightedRandomSampler(weights, len(weights))

    # Create the dataloaders
    loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)

    return loader
