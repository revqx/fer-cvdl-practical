import numpy as np  # needed for custom train method
import torch  # needed for device and custom train method
import torch.nn.functional as F  # needed for custom train method
import torch.nn as nn  # needed for dynamic model
import wandb
import torch.optim as optim #optimizer
import torch #needed for device and custom train method
import torch.nn.functional as F #needed for custom train method
from torch.utils.data import DataLoader, WeightedRandomSampler#needed for custom train method
import numpy as np #needed for custom train method
from sklearn.model_selection import train_test_split    #needed for custom train method

from dataset import get_dataset  # needed for custom train method
from model import get_model, _create_conv_block, _create_conv_block_2  # needed for custom train method
from preprocessing import select_preprocessing  # needed for custom train method


def get_sweep_config(metric="loss", goal="minimize", method="random",
                     custom_model=True, early_terminate=None):
    sweep_config = {
        "method": method #to be specified by user
        }
    
    metric = {
        "name": metric, #to be specified by user
        "goal": goal #to be specified by user
        }
    sweep_config["metric"] = metric

    #parameters to sweep over (dropout not possible atm because models need custom input for dropout)
    parameters_dict = {
        "optimizer": {
            "values": ['sgd']  # options: adam, sgd
        },
        "dataset": {
            "values": ["RAF-DB"]  # options: AffectNet, RAF-DB
        },
        "batch_size": {
            "values": [16, 24, 32]  # defined here since log distribution causes bad comparability
        }
    }
    sweep_config["parameters"] = parameters_dict

    if custom_model:
        parameters_dict.update({
            "dropout": {
                "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "layer_count": {
                "values": [0, 1, 2, 3, 4, 5]
            }
        })
        sweep_config["parameters"] = parameters_dict
    else:
        parameters_dict.update({
        "model_name": {
            "values": ["model_name"]
        }  # options: EmotionModel_2, CustomEmotionModel_3, LeNet, ResNet18
        })
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
                'value': 3 # adjust to your liking (3 gives more accurate results than 1)
            }
        })
    sweep_config["parameters"] = parameters_dict

    if early_terminate is not None:
        sweep_config["early_terminate"] = {"type": "hyperband", "min_iter": 5, "eta": 3}

    return sweep_config


def train_sweep(custom_model=True):
    config = get_sweep_config(custom_model=custom_model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with wandb.init(config=config):
        config = wandb.config

        # get dataset and dataloaders (val_loader not needed for now)
        loader = get_sweep_loader(config)

        # get model
        if custom_model:
            model = get_dynamic_model(config)
        else:
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


class DynamicModel(nn.Module):
    def __init__(self, hidden_layers, dropout, num_classes=6):
        super(DynamicModel, self).__init__()

        self.conv_block1 = _create_conv_block(3, 64)
        self.conv_block2 = _create_conv_block(64, 128)
        self.conv_block3 = _create_conv_block(128, 256)
        self.conv_block4 = _create_conv_block(256, 512, pool=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.hidden_layers = nn.ModuleList()

        if hidden_layers > 0:
            # first hidden layer has output size of last conv block
            self.hidden_layers.append(nn.Linear(512, 256))

            # dynamic number of hidden layers
            for _ in range(hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(256 // 2**_, 256 // 2**(_ + 1)))

            # define output layer depending on number of hidden layers
            self.output = nn.Linear(256 // 2**(hidden_layers - 1), num_classes)

            self.dropout = nn.Dropout(dropout)

        else:
            # define output layer in case of no hidden layers
            self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return x
    

def get_dynamic_model(config):
    model = DynamicModel(config["layer_count"], config["dropout"])
    return model
    