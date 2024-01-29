import os
from datetime import datetime

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import get_dataset
from model import get_model
from preprocessing import select_preprocessing


def train_model(config: dict):
    path = os.getenv('MODEL_SAVE_PATH')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} not found. Will not be able to save model.")

    # Set device
    if (config["device"] == "cuda") and (not torch.cuda.is_available()):
        config["device"] = "cpu"
        print("CUDA not available. Using CPU instead.")
    device = torch.device(config["device"])

    # Define the preprocessing
    preprocessing = select_preprocessing(config['preprocessing'])

    dataset = get_dataset(config["train_data"], preprocessing=preprocessing, black_and_white=config["black_and_white"])
    train_loader, val_loader = train_val_dataloaders(dataset, config)

    # Model selection
    model = get_model(config["model_name"])
    model.to(device)

    # TODO: Add loss function selection
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # Adam with beta1=0.9 and beta2=0.999
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config["patience"], verbose=True)

    # Train and evaluate the model
    training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config)

    # Save model and transforms
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    wandb_id = wandb.run.id
    path = os.path.join(path, f"{config['model_name']}-{timestamp}-{wandb_id}.pth")
    torch.save({'model': model.state_dict(), 'preprocessing': preprocessing}, path)
    print(f"Saved the model to {path}.")
    return model


def training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    for epoch in range(config["epochs"]):
        # Dictionaries to store metrics for each phase
        metrics = {'train_loss': 0.0, 'train_acc': 0.0, 'val_loss': 0.0, 'val_acc': 0.0}

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} {phase}")

            for inputs, labels in progress_bar:
                inputs = inputs.to(config["device"])
                labels = labels.to(config["device"])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix({'loss': f"{loss.item():.2f}"})

            # Calculate and store metrics
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            metrics[f'{phase}_loss'] = epoch_loss
            metrics[f'{phase}_acc'] = epoch_acc

            print(f"{phase} loss: {epoch_loss}, acc: {epoch_acc}")

        scheduler.step(metrics['val_loss'])

        # Log all metrics for the epoch at once
        metrics['epoch'] = epoch
        wandb.log(metrics)


def train_val_dataloaders(dataset, config):
    if config["sampler"] == "uniform":
        labels = [label for _, label in dataset]
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        weights = class_weights[labels]

        if config["weak_class_adjust"]:
            # Increase the weight of the fear class
            fear_class = 2  # the class index should be 2
            weights[labels == fear_class] *= 1
            disgust_class = 1
            weights[labels == disgust_class] *= 1

        sampler = WeightedRandomSampler(weights, len(weights))

        loader = DataLoader(dataset, batch_size=len(dataset), sampler=sampler)

        # has to be turned back into a dataset to use train_test_split
        uniform_dataset, uniform_labels = next(iter(loader))

    else: # in case no input or other than uniform is given
        uniform_dataset, uniform_labels = dataset, [label for _, label in dataset]

    # redefine labels to base the stratification on so distribution is consistent
    train_data, val_data, train_labels, val_labels = train_test_split(uniform_dataset, uniform_labels, test_size=config["validation_split"], stratify=uniform_labels)

    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=config["batch_size"])
    val_loader = DataLoader(list(zip(val_data, val_labels)), batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader
