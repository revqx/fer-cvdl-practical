import os
from datetime import datetime

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from dataset import get_dataset, DatasetWrapper
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

    dataset = get_dataset(config["train_data"])
    train_loader, val_loader = train_val_dataloaders(dataset, preprocessing,
                                                     config["validation_split"], config["batch_size"],
                                                     config["sampler"], config["weak_class_adjust"])

    # Model selection
    model = get_model(config["model_name"])
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

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


def train_val_dataloaders(dataset, preprocessing, validation_split, batch_size, sampler=None, weak_class_adjust=False):
    img_paths = [path for path, _ in dataset]
    labels = [label for _, label in dataset]

    if sampler == "uniform":
        # Calculate class weights for balancing
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts

        if weak_class_adjust:
            # Adjust weights for specific classes
            fear_class, disgust_class = 2, 1
            class_weights[fear_class] *= 1
            class_weights[disgust_class] *= 1

        weights = class_weights[labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        loader = DataLoader(dataset, batch_size=len(dataset), sampler=sampler)

        img_paths, labels = next(iter(loader))

    # Apply stratified train-test split on the original dataset
    train_data, val_data, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=validation_split, stratify=labels)

    # Create augmented dataset instances for training and validation
    train_dataset = DatasetWrapper(train_data, train_labels, preprocessing)
    val_dataset = DatasetWrapper(val_data, val_labels, preprocessing, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
