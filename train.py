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
from augment import select_augmentations


def train_model(config: dict):
    model_save_path = os.getenv('MODEL_SAVE_PATH')
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Directory {model_save_path} not found. Will not be able to save model.")

    # Set device
    if (config["device"] == "cuda") and (not torch.cuda.is_available()):
        config["device"] = "cpu"
        print("CUDA not available. Using CPU instead.")
    device = torch.device(config["device"])

    preprocessing = select_preprocessing(config['preprocessing'])
    augmentations = select_augmentations(config['augmentations'])

    dataset = get_dataset(config["train_data"])
    train_loader, val_loader = train_val_dataloaders(dataset, preprocessing, augmentations,
                                                     config["validation_split"], config["batch_size"],
                                                     config["sampler"], config["class_weight_adjustments"])

    # Model selection
    model = get_model(config["model_name"])
    model.to(device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer selection with error handling
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Invalid optimizer selection: '{config['optimizer']}'. Choose 'Adam' or 'SGD'.")

    # Scheduler selection with error handling
    if config["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config["ReduceLROnPlateau_patience"], verbose=True)
    elif config["scheduler"] == "InverseTimeDecay":
        lambda_func = lambda epoch: 1 / (1 + config["InverseTimeDecay_decay_rate"] * epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    else:
        raise ValueError(
            f"Invalid scheduler selection: '{config['scheduler']}'. Choose 'ReduceLROnPlateau' or 'InverseTimeDecay'.")

    # Define model path
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    wandb_id = wandb.run.id
    model_save_path = os.path.join(model_save_path, f"{config['model_name']}-{timestamp}-{wandb_id}.pth")

    # Train and evaluate the model
    training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config)

    # Save model and transforms
    torch.save({'model': model.state_dict(), 'preprocessing': preprocessing}, model_save_path)
    print(f"Saved the model to {model_save_path}.")


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
            epoch_acc = float(running_corrects) / len(data_loader.dataset)
            metrics[f'{phase}_loss'] = epoch_loss
            metrics[f'{phase}_acc'] = epoch_acc

            print(f"{phase} loss: {epoch_loss}, acc: {epoch_acc}")

        scheduler.step(metrics['val_loss'])

        # Log all metrics for the epoch at once
        metrics['epoch'] = epoch
        wandb.log(metrics)


def train_val_dataloaders(dataset, preprocessing, augmentations, validation_split, batch_size, sampler=None,
                          class_weight_adjustments=None):
    img_paths = [path for path, _ in dataset]
    labels = [label for _, label in dataset]

    if sampler == "uniform":
        # Calculate class weights for balancing
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts

        if class_weight_adjustments is not None:
            # Ensure the adjustments array has the same length as the number of classes
            num_classes = len(class_weights)
            if len(class_weight_adjustments) != num_classes:
                raise ValueError(f"Invalid class_weight_adjustments: {class_weight_adjustments}. It should contain "
                                 f"{num_classes} elements in the same order as LABEL_TO_STR is defined in utils.py.")

            # Apply the weight adjustments
            class_weights *= class_weight_adjustments

        weights = class_weights[labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        loader = DataLoader(dataset, batch_size=len(dataset), sampler=sampler)

        img_paths, labels = next(iter(loader))

    # Apply stratified train-test split on the original dataset
    train_data, val_data, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=validation_split, stratify=labels)

    # Create augmented dataset instances for training and validation
    train_dataset = DatasetWrapper(train_data, train_labels, preprocessing, augmentations)
    # No augmentations for validation
    val_dataset = DatasetWrapper(val_data, val_labels, preprocessing)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

