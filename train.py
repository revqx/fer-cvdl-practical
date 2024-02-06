import copy
import os
from datetime import datetime

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from augmentation import select_augmentations
from dataset import get_dataset, DatasetWrapper
from model import get_model
from preprocessing import select_preprocessing
from utils import load_model_and_preprocessing


def train_model(config: dict):
    """Train a model with the given config."""

    # Set up model saving
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Directory {model_save_path} not found. Will not be able to save model.")

    # Set device
    if (config["device"] == "cuda") and (not torch.cuda.is_available()):
        config["device"] = "cpu"
        print("CUDA not available. Using CPU instead.")
    device = torch.device(config["device"])

    # Preprocessing and augmentations
    preprocessing = select_preprocessing(config["preprocessing"])
    augmentations = select_augmentations(config["augmentations"])

    # Model selection
    if config["pretrained_model"] != "":
        # Overwrite preprocessing with the pretrained model"s preprocessing
        model_id, model, preprocessing = load_model_and_preprocessing(config["pretrained_model"])
        config["model_name"] = model_id if config["model_name"] is None else config["model_name"]
    elif config["model_name"] != "":
        model = get_model(config["model_name"], hidden_layers=config["DynamicModel_hidden_layers"],
                          dropout=config["DynamicModel_hidden_dropout"])
    else:
        raise ValueError("Either 'pretrained_model' or 'model_name' must be specified.")

    model.to(device)

    dataset = get_dataset(config["train_data"])
    train_loader, val_loader = train_val_dataloaders(dataset, preprocessing, augmentations,
                                                     config["validation_split"], config["batch_size"],
                                                     config["sampler"], config["class_weight_adjustments"])

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
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=config["ReduceLROnPlateau_patience"])
    elif config["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer, 1, gamma=config["StepLR_decay_rate"])
    else:
        raise ValueError(
            f"Invalid scheduler selection: '{config['scheduler']}'. Choose 'ReduceLROnPlateau' or 'StepLR'.")

    # Define model path
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    wandb_id = wandb.run.id
    model_save_path = os.path.join(model_save_path, f"{config['model_name']}-{timestamp}-{wandb_id}.pth")

    # Train and evaluate the model
    model = training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config)

    # Save model and transforms
    torch.save({"model": model.state_dict(), "preprocessing": preprocessing}, model_save_path)
    print(f"Saved the model to {model_save_path}.")

    return model


def training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    phases = ["train"]
    if val_loader is not None:
        phases.append("val")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    patience = config["early_stopping_patience"]

    for epoch in range(config["max_epochs"]):
        wandb.log({"scheduler": [group['lr'] for group in optimizer.param_groups][0]}, step=epoch + 1)

        metrics = {
            "train_loss": 0.0,
            "train_acc": 0.0,
            "val_loss": 0.0,
            "val_acc": 0.0,
        }

        for phase in phases:
            if phase == "train":
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{config['max_epochs']} {phase}")

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

                torch.cuda.empty_cache()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix({"loss": f"{loss.item():.2f}"})

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = float(running_corrects) / len(data_loader.dataset)
            metrics[f"{phase}_loss"] = epoch_loss
            metrics[f"{phase}_acc"] = epoch_acc

            print(f"{phase} loss: {epoch_loss}")
            print(f"{phase} acc: {epoch_acc}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            elif phase == 'val':
                epochs_no_improve += 1

        scheduler.step(metrics["val_loss"] if config["scheduler"] == "ReduceLROnPlateau" else None)
        wandb.log(metrics, step=epoch + 1)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    model.load_state_dict(best_model_wts)
    return model


def get_weighted_sampler(labels, class_weight_adjustments=None):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts

    if class_weight_adjustments is not None:
        if len(class_weight_adjustments) != len(class_weights):
            raise ValueError(f"Invalid class_weight_adjustments. Expected length: {len(class_weights)}")

        class_weights *= class_weight_adjustments

    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def train_val_dataloaders(dataset, preprocessing, augmentations, validation_split, batch_size, sampler=None,
                          class_weight_adjustments=None):
    if validation_split < 0 or validation_split >= 1:
        raise ValueError(f"Invalid validation_split: {validation_split}. It should be in the range [0, 1).")

    images = [img for img, _ in dataset]
    labels = [label for _, label in dataset]

    # Stratified train-test split
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=validation_split, stratify=labels)

    print(f"Prior train distribution (Σ: {len(train_images)}): {np.bincount(train_labels)}")
    print(f"Prior val distribution (Σ: {len(val_images)}): {np.bincount(val_labels)}")

    print("Applying augmentations to the train set...") if augmentations != [] else None

    # Create datasets
    train_dataset = DatasetWrapper(train_images, train_labels, preprocessing, augmentations)
    val_dataset = DatasetWrapper(val_images, val_labels, preprocessing)

    train_sampler = val_sampler = None
    if sampler == "uniform":
        augmented_train_labels = [label for _, label in train_dataset]
        print("Oversampling train and val datasets to balance class distribution...")
        train_sampler = get_weighted_sampler(augmented_train_labels, class_weight_adjustments)
        val_sampler = get_weighted_sampler(val_labels, class_weight_adjustments)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=val_sampler is None)

    print(f"Sample of updated train distribution (Σ: {len(train_sampler)}): "
          f"{np.bincount(np.concatenate([labels.numpy() for _, labels in train_loader]))}") \
        if augmentations != [] or sampler == "uniform" else None
    print(f"Sample of updated val distribution (Σ: {len(val_sampler)}): "
          f"{np.bincount(np.concatenate([labels.numpy() for _, labels in val_loader]))}") \
        if augmentations != [] or sampler == "uniform" else None

    return train_loader, val_loader
