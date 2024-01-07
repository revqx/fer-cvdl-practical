import os
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from tqdm import tqdm

from dataset import get_dataset
from model import get_model
from preprocessing import select_transform


def train_model(config: dict):
    path = os.getenv('MODEL_SAVE_PATH')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} not found. Will not be able to save model.")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the preprocessing
    preprocessing = select_transform(config['preprocessing'])

    dataset = get_dataset(config["train_data"], preprocessing=preprocessing, black_and_white=config["black_and_white"])
    train_loader, val_loader = train_val_dataloaders(dataset, config)

    # Model selection
    model = get_model(config["model_name"])
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train and evaluate the model
    training_loop(model, train_loader, val_loader, criterion, optimizer, config)

    # Save model and transforms
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    wandb_id = wandb.run.id
    path = os.path.join(path, f"{config['model_name']}-{timestamp}-{wandb_id}.pth")
    torch.save({'model': model.state_dict(), 'preprocessing': preprocessing}, path)
    print(f"Saved the model to {path}.")
    return model


def training_loop(model, train_loader, val_loader, criterion, optimizer, config):
    # Training loop
    for epoch in range(config["epochs"]):
        # Training and validation phases
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Use tqdm for progress bar
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} {phase}")

            # Iterate over data
            for inputs, labels in progress_bar:
                inputs = inputs.to(config["device"])
                labels = labels.to(config["device"])

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.2f}"})

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f"{phase} loss: {epoch_loss}, acc: {epoch_acc}")

            # Log metrics to wandb
            wandb.log({f"{phase}_loss": epoch_loss, 'epoch': epoch})
            wandb.log({f"{phase}_acc": epoch_acc, 'epoch': epoch})


def train_val_dataloaders(dataset, config):
    train_data, val_data = train_test_split(dataset, test_size=config["validation_split"])

    sampler = None
    if config.get("sampler") == "uniform":
        # count labels and create sampler
        y = [label for _, label in train_data]
        class_counts = np.bincount(y)
        class_weights = 1. / class_counts
        weights = class_weights[y]
        # Create the sampler
        sampler = WeightedRandomSampler(weights, len(weights))

    # Create the dataloaders
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader