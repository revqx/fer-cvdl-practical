def get_sweep_config(metric="val_loss", goal="minimize", method="bayes",
                     custom_model=False, early_terminate=False):
    sweep_config = {
        "method": method
    }

    metric = {
        "name": metric,
        "goal": goal
    }
    sweep_config["metric"] = metric

    # parameters to sweep over (dropout not possible atm because models need custom input for dropout)
    parameters_dict = {
        "optimizer": {
            "values": ["Adam"]  # options: Adam, SGD
        },
        "dataset": {
            "values": ["RAF-DB"]  # options: AffectNet, RAF-DB
        },
        "batch_size": {
            "values": [24, 32]  # defined here since log distribution causes bad comparability
        },
        "validation_split": {
            "values": [0.2]  # once sweeped to be set as constant
        },
        "weak_class_adjust": {
            "value": [1, 1, 1, 1, 1, 1]  # can be set to True and weights have to be adjusted in get_sweep_loader
        },
        "ReduceLROnPlateau_patience": {
            "values": [2, 4, 8]
        },
        "augmentations": {
            "values": [
                "HorizontalFlip, RandomRotation, RandomCrop, TrivialAugmentWide, TrivialAugmentWide",
                "HorizontalFlip, RandomRotation, RandomCrop, RandAugment, RandAugment",
                "HorizontalFlip, RandomRotation, RandomCrop, TrivialAugmentWide, RandAugment",
            ]
        }
    }

    sweep_config["parameters"] = parameters_dict

    if custom_model:
        parameters_dict.update({
            "model_name": {
                "values": ["DynamicModel"]
            },
            "DynamicModel_dropout": {
                "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "DynamicModel_hidden_layers": {
                "values": [0, 1, 2, 3, 4, 5]
            }
        })
        sweep_config["parameters"] = parameters_dict
    else:
        parameters_dict.update({
            "model_name": {
                "values": ["CustomEmotionModel3"]
            }  # options: EmotionModel_2, CustomEmotionModel3, LeNet, ResNet18
        })
        sweep_config["parameters"] = parameters_dict

    if sweep_config["method"] == "grid" or sweep_config["method"] == "bayes":
        parameters_dict.update({
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                "values": [0.001, 0.0001]
            },
            "epochs": {
                "values": [10, 15, 20]
            }
        })
    else:
        parameters_dict.update({
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.001
            },
            "epochs": {
                "values": [5]  # adjust to your liking (3 gives more accurate results than 1)
            }
        })
    sweep_config["parameters"] = parameters_dict

    if early_terminate:
        sweep_config["early_terminate"] = {"type": "hyperband", "min_iter": 5, "eta": 3}

    return sweep_config
