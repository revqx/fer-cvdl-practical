import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_model(model_name, **kwargs):
    if model_name not in MODELS:
        raise ValueError(f"Model not supported: {model_name}")

    return MODELS[model_name](**kwargs)


class LeNet(nn.Module):
    def __init__(self, num_classes=6, input_size=64, **kwargs):
        super(LeNet, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate size after convolutions and pooling
        def conv_output_size(in_size, kernel_size=5, stride=1, padding=0):
            return (in_size - kernel_size + 2 * padding) // stride + 1

        self.conv1_size = conv_output_size(input_size, 5, 1, 0)
        self.pool1_size = conv_output_size(self.conv1_size, 2, 2, 0)
        self.conv2_size = conv_output_size(self.pool1_size, 5, 1, 0)
        self.pool2_size = conv_output_size(self.conv2_size, 2, 2, 0)

        linear_input_size = 16 * self.pool2_size * self.pool2_size
        self.fc1 = nn.Linear(linear_input_size, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * self.pool2_size * self.pool2_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=6, input_size=64, **kwargs):
        super(ResNet18, self).__init__()
        self.input_size = input_size
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=6, input_size=64, **kwargs):
        super(ResNet50, self).__init__()
        self.input_size = input_size
        self.model = models.resnet50(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=6, input_size=64, **kwargs):
        super(MobileNetV2, self).__init__()
        self.input_size = input_size
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class EmotionModel(nn.Module):
    def __init__(self, num_classes=6, **kwargs):
        super(EmotionModel, self).__init__()

        self.conv_block1 = _create_conv_block(3, 64)
        self.conv_block2 = _create_conv_block(64, 128)
        self.conv_block3 = _create_conv_block(128, 256)
        self.conv_block4 = _create_conv_block(256, 512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def _create_conv_block(in_channels, out_channels, pool=True):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

    if pool:
        block.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return block


def _create_conv_block_2(in_channels, out_channels, pool=True):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    if pool:
        block.append(nn.MaxPool2d(kernel_size=3, stride=2))

    return block


def _create_conv_block_4(in_channels, out_channels, pool=True):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )

    if pool:
        block.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return block

class CustomEmotionModel3(nn.Module):
    def __init__(self, num_classes=6, **kwargs):
        super(CustomEmotionModel3, self).__init__()

        self.conv_block1 = _create_conv_block(3, 64)
        self.conv_block2 = _create_conv_block(64, 128)
        self.conv_block3 = _create_conv_block(128, 256)
        self.conv_block4 = _create_conv_block(256, 512, pool=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.2)  # Dropout to prevent overfitting
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CustomEmotionModel4(nn.Module):
    def __init__(self, num_classes=6, **kwargs):
        super(CustomEmotionModel4, self).__init__()

        self.conv_block1 = _create_conv_block(3, 64)
        self.conv_block2 = _create_conv_block_2(64, 128)
        self.conv_block3 = _create_conv_block_2(128, 256, pool=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CustomEmotionModel5(nn.Module):
    def __init__(self, num_classes=6, **kwargs):
        super(CustomEmotionModel5, self).__init__()

        self.conv_block1 = _create_conv_block(3, 64)
        self.conv_block2 = _create_conv_block_2(64, 128)
        self.conv_block3 = _create_conv_block(128, 256)
        self.conv_block4 = _create_conv_block_2(256, 128,
                                                pool=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Apron(nn.Module):
    def __init__(self, num_classes=6, **kwargs):
        super(Apron, self).__init__()

        self.conv_block1 = _create_conv_block_4(3, 32)
        self.conv_block2 = _create_conv_block_4(32, 64)
        self.conv_block3 = _create_conv_block_4(64, 128,
                                                pool=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


class CustomEmotionModel7(nn.Module):
    def __init__(self, num_classes=6, **kwargs):
        super(CustomEmotionModel7, self).__init__()

        self.conv_block1 = _create_conv_block(3, 64)
        self.conv_block2 = _create_conv_block_2(64, 96)
        self.conv_block3 = _create_conv_block_2(96, 128, pool=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


class DynamicModel(nn.Module):
    def __init__(self, num_classes=6, hidden_layers=1, dropout=0.2, **kwargs):
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
                self.hidden_layers.append(nn.Linear(256 // 2 ** _, 256 // 2 ** (_ + 1)))

            # define output layer depending on number of hidden layers
            self.output = nn.Linear(256 // 2 ** (hidden_layers - 1), num_classes)

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


MODELS = {
    "DynamicModel": DynamicModel,
    "LeNet": LeNet,
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "EmotionModel2": EmotionModel,
    "CustomEmotionModel3": CustomEmotionModel3,
    "CustomEmotionModel4": CustomEmotionModel4,
    "CustomEmotionModel5": CustomEmotionModel5,
    "CustomEmotionModel7": CustomEmotionModel3,
    "Apron": Apron,
    "MobileNetV2": MobileNetV2
}
