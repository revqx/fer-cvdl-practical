import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import get_images_and_labels, LABEL_TO_STR

"""
Good pictures for Model 52q6cmxc:
    - img 167: fear
    - img 470: sadness
    - img 017: mislabeled as sadness, actually surprise (the net has a point there)
    - img 515: anger
    - img 397: happiness
    - img 126: fear (works even though there are hands in the face)
"""


class LeNetGrad(nn.Module):
    def __init__(self, trained_model):
        super(LeNetGrad, self).__init__()
        self.model = trained_model
        self.model.eval()
        self.gradients = None

    def activations_hook(self, grad):
        # hook for the gradients of the activations
        self.gradients = grad

    def forward(self, x):
        x = F.relu(self.model.conv1(x))
        x = self.model.pool(x)
        x = F.relu(self.model.conv2(x))
        h = x.register_hook(self.activations_hook)
        x = self.model.pool(x)
        x = x.view(-1, 16 * self.model.pool2_size * self.model.pool2_size)
        x = F.relu(self.model.fc1(x))
        x = self.model.fc2(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = F.relu(self.model.conv1(x))
        x = self.model.pool(x)
        x = F.relu(self.model.conv2(x))
        x = self.model.pool(x)
        return x


class CustomEmotionModel3Grad(nn.Module):
    def __init__(self, trained_model):
        super(CustomEmotionModel3Grad, self).__init__()
        self.model = trained_model
        self.model.eval()
        self.gradients = None

    def activations_hook(self, grad):
        # hook for the gradients of the activations
        self.gradients = grad

    def forward(self, x):
        x = self.model.conv_block1(x)
        x = self.model.conv_block2(x)
        x = self.model.conv_block3(x)
        x = self.model.conv_block4(x)
        x.register_hook(self.activations_hook)
        x = self.model.avgpool(x)
        x = self.model.flatten(x)
        x = F.relu(self.model.fc1(x))
        x = self.model.dropout(x)
        x = F.relu(self.model.fc2(x))
        x = self.model.fc3(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.model.conv_block1(x)
        x = self.model.conv_block2(x)
        x = self.model.conv_block3(x)
        x = self.model.conv_block4(x)
        return x


def grad_cam(model: nn.Module, data_path, examples=10, random=True, path_contains=None, save_path=None):
    model_grad = GRAD_MODELS[model.__class__.__name__](model)
    tensors, labels, filenames = get_images_and_labels(data_path, limit=examples, random=random, path_contains=path_contains)
    print(model_grad)

    for t, l, p in zip(tensors, labels, filenames):
        explain_image(model_grad, t, l, p, save_path=save_path)


def for_all_labels(model_grad, tens, label):
    # plt with 2 cols and 6 rows
    fig, ax = plt.subplots(2, 6)
    for l in range(6):
        activations, pooled_gradients = cam_for_label(model_grad, tens, l, true_label=True)
        heatmap = create_heatmap(activations, pooled_gradients)
        ax[0, l].axis('off')
        ax[1, l].axis('off')
        img = tens.squeeze().numpy().transpose(1, 2, 0)
        img = np.uint8((img + 1.) * 126)
        heat_img = heatmap_on_img(img, heatmap)
        ax[0, l].title.set_text(f"{LABEL_TO_STR[l]}{'!' if l == label else ''}")
        ax[1, l].imshow(heatmap)
        ax[0, l].imshow(heat_img)


def cam_for_label(model_grad, tens, label, true_label=False):
    model_grad.eval()
    model_grad.to("cpu")
    pred = model_grad(tens)
    exact_pred = pred.argmax(dim=1, keepdim=True)
    pred[:, label if true_label else exact_pred].backward()
    gradients = model_grad.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model_grad.get_activations(tens).detach()
    return activations, pooled_gradients


def create_heatmap(activations, pooled_gradients, activation_weighted=True):
    if activation_weighted:
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    return heatmap


def heatmap_on_img(img, heatmap, alpha=0.4):
    heatmap = np.uint8(heatmap*255)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.blur(heatmap, (5, 5))
    # superimposed img as imag with transparent overlay of heatmap
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
    return np.uint8(superimposed_img)


def explain_image(model_grad, tens, label, path, save_path=None):
    fig, ax = plt.subplots(2, 2)
    for a in ax.reshape(-1):
        a.axis('off')

    # model prediction
    pred = np.argsort(model_grad(tens.unsqueeze(0)).detach()).numpy()[0][-1]
    pred_str = LABEL_TO_STR[pred]

    # original image
    img = tens.squeeze().numpy().transpose(1, 2, 0)
    img = np.uint8((img + 1.) * 126)
    # set title to filename, last part of path
    ax[0, 0].title.set_text(f"{path.split('/')[-1].split('.')[0]}")
    ax[0, 0].imshow(img)

    # weighted heatmap
    activations, pooled_gradients = cam_for_label(model_grad, tens.unsqueeze(0), label, true_label=False)
    weighted_heatmap = create_heatmap(activations, pooled_gradients, activation_weighted=True)
    ax[1, 0].title.set_text(f"Predicted: {pred_str}")
    ax[1, 0].imshow(weighted_heatmap)

    # heatmap
    activations, pooled_gradients = cam_for_label(model_grad, tens.unsqueeze(0), label, true_label=False)
    heatmap = create_heatmap(activations, pooled_gradients, activation_weighted=False)
    ax[0, 1].title.set_text(f"after conv")
    ax[0, 1].imshow(heatmap)

    # image with heatmap overlay
    heat_img = heatmap_on_img(img, weighted_heatmap)
    ax[1, 1].imshow(heat_img)

    if save_path is not None:
        full_save_file = os.path.join(save_path, f"{path.split('/')[-1].split('.')[0]}.png")
        print(f"Saving to {full_save_file}")
        plt.savefig(full_save_file)
    else:
        plt.show()


def overlay(image: np.array, model):
    model_grad = GRAD_MODELS[model.__class__.__name__](model)
    model_grad.eval()
    model_grad.to("cpu")
    # resize image
    image = cv2.resize(image, (64, 64))
    image = (image / 255. - 0.5) * 2
    torch_image = (
        torch.from_numpy(image)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
        .float()
        .to("cpu")
    )
    pred = model_grad(torch_image)
    exact_pred = pred.argmax(dim=1, keepdim=True)
    # picture with heatmap overlay
    activations, pooled_gradients = cam_for_label(model_grad, torch_image, exact_pred, true_label=True)
    heatmap = create_heatmap(activations, pooled_gradients)
    # torch image to np.array with 0to 1 values
    np_image = torch_image.squeeze().numpy().transpose(1, 2, 0)
    np_image = (np_image + 1.) / 2
    # to uint8
    np_image = np.uint8(np_image * 255)
    heat_img = heatmap_on_img(np_image, heatmap)
    return pred, heat_img


GRAD_MODELS = {
    "LeNet": LeNetGrad,
    "CustomEmotionModel3": CustomEmotionModel3Grad,
}
