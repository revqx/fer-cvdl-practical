import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import get_images_and_labels, LABEL_TO_STR
from analyze import accuracies
from inference import use_model


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


def grad_cam(model: nn.Module, data_path, examples=10, random=True):
    print(model.__class__)
    model_grad = LeNetGrad(model)
    _, result = use_model('no_id_known', model_grad, data_path)
    acc = accuracies(result)
    print(acc)
    tensors, labels, filenames = get_images_and_labels(data_path, limit=examples, random=random)

    for t, l, p in zip(tensors, labels, filenames):
        print(f"Label: {l}")
        explain_image(model_grad, t, l, p)


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
        for i in range(16):
            activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    return heatmap


def heatmap_on_img(img, heatmap, alpha=0.4):
    print(img.shape)
    print(heatmap)
    heatmap = np.uint8(heatmap*255)
    #heatmap = np.ones_like(heatmap) * 255 - heatmap
    print(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    #heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)
    heatmap = cv2.blur(heatmap, (5, 5))
    # superimposed img as imag with transparent overlay of heatmap
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
    return np.uint8(superimposed_img)


def explain_image(model_grad, tens, label, path):
    fig, ax = plt.subplots(2, 2)
    for a in ax.reshape(-1):
        a.axis('off')

    # model prediction
    pred = np.argsort(model_grad(tens).detach()).numpy()[0][-1]
    pred_str = LABEL_TO_STR[pred]

    # original image
    img = tens.squeeze().numpy().transpose(1, 2, 0)
    img = np.uint8((img + 1.) * 126)
    # set title to path
    ax[0, 0].title.set_text(f"{path}")
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

    plt.show()


GRAD_MODELS = {
    "LeNet": LeNetGrad,
}
