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


def grad_cam(model: nn.Module, data_path, examples=10):
    print(model.__class__)
    model_grad = LeNetGrad(model)
    model_grad.eval()
    model_grad.to("cpu")
    _, result = use_model('no_id_known', model_grad, data_path)
    acc = accuracies(result)
    print(acc)
    tensors, labels = get_images_and_labels(data_path, limit=examples)
    for t, l in zip(tensors, labels):
        print(f"Label: {l}")
        t = t.unsqueeze(0)
        heatmap_pred, exact_pred = cam_for_label(model_grad, t, l, true_label=False)
        heatmap_label, _ = cam_for_label(model_grad, t, l, true_label=True)

        img = t.squeeze().numpy().transpose(1, 2, 0)
        img = np.uint8((img + 1.) * 126)
        heat_img = heatmap_on_img(img, heatmap_pred)
        # img and heat_img side by side
        fig, ax = plt.subplots(1, 2)
        ax[0].title.set_text(f"Prediction: {LABEL_TO_STR[exact_pred.item()]}, True: {LABEL_TO_STR[l]}")
        ax[0].imshow(img)
        ax[1].imshow(heat_img)
        plt.show()


def cam_for_label(model_grad, tens, label, true_label=False):
    model_grad.eval()
    model_grad.to("cpu")
    pred = model_grad(tens)
    exact_pred = pred.argmax(dim=1, keepdim=True)
    pred[:, label if true_label else exact_pred].backward()
    gradients = model_grad.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model_grad.get_activations(tens).detach()
    # weight the channels by corresponding gradients
    for i in range(16):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    # draw the heatmap
    return heatmap, exact_pred


def heatmap_on_img(img, heatmap):
    print(img.shape)
    print(heatmap)
    heatmap = np.uint8(heatmap*255)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.3 + img
    return np.uint8(superimposed_img)


GRAD_MODELS = {
    "LeNet": LeNetGrad,
}
