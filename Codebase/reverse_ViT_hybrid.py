import torch
from transformers import ViTModel, ViTConfig, ViTImageProcessor
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def initialize_res18(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

    for param in model.parameters():
        param.requires_grad = False

    num_classes = 10
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return model, optimizer, scheduler

def initialize_res50(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """


    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)

    num_classes = 10
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return model, optimizer, scheduler