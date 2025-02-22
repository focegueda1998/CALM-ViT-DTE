import torch
import torchvision
import torch.optim as optim
import numpy as np
import torchvision.transforms.functional
from time import time
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, udf, rand
from pyspark.sql.types import IntegerType, StructField, StructType, BinaryType, StringType, ArrayType
from subprocess import call
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from io import BytesIO
from torchvision import models
from pyspark.ml.torch.distributor import TorchDistributor

class ImageTokenizer(torch.nn.Module):
    """
    Image Tokenizer class for the Vision Transformer model.
    """
    def __init__(self, num_channels=3, patch_size=16, embed_dim=768, image_size=224):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, embed_dim))
    
    def forward(self, x):
        # x = Image.open(BytesIO(x))
        # x = torchvision.transforms.functional.resize(x, (224, 224), antialias=True)
        # x = torchvision.transforms.functional.to_tensor(x)
        x = x.unsqueeze(0)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.pos_embed.expand(x.size(0), -1, -1)
        x += pos_embed
        return x

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

def vision_transformer(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)

