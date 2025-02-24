import torch
import torchvision
import torch.optim as optim
import numpy as np
import torchvision.transforms.functional
from time import time
from subprocess import call
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import csv
from io import BytesIO
from torchvision import models, datasets
import torch.distributed as dist

# class ImageTokenizer(torch.nn.Module):
#     """
#     Image Tokenizer class for the Vision Transformer model.
#     """
#     def __init__(self, device, num_channels=3, patch_size=16, embed_dim=768, image_size=224):
#         super().__init__()
#         self.num_channels = num_channels
#         self.patch_size = patch_size
#         self.device = device
#         self.proj = torch.nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size).to(device)
#         self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim)).to(device)
#         self.pos_embed = torch.nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, embed_dim)).to(device)
    
#     def forward(self, x):
#         x = Image.open(BytesIO(x))
#         x = torchvision.transforms.functional.resize(x, (224, 224), antialias=True)
#         x = torchvision.transforms.functional.to_tensor(x).to(self.device)
#         x = x.unsqueeze(0)
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)
#         cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         pos_embed = self.pos_embed.expand(x.size(0), -1, -1)
#         x += pos_embed
#         return x

class ViT(torch.nn.Module):  # Inherit from nn.Module
    def __init__(self, device, in_features=1024, out_features=1):  # Pass device, in_features, out_features
        super().__init__()  # Call super().__init__()
        self.device = device
        self.vit = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT).to(device)

        # Freeze all the parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Modify the classifier head
        self.vit.heads = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=True).to(device)
        )

    def forward(self, x):
        x = self.vit(x)  # Pass the image directly to the ViT model
        return x

class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.data = []
        with open(root_dir + csv_file, 'r') as file:
            reader = csv.reader(file)
            reader.__next__() # Skip the header
            for row in reader:
                self.data.append(row)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = "/home/AI_Human_Generated_Images/"+ self.data[idx][1]
        image = self.transform(Image.open(img_name))
        label = torch.tensor(int(self.data[idx][2]))
        return image, label

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

def initialize_vit(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """
    model = ViT(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    return model, optimizer, scheduler, criterion

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, scheduler, criterion = initialize_vit(device)
    print(model)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x)
    ])

    dataset = ImageDataset("/home/AI_Human_Generated_Images/", "train.csv", transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(1):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y.float()) # The labels need to be floating point
            loss.backward()
            optimizer.step()
            y_prob = torch.sigmoid(y_hat)
            y_pred = (y_prob > 0.5).float()
            y_pred = y_pred.squeeze().tolist()
            predicted = 0
            for j, pred in enumerate(y_pred):
                if pred == y.float()[j]:
                    predicted += 1
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, Predicted: {predicted}/{len(y_pred)}")