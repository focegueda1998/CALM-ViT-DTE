import torch
import numpy as np
import torchvision
import torch.optim as optim
import torchvision.transforms.functional
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import csv
from torchvision import models

parent_dir = "/config"

class ViT(torch.nn.Module):
    def __init__(self, device, weights=None, type="b", out_features=1): 
        super().__init__()
        self.device = device
        if type == "b":
            in_features = 768
            self.vit = models.vit_b_16(weights=weights).to(device) if weights else models.vit_b_16().to(device)
        elif type == "l":
            in_features = 1024
            self.vit = models.vit_l_16(weights=weights).to(device) if weights else models.vit_l_16().to(device)
        self.vit.num_classes = out_features
        for param in self.vit.parameters():
            param.requires_grad = True
        self.vit.heads = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=True).to(device)
        )

    def forward(self, x):
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1,2)
        batch_size, seq_len, _ = x.shape
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x += self.vit.encoder.pos_embedding[:, : (seq_len + 1), :]
        x = self.vit.encoder.dropout(x)
        attention_map = None
        for layer in self.vit.encoder.layers:
            y = x
            x = layer.ln_1(x)
            x, attention_map = layer.self_attention(x, x, x, need_weights=True)
            x = layer.dropout(x)
            x = x + y
            y = layer.ln_2(x)
            y = layer.mlp(y)
            x = x + y
        x = self.vit.encoder.ln(x)
        cls_output = x[:, 0]
        x = self.vit.heads(cls_output)
        attention_map = attention_map[:, :, 1:]
        attention_map = attention_map.mean(dim=1)
        return x, attention_map

class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, split_ratio=0.8, train=True):
        self.data = []
        self.train = train
        with open(root_dir + csv_file, 'r') as file:
            reader = csv.reader(file)
            reader.__next__() # Skip the header
            for row in reader:
                self.data.append(row)
            random.shuffle(self.data)
        self.split = int(split_ratio * len(self.data))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data[:self.split]) if self.train else len(self.data[self.split:])

    def __getitem__(self, idx):
        data = self.data[:self.split] if self.train else self.data[self.split:]
        img_name = f"{parent_dir}/AI_Human_Generated_Images/"+ data[idx][1]
        image = self.transform(Image.open(img_name))
        label = torch.tensor(int(data[idx][2]))
        return image, label

    def reshuffle(self):
        random.shuffle(self.data)

def save_samples(batch, attention_maps):
    try:
        for i, attention_map in enumerate(attention_maps):
            sample = batch[i].permute(1, 2, 0).cpu().detach().numpy()
            Image.fromarray((sample * 255).astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample.jpg")
            a_min = attention_map.min()
            a_max = attention_map.max()
            a_diff = a_max - a_min if a_max - a_min > 1e-6 else 1e-6
            full_attention = (attention_map - a_min) / a_diff
            full_attention = full_attention.mul(255).clamp(0, 255)
            Image.fromarray(full_attention.astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample_attn.jpg")
    except Exception as e:
        print(f"Something went wrong: {e}")

def initialize_res18(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

    for param in model.parameters():
        param.requires_grad = False

    num_classes = 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model

def initialize_res50(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)

    num_classes = 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True

    return model

def initialize_vit(device, weights: str="DEFAULT", type="b"):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """
    model = None
    #! Add more pretrained weights later.
    match weights:
        case "DEFAULT":
            model = ViT(
                device, 
                weights=models.ViT_L_16_Weights.DEFAULT if type == "l" else models.ViT_B_16_Weights.DEFAULT,
                type=type
            ).to(device)
        case "":
            model = ViT(device, type=type).to(device)
        case _:
            model = ViT(device, type=type).to(device)
            try:
                model.load_state_dict(torch.load(weights, weights_only=True))
            except Exception as e:
                print(f"Could not load the weights due to {e}. No weights loaded.")
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_vit(device, type="l")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(model)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])

    dataset = ImageDataset(f"{parent_dir}/AI_Human_Generated_Images/", "train.csv", transform=transform, train=True)
    dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

    for epoch in range(1):
        dataset.reshuffle()
        dataset.train = True
        dataloader = DataLoader(dataset, batch_size=36, shuffle=True)
        model.train()
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat, _ = model(x)
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

        dataset.train = False
        dataloader = DataLoader(dataset, batch_size=36, shuffle=True)
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                batch = x
                x = x.to(device)
                y = y.to(device)
                y_hat, attention_maps = model(x)
                y_prob = torch.sigmoid(y_hat)
                y_pred = (y_prob > 0.5).float()
                y_pred = y_pred.squeeze().tolist()
                predicted = 0
                for j, pred in enumerate(y_pred):
                    if pred == y.float()[j]:
                        predicted += 1
                print(f"Epoch {epoch}, Batch {i}, Predicted: {predicted}/{len(y_pred)}")
        
        save_samples(batch, attention_maps)
