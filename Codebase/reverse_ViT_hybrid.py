import torch
import numpy as np
import torchvision
import torch.optim as optim
import torchvision.transforms.functional
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import csv
from torchvision import models
from torchvision.models.vision_transformer import MLPBlock, EncoderBlock
from typing import Callable
from functools import partial

parent_dir = "/config"

# Multi-Head Latent Distribution Attention
class MLDA_Block(torch.nn.Module):
    def __init__(
        self,
        heads: int,
        upsamp_hidden: int,
        downsamp_hidden: int,
        mean_var_hidden: int,
        mlp_dim: int,
        dropout=0.0,
        attention_dropout=0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.heads = heads
        self.ln_1 = norm_layer(upsamp_hidden)
        #! This is ugly but I don't care :)
        # One-Shot Encoders
        self.cq_downsample = torch.nn.Sequential(
            torch.nn.Linear(upsamp_hidden, downsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.cq_ln = norm_layer(downsamp_hidden)
        self.ckv_downsample = torch.nn.Sequential(
            torch.nn.Linear(upsamp_hidden, downsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.ckv_ln = norm_layer(downsamp_hidden)
        # Mean and Variance Bottleneck
        self.mean_q = torch.nn.Sequential(
            torch.nn.Linear(downsamp_hidden, mean_var_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.mean_q_ln = norm_layer(mean_var_hidden)
        self.var_q = torch.nn.Sequential(
            torch.nn.Linear(downsamp_hidden, mean_var_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.var_q_ln = norm_layer(mean_var_hidden)
        self.mean_kv = torch.nn.Sequential(
            torch.nn.Linear(downsamp_hidden, mean_var_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.mean_kv_ln = norm_layer(mean_var_hidden)
        self.var_kv = torch.nn.Sequential(
            torch.nn.Linear(downsamp_hidden, mean_var_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.var_kv_ln = norm_layer(mean_var_hidden)
        # Decoders
        self.qc_upsample = torch.nn.Sequential(
            torch.nn.Linear(mean_var_hidden, downsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(downsamp_hidden, upsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.qc_ln = norm_layer(upsamp_hidden)
        self.kc_upsample = torch.nn.Sequential(
            torch.nn.Linear(mean_var_hidden, downsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(downsamp_hidden, upsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.kc_ln = norm_layer(upsamp_hidden)
        self.vc_upsample = torch.nn.Sequential(
            torch.nn.Linear(mean_var_hidden, downsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(downsamp_hidden, upsamp_hidden, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.vc_ln = norm_layer(upsamp_hidden)
        self.self_attention = torch.nn.MultiheadAttention(upsamp_hidden, heads, dropout=attention_dropout, bias=True, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.ln_2 = norm_layer(upsamp_hidden)
        self.mlp = MLPBlock(upsamp_hidden, mlp_dim, dropout)

    def forward(self, input):
        # One-Shot Encoders
        x = input
        x = self.ln_1(x)
        cq = self.cq_downsample(x)
        cq = self.cq_ln(cq)
        ckv = self.ckv_downsample(x)
        ckv = self.ckv_ln(ckv)
        # Mean and Variance Bottleneck
        mean_q = self.mean_q(cq)
        mean_q = self.mean_q_ln(mean_q)
        var_q = self.var_q(cq)
        var_q = self.var_q_ln(var_q)
        mean_kv = self.mean_kv(ckv)
        mean_kv = self.mean_kv_ln(mean_kv)
        var_kv = self.var_kv(ckv)
        var_kv = self.var_kv_ln(var_kv)
        # Compute samples
        zcq = mean_q + torch.randn_like(var_q) * torch.exp(0.5 * var_q)
        zckv = mean_kv + torch.randn_like(var_kv) * torch.exp(0.5 * var_kv)
        # Decoders
        qc = self.qc_upsample(zcq)
        qc = self.qc_ln(qc)
        kc = self.kc_upsample(zckv)
        kc = self.kc_ln(kc)
        vc = self.vc_upsample(zckv)
        vc = self.vc_ln(vc)
        # Attention
        x, _ = self.self_attention(qc, kc, vc, need_weights=False)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class ViT(torch.nn.Module):
    def __init__(self, device, weights=None, type="b", downsamp_hidden=256, mean_var_hidden=128, out_features=1): 
        super().__init__()
        self.device = device
        in_features = 768
        if type == "b":
            in_features = 768
            self.vit = models.vit_b_16(weights=weights).to(device) if weights else models.vit_b_16().to(device)
        elif type == "l":
            in_features = 1024
            self.vit = models.vit_l_16(weights=weights).to(device) if weights else models.vit_l_16().to(device)
        self.vit.num_classes = out_features
        for i, block in enumerate(self.vit.encoder.layers):
            if isinstance(block, EncoderBlock):
                self.vit.encoder.layers[i] = MLDA_Block(
                    heads=16,
                    upsamp_hidden=in_features,
                    downsamp_hidden=downsamp_hidden,
                    mean_var_hidden=mean_var_hidden,
                    mlp_dim=in_features
                ).to(device)
        self.vit.heads = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=True).to(device)
        )
        for param in self.vit.parameters(recurse=True):
            param.requires_grad = True
    
    def forward(self, x):
        return self.vit(x)

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

def save_samples(batch, attention_maps, y_pred, y_actual):
    try:
        for i, attention_map in enumerate(attention_maps):
            sample = batch[i]
            sample = torchvision.transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )(sample)
            sample = sample.permute(1, 2, 0).cpu().detach().numpy()
            Image.fromarray((sample * 255).astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample.jpg")
            full_attention = attention_map
            f_min = full_attention.min()
            f_max = full_attention.max()
            f_diff = f_max - f_min if f_max - f_min > 1e-6 else 1e-6
            full_attention = (full_attention - f_min) / f_diff
            full_attention_2 = full_attention.mul(255).clamp(0, 255).reshape(14, 14).cpu().detach().numpy()
            Image.fromarray(full_attention_2.astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample_attn.jpg")
            full_attention, _ =  torch.sort(full_attention)
            full_attention = full_attention.cpu().detach().numpy()
            plt.scatter(range(196), full_attention)
            plt.grid(True)
            plt.xlabel("Attention Heads")
            plt.ylabel("Attention Weights")
            plt.title(f"Predicted: {int(y_pred[i])}, Actual: {int(y_actual[i])}")
            plt.savefig(f"{parent_dir}/Codebase/samples/{i}_sample_attn_scatter.jpg")
            plt.close()
    except Exception as e:
        print(f"Something went wrong: {e}")

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
            model = ViT(device, type=type)
            try:
                model.load_state_dict(torch.load(weights, weights_only=True))
            except Exception as e:
                print(f"Could not load the weights due to {e}. No weights loaded.")
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = initialize_vit(device, type="l")
    model = initialize_vit(device, weights=f"", type="b")
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(f"{parent_dir}/AI_Human_Generated_Images/", "train.csv", transform=transform, train=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(1):
        dataset.reshuffle()
        dataset.train = True
        dataloader = DataLoader(dataset, batch_size=96, shuffle=True)
        model.train()
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

        dataset.train = False
        dataloader = DataLoader(dataset, batch_size=36, shuffle=True)
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                batch = x
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y_prob = torch.sigmoid(y_hat)
                y_pred = (y_prob > 0.5).float()
                y_pred = y_pred.squeeze().tolist()
                predicted = 0
                for j, pred in enumerate(y_pred):
                    if pred == y.float()[j]:
                        predicted += 1
                print(f"Epoch {epoch}, Batch {i}, Predicted: {predicted}/{len(y_pred)}")