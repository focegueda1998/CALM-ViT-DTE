import torch
import torchvision.transforms.v2 as transforms
import Vi_Tools_CNN_less_V2 as vt
import torchvision
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.utils import spectral_norm as sn, remove_spectral_norm as rsn
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.datasets import ImageNet
from torch.optim.lr_scheduler import StepLR
import csv
from torchvision.transforms.functional import InterpolationMode
from torchvision import models
from torch.amp import autocast, GradScaler

parent_dir = "/config"

# Now completely decoupled from the PyTorch module!
class ViT(torch.nn.Module):
    def __init__(self, device, type=8, heads=12, seq_length=256, in_features=768,
                 dim_step=48, mean_var_hidden=192,
                 seq_len_step=16, seq_len_reduce=128, out_features=1000,
                 force_reduce=False, generate=True):
        super().__init__()
        self.device = device
        self.generate = generate
        self.num_classes = out_features
        self.seq_length = seq_length
        # We use two sets of positional embeddings, one for the encoder (row tokens) and one
        # for the decoder (columns).
        # self.pos_embeddings = [torch.nn.Parameter(torch.randn(1, seq_length, in_features)).to(device), 
        #                       torch.nn.Parameter(torch.randn(1, seq_length, in_features)).to(device)]
        if type == 8:
            self.autoencoder = vt.EncoderDecoder_8(
                heads=heads,
                dim1=in_features,
                dim_step=dim_step,
                mean_var_hidden=mean_var_hidden,
                seq_length=seq_length,
                seq_len_step=seq_len_step,
                seq_len_reduce=seq_len_reduce,
                out_features_override=None,
                force_reduce=force_reduce
            ).to(device)
        if not generate:
            self.pool = torch.nn.AdaptiveAvgPool1d(1).to(device)
            self.head = torch.nn.Sequential(
                sn(torch.nn.Linear(in_features, in_features * 2, bias=False)).to(device),
                torch.nn.GELU().to(device),
                sn(torch.nn.Linear(in_features * 2, out_features, bias=False)).to(device)
            ).to(device)
        else:
            # self.head = torch.nn.Sequential(
            #     sn(torch.nn.Linear(in_features, in_features * 2, bias=False)).to(device),
            #     torch.nn.GELU().to(device),
            #     sn(torch.nn.Linear(in_features * 2, in_features, bias=False)).to(device),
            # ).to(device)
            hidden_channels = 32
            self.proj = torch.nn.Sequential(
                sn(torch.nn.Conv2d(3, hidden_channels, kernel_size=1, groups=1, bias=True)),
                torch.nn.GELU(approximate='none'),
                sn(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, groups=hidden_channels, padding_mode='zeros')),
                torch.nn.GELU(approximate='none'),
                sn(torch.nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True))
            )

    
    def forward(self, q):
        if not self.generate:
            # Average pool the sequence dimension
            x, kl_loss = self.autoencoder(q)
            x = x.permute(0, 2, 1)
            x = self.pool(x).squeeze(-1)
            x = self.head(x)
        else:
            x, kl_loss = self.autoencoder(q)
            # x = self.head(x)
            x_img = self.proj(x.reshape(x.shape[0], x.shape[1], x.shape[1], 3).permute(0, 3, 1, 2))
            x_img = x_img.permute(0, 2, 3, 1)
            x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], x_img.shape[2] * x_img.shape[3])
            x = x + x_img
        return x, kl_loss

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

def save_samples(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    imgs = torch.sigmoid(imgs)
    for i, img in enumerate(imgs):
        img = img.permute(1, 2, 0)  # C x H x W to H x W x C
        img = img.detach().cpu().numpy()
        plt.imsave(f"{parent_dir}/Codebase/samples/sample_{i}.png", img)

def initialize_vit(device, weights: str="DEFAULT"):
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
            model = ViT(device).to(device)
        case _:
            model = ViT(device)
            try:
                model.load_state_dict(torch.load(weights, weights_only=True))
            except Exception as e:
                print(f"Could not load the weights due to {e}. No weights loaded.")
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(device, type=8, heads=12, seq_length=224, in_features=672,
                 dim_step=48, mean_var_hidden=240,
                 seq_len_step=16, seq_len_reduce=80, out_features=1000,
                 force_reduce=False, generate=False)
    model.load_state_dict(torch.load(f"{parent_dir}/Codebase/models/model_cls.pth", map_location=device, weights_only=True), strict=False)
    optimizer = optim.Adam(model.parameters(), lr=3.1e-3, weight_decay=0.02)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.HuberLoss(delta=1.0)
    criterion_KL = torch.nn.KLDivLoss(reduction='batchmean')
    print(model)

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=(0.5, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=(-0.125, 0.125)),
        transforms.RandomSolarize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    split = "train"

    dataset = ImageNet(
        root=parent_dir + "/imagenet/",
        split=split,
        transform=transform
    )
    cut_mix = transforms.CutMix(num_classes=1000, alpha=1.0)
    mix_up = transforms.MixUp(num_classes=1000, alpha=0.8)
    mix_both = transforms.RandomChoice([cut_mix, mix_up])
    def collate_fn(batch): return mix_both(*default_collate(batch))
    dataloader = DataLoader(dataset, batch_size=100, num_workers=1, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    scaler = GradScaler(enabled=True)
    if split == "train":
        for epoch in range(5):
            model.train()
            for i, (x, y) in enumerate(dataloader):
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    x = x.to(device)
                    y = y.to(device)
                    y_hat, kl_loss = model(x)
                    # img = y_hat.reshape(-1, 224, 224, 3)
                    # img = img.permute(0, 3, 1, 2)
                    loss_1 = criterion(y_hat.squeeze(), y) # The labels need to be floating point
                    # loss_2 = criterion_mse(img, x)
                    # img_flat = img.reshape(-1, 224 * 224 * 3)
                    # x_flat = x.reshape(-1, 224 * 224 * 3)
                    # img_log = torch.nn.functional.log_softmax(img_flat, dim=1)
                    # x_soft = torch.nn.functional.softmax(x_flat, dim=1)
                    # loss_3 = criterion_KL(img_log, x_soft)
                    loss = loss_1 # + loss_2 + loss_3
                    # loss = loss_2 + kl_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # y_hat = model(x)
                # loss = criterion(y_hat.squeeze(), y)
                # loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=False)

                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad()
                # optimizer.step()
                # _, predicted = torch.max(y_hat.data, 1)
                # _, y_labels = torch.max(y.data, 1)  
                # correct = (predicted == y_labels).sum().item()
                # accuracy = correct / y.size(0)
                # print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, Accuracy: {accuracy}")
                # save_samples(img)
                # print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(f"- {name}")
            scheduler.step()
    else:
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for i, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                print(f"Batch {i}, Accuracy: {(correct / total) * 100}%")
                # save_samples(img)