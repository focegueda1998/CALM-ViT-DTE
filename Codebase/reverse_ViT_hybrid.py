import torch
import Vi_Tools_CNN_less as vt
import torchvision
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import csv
from torchvision import models

parent_dir = "/config"

# Now completely decoupled from the PyTorch module!
class ViT(torch.nn.Module):
    def __init__(self, device, type=8, heads=16, seq_length=256, in_features=768, dim2=256, mean_var_hidden=192, out_features=1): 
        super().__init__()
        self.device = device
        self.num_classes = out_features
        self.seq_length = seq_length
        if type == 8:
            self.autoencoder = vt.EncoderDecoder_8(
                heads=heads,
                seq_length=seq_length + 1,
                dim1=in_features,
                dim2=dim2,
                mean_var_hidden=mean_var_hidden
            ).to(device)
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=False).to(device)
        )
        self.img_head_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features*2, bias=True),
            torch.nn.Linear(in_features*2, in_features*2, bias=True),
            torch.nn.Linear(in_features*2, in_features, bias=True)
        ).to(device)
        # self.img_head = torch.nn.ConvTranspose2d(in_features, 3, kernel_size=(1, seq_length), stride=(1, seq_length), padding=0, bias=True).to(device)
        self.constrain = torch.nn.Tanh()

    def forward(self, q):
        x = self.autoencoder(q)
        cls = self.cls_head(x[:, 0])
        img = self.img_head_2(x[:, 1:])
        # img = self.img_head(img.permute(0, 2, 1).reshape(-1, img.shape[2], img.shape[1], 1))
        img = self.constrain(img.reshape(-1, self.seq_length, self.seq_length, 3).permute(0, 3, 1, 2))
        return cls, img

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

def save_samples(imgs):
    for i, img in enumerate(imgs):
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = (img + 1)/2
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
    # model = initialize_vit(device, type="l")
    model = initialize_vit(device, weights=f"")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion_binary = torch.nn.BCEWithLogitsLoss()
    criterion_mse = torch.nn.MSELoss()
    criterion_KL = torch.nn.KLDivLoss(reduction='batchmean')
    print(model)

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.Lambda(lambda x: torch.nn.functional.tanh(x))
    ])

    dataset = ImageDataset(f"{parent_dir}/AI_Human_Generated_Images/", "train.csv", transform=transform, train=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(5):
        dataset.reshuffle()
        dataset.train = True
        dataloader = DataLoader(dataset, batch_size=48, shuffle=True)
        model.train()
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat, img = model(x)
            loss_1 = criterion_binary(y_hat.squeeze(), y.float()) # The labels need to be floating point
            loss_2 = criterion_mse(img, x)
            img_flat = img.reshape(-1, 256 * 256 * 3)
            x_flat = x.reshape(-1, 256 * 256 * 3)
            img_log = torch.nn.functional.log_softmax(img_flat, dim=1)
            x_soft = torch.nn.functional.softmax(x_flat, dim=1)
            loss_3 = criterion_KL(img_log, x_soft)
            loss = loss_1 + loss_2 + loss_3
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
            if i % 10 == 0:
                save_samples(img)
        scheduler.step()