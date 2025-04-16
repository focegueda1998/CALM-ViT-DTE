import torch
import torchvision.transforms.v2
import Vi_Tools_CNN_less as vt
import torchvision
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from torch.optim.lr_scheduler import StepLR
import csv
from torchvision import models

parent_dir = "/config"

# Now completely decoupled from the PyTorch module!
class ViT(torch.nn.Module):
    def __init__(self, device, type=8, heads=16, seq_length=256, in_features=768, dim2=256, mean_var_hidden=192, out_features=1000, generate=False):
        super().__init__()
        self.device = device
        self.generate = generate
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
        if not generate:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features*2, bias=False),
                torch.nn.Linear(in_features*2, in_features*2, bias=False),
                torch.nn.Linear(in_features*2, out_features, bias=False)
            ).to(device)
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features*2, bias=True),
                torch.nn.Linear(in_features*2, in_features*2, bias=True),
                torch.nn.Linear(in_features*2, in_features, bias=True)
            ).to(device)

    def forward(self, q):
        x = self.autoencoder(q)
        if not self.generate:
            x = self.head(x[:, 0])
        else:
            x = self.head(x[:, 1:])
        return x

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
    criterion = torch.nn.CrossEntropyLoss()
    # criterion_mse = torch.nn.MSELoss()
    # criterion_KL = torch.nn.KLDivLoss(reduction='batchmean')
    print(model)

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.v2.Compose([
        torchvision.transforms.v2.Resize((256, 256)),
        torchvision.transforms.v2.ToImage(),
        torchvision.transforms.v2.ToDtype(dtype=torch.float32, scale=True),
        torchvision.transforms.v2.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    split = "train"

    dataset = ImageNet(
        root=parent_dir + "/imagenet/",
        split=split,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=48, shuffle=True)
    if split == "train":
        for epoch in range(5):
            model.train()
            for i, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss_1 = criterion(y_hat.squeeze(), y) # The labels need to be floating point
                # loss_2 = criterion_mse(img, x)
                # img_flat = img.reshape(-1, 256 * 256 * 3)
                # x_flat = x.reshape(-1, 256 * 256 * 3)
                # img_log = torch.nn.functional.log_softmax(img_flat, dim=1)
                # x_soft = torch.nn.functional.softmax(x_flat, dim=1)
                # loss_3 = criterion_KL(img_log, x_soft)
                loss = loss_1 # + loss_2 + loss_3
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(y_hat.data, 1)
                correct = (predicted == y).sum().item()
                accuracy = correct / y.size(0)
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, Accuracy: {accuracy}")
                # if i % 10 == 0:
                #     save_samples(img)
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
                correct = (predicted == y).sum().item()
                total = y.size(0)
                print(f"Batch {i}, Accuracy: {correct} / {total}")
                # save_samples(img)