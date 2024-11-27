import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 将图像转换为 Tensor 并增加通道维度 (unsqueeze(0) adds a channel dimension)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, 28, 28)
        image = image / 255.0  # Normalize to [0, 1]

        label = torch.tensor(label, dtype=torch.long)

        return image, label

class ImprovedCNN2(torch.nn.Module):
    def __init__(self):
        super(ImprovedCNN2, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # New conv layer
        self.bn3 = torch.nn.BatchNorm2d(128)

        # Update the input size of fc1 based on the output size of the last conv layer
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 256)  # Adjusted size
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(0.5)
        self.swish = torch.nn.SiLU()

    def forward(self, x):
        x = self.swish(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = self.swish(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = self.swish(self.bn3(self.conv3(x)))  # New conv layer
        x = torch.max_pool2d(x, 2)

        # Print the shape before flattening to determine the input size for fc1
        #print(x.shape)  # Debugging line to check the shape

        x = x.view(x.size(0), -1)  # Flatten the output

        # Now x has the correct size for fc1
        x = self.swish(self.fc1(x))  # Swish activation
        x = self.dropout(x)
        x = self.swish(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x