import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class MNISTDataset(Dataset):
    """
    PyTorch Dataset class for the MNIST dataset.

    This class takes the images and labels as input and provides methods for:
    - Retrieving the total number of samples in the dataset.
    - Accessing each image-label pair at a specific index.

    The images are normalized to the range [0, 1] and reshaped to include a channel dimension.

    Parameters:
    - images (list or array): List or array of image data (each image is 28x28).
    - labels (list or array): List or array of labels corresponding to the images.

    Attributes:
    - images (list or array): The image data.
    - labels (list or array): The label data.
    """
    def __init__(self, images, labels):
        """
        Initialize the MNIST dataset.
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image-label pair at a specific index.

        The image is converted to a tensor and normalized to the range [0, 1], with an added channel dimension (1, 28, 28).

        Parameters:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - tuple: (image, label) where `image` is a tensor of shape (1, 28, 28) and `label` is a tensor of shape (long).
        """
        image = self.images[idx]  # Get the image at the specified index
        label = self.labels[idx]  # Get the corresponding label

        # Convert the image to a tensor, normalize, and add a channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, 28, 28)
        image = image / 255.0  # Normalize image pixel values to the range [0, 1]

        # Convert the label to a tensor of type long
        label = torch.tensor(label, dtype=torch.long)

        return image, label  # Return the image and label as a tuple

class ImprovedCNN(torch.nn.Module):
    def __init__(self):
        """
        Initialize the improved CNN model with additional convolutional layers, batch normalization,
        and dropout layers to improve performance and prevent overfitting.

        The network architecture includes:
        - 3 convolutional layers with increasing output channels (32, 64, 128).
        - Batch normalization after each convolutional layer.
        - Dropout to reduce overfitting.
        - Fully connected layers (fc1, fc2, fc3) for classification.
        - Swish activation function.
        """
        super(ImprovedCNN, self).__init__()

        # First convolutional layer with batch normalization
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)

        # Second convolutional layer with batch normalization
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)

        # Third convolutional layer with batch normalization (new layer)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        # Fully connected layers (fc1, fc2, fc3)
        # The input size to fc1 is based on the output size of the last convolutional layer
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 256)  # Adjusted input size
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

        # Dropout layer to prevent overfitting
        self.dropout = torch.nn.Dropout(0.5)

        # Swish activation function
        self.swish = torch.nn.SiLU()

    def forward(self, x):
        """
        Define the forward pass of the model. The input `x` passes through the following sequence:
        1. Convolutional layers with Batch Normalization and Swish activation.
        2. Max pooling layers to reduce spatial dimensions.
        3. Fully connected layers with Swish activation and dropout.
        4. Final output layer for classification.

        Parameters:
        - x (Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
        - Tensor: The output tensor representing class scores.
        """
        # Pass through the first convolutional layer, batch normalization, and Swish activation
        x = self.swish(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)  # Max pooling

        # Pass through the second convolutional layer, batch normalization, and Swish activation
        x = self.swish(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)  # Max pooling

        # Pass through the third convolutional layer, batch normalization, and Swish activation
        x = self.swish(self.bn3(self.conv3(x)))  # New conv layer
        x = torch.max_pool2d(x, 2)  # Max pooling

        # Debugging: Uncomment to print the shape of `x` before flattening
        # print(x.shape)  # For debugging: Check the shape of the tensor before flattening

        # Flatten the tensor before passing it to the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers

        # Pass through the fully connected layers with Swish activation and dropout
        x = self.swish(self.fc1(x))  # First fully connected layer with Swish activation
        x = self.dropout(x)  # Apply dropout
        x = self.swish(self.fc2(x))  # Second fully connected layer with Swish activation
        x = self.fc3(x)  # Output layer (class scores)

        return x

def initialize_cnn(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """
    # Instantiate the ImprovedCNN model and move it to the specified device (CPU or GPU)
    model = ImprovedCNN().to(device)

    # Use the Adam optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return model, optimizer, scheduler

def initialize_res18(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """
    # Instantiate the ImprovedCNN2 model and move it to the specified device (CPU or GPU)
    #model = ImprovedCNN().to(device)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    # Modify the first convolutional layer to accept 1 input channel instead of 3
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.maxpool = torch.nn.Identity()  # Skip the max pooling layer

    # 冻结ResNet的前几层（例如冻结前面的18层）
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层以适应您的数据集类别数
    num_classes = 10
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)  # 只优化最后一层
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return model, optimizer, scheduler

def initialize_res50(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """
    # Load a pretrained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)

    # Modify the first convolutional layer to accept 1 input channel instead of 3
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Replace the final fully connected layer to fit the number of classes in your dataset
    num_classes = 10
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # **Freeze the parameters of the earlier layers**
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Unfreeze the parameters of the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Unfreeze the parameters of all BatchNorm layers (important for training)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True

    # Use the SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Return the model and optimizer
    return model, optimizer, scheduler