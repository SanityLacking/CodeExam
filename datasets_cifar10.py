import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
# import ssl
# # prevents ssl error on windows machines.
# ssl._create_default_https_context = ssl._create_unverified_context


# Define the data transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the training set
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
val_size = int(len(train_dataset) * 0.1)
train_size = int(len(train_dataset)*0.9)

train_dataset, validation_dataset = random_split(train_dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=0)

validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32,
                                          shuffle=False, num_workers=0)

# Load the test set
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                         shuffle=False, num_workers=0)

# Define the classes of the dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

