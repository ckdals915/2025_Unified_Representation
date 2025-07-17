import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms



def get_dataloaders_MNIST(batch_size, SEED, num_workers=4):
    # Download and prepare the MNIST dataset
    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Set the random seed for reproducibility
    g = torch.Generator()
    g.manual_seed(SEED)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              generator=g,
                              
                              )
    test_loader  = DataLoader(test_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              
                              )
    return train_loader, test_loader


def get_dataloaders_CIFAR100(batch_size, num_workers=4):
    # Download and prepare the CIFAR-100 dataset
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((32,32), scale=(0.05,1.0)),
        transforms.RandomHorizontalFlip(),          
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.CIFAR100(root='./data', train=True,
                                download=True, transform=transform_train)
    test_ds  = datasets.CIFAR100(root='./data', train=False,
                                download=True, transform=transform_test)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


class CWRUDataset(Dataset):
    def __init__(self):
        self.data = np.load('./data/CWRU/CWRU_Numpy_Data.npy')

    def __getitem__(self, index):
        # Input X
        x_np = self.data[index][:2048]
        x_np = x_np.reshape(1, 2048)

        # One-hot encoded label
        y_onehot = self.data[index][2048:]
        y_int = int(np.argmax(y_onehot))

        X = torch.from_numpy(x_np)
        y = torch.tensor(y_int, dtype=torch.long)

        return X, y

    def __len__(self):
        return len(self.data)

def get_dataloaders_CWRU(batch_size, num_workers=4):
    full_dataset = CWRUDataset()
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_valid = n_total - n_train

    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_valid])
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers,
                              pin_memory=True,
                             ) 
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers,
                              pin_memory=True,
                             )
    
    return train_loader, valid_loader

def get_dataloaders_CIFAR100(batch_size, num_workers=4):
    # Download and prepare the CIFAR-100 dataset
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((32,32), scale=(0.05,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True,
                                download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR100(root='./data', train=False,
                                download=True, transform=transform_test)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader
    
    
    