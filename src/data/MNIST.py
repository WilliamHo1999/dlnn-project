import os
import json
import PIL.Image
from torchvision import transforms
import torchvision

import torch
from torch.utils.data import Dataset, DataLoader


class MNISTBoundingBoxDataset(Dataset):
    def __init__(self, root_dir, train = True, transform=transforms.Compose([transforms.ToTensor()])):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.train = train

        self.data = torchvision.datasets.MNIST(root = 'Data/', train = train, download= True, transform=transform)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        label = torch.tensor(label)
        
        bounding_box = self.get_bounding_box(image)
        
        return {'image': image, 'label': label, 'bounding_box': bounding_box}


    
    def get_bounding_box(self, image):
        
        binary = image  > 0
        non_zero = torch.nonzero(binary.squeeze(), as_tuple=True)

        x_min, x_max = non_zero[1].min(), non_zero[1].max()
        y_min, y_max = non_zero[0].min(), non_zero[0].max()

        return [x_min, y_min, x_max, y_max]
    
    
def get_mnist_bounding_box_data_loader(root_dir, batch_size = 128):
    
    mnist_train = MNISTBoundingBoxDataset(root_dir = root_dir, train = True)
    mnist_test = MNISTBoundingBoxDataset(root_dir = root_dir, train = False)
    
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    return dataloaders