import os
import json
import PIL.Image

import torch
from torch.utils.data import Dataset

class ImageNet100ClassesDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        f = open(os.path.join(root_dir, 'Labels.json'))
        self.class_name_dict = json.load(f)
        
        self.transform = transform
        
        self.file_names = []
        self.labels = []
        
        
        for root, dirs, files in os.walk(self.root_dir):
            if 'imagenet300classes/train.X' in root:
                class_name = root.split('/')[-1]
                for ct, name in enumerate(files):
                    self.file_names.append(os.path.join(root, name))
                    self.labels.append(class_name)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        image_file_name = self.file_names[idx]
        label = self.labels[idx]
        
        image = PIL.Image.open(image_file_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'filename': image_file_name}

        return sample