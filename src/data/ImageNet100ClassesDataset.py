import os
import json
import PIL.Image
from torchvision import transforms

import torch
from torch.utils.data import Dataset, DataLoader

class ImageNet100ClassesDataset(Dataset):
    
    def __init__(self, root_dir, train_val = 0, transform=None, full_imagenet_classes = None):
        
        self.root_dir = root_dir
        f = open(os.path.join(root_dir, 'Labels.json'))
        self.class_name_dict = json.load(f)
        
        if not full_imagenet_classes:
            self.class_to_label = {}
            self.label_to_class = {}
            for cls_idx, class_name in enumerate(self.class_name_dict.keys()):
                self.class_to_label[class_name] = cls_idx
                self.label_to_class[cls_idx] = class_name
        else:
            reverse_class_name_dict = {v: k for k, v in self.class_name_dict.items()}
    
            self.label_to_class_name = full_imagenet_classes
            self.label_to_class = {}
            self.class_to_label = {}
            new_class_name_dict = {}
            
            for cls_idx, class_name in self.label_to_class_name.items():
                
                if class_name in reverse_class_name_dict.keys():
                    if cls_idx > 500:
                        continue
                    self.class_to_label[reverse_class_name_dict[class_name]] = cls_idx
                    self.label_to_class[cls_idx] = reverse_class_name_dict[class_name]
                    new_class_name_dict[reverse_class_name_dict[class_name]] = class_name
            
            self.class_name_dict = new_class_name_dict

        
        self.transform = transform
        
        self.file_names = []
        self.labels = []
        
        self.train_val = ("train" if train_val == 0 else "val")
        
        for root, dirs, files in os.walk(self.root_dir):
            if f'imagenet100classes/{self.train_val}.X' in root:
                class_name = root.split('/')[-1]
                if 'ipynb' in class_name:
                    continue
                for ct, name in enumerate(files):
                    self.file_names.append(os.path.join(root, name))
                    self.labels.append(self.class_to_label[class_name])
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        
        image_file_name = self.file_names[idx]
        label = self.labels[idx]
        
        image = PIL.Image.open(image_file_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'filename': image_file_name}

        return sample
    

def prepare_dataloaders_ImageNet100ClassesDataset(root_dir, batch_size = 64, full_imagenet_classes = None):
    
    train_transforms = transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    
    val_transforms = transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    

    train_dataset = ImageNet100ClassesDataset(root_dir, 0, transform = train_transforms, full_imagenet_classes = full_imagenet_classes)
    val_dataset = ImageNet100ClassesDataset(root_dir, 1, transform = val_transforms, full_imagenet_classes = full_imagenet_classes)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    
    return train_loader, val_loader