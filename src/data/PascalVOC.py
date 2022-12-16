import os

import numpy as np
import PIL.Image
import pandas as pd

import torch
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader



class dataset_voc(Dataset):
    def __init__(self, root_dir, trvaltest, transform=None):

        self.root_dir = root_dir

        self.transform = transform
        self.imgfilenames=[]
        self.labels=[]

        pv=PascalVOC(root_dir)
        cls=pv.list_image_sets()

        if trvaltest==0:
            dataset='train'
        elif trvaltest==1:
            dataset='val'
        else:
            print('Not a split')
            exit()

        
        filenamedict={}
        for c,cat_name in enumerate(cls):
            imgstubs=pv.imgs_from_category_as_list(cat_name, dataset)
            for st in imgstubs:
                if st in filenamedict:
                    filenamedict[st][c]=1
                else:
                    vals=np.zeros(20,dtype=np.int32)
                    vals[c]=1
                    filenamedict[st]=vals


        self.labels=np.zeros((len(filenamedict),20))
        tmpct=-1
        for key,value in filenamedict.items():
            tmpct+=1
            self.labels[tmpct,:]=value

            fn=os.path.join(self.root_dir,'JPEGImages',key+'.jpg')
            self.imgfilenames.append(fn)


    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
        label = self.labels[idx,:].astype(np.float32)

        if self.transform:
            image = self.transform(image)

        if image.size()[0]==1:
            image=image.repeat([3,1,1])

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

        return sample


class PascalVOC:
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values
    
    
def prepare_dataloaders_pascal_voc(root_dir, batch_size = 64):
    
    data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize((256,256)),
          transforms.CenterCrop(256),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

    image_datasets={}
    image_datasets['train']= dataset_voc(root_dir=root_dir, trvaltest=0, transform = data_transforms['train'])

    image_datasets['val']= dataset_voc(root_dir=root_dir, trvaltest=1, transform = data_transforms['val'])


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=1)
    
    classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']
        
    return dataloaders, classes