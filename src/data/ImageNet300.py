import os

import numpy as np
import PIL.Image
import pandas as pd

import torch
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from src.utils.getimagenetclasses import get_classes, parsesynsetwords, parseclasslabel

class ImageNet300Dataset(Dataset):
    def __init__(self, root_dir, xmllabeldir, synsetfile, maxnum, transform=None):

        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.xmllabeldir = xmllabeldir
        self.transform = transform
        self.imgfilenames = []
        self.labels = []
        self.ending=".JPEG"

        self.clsdict=get_classes()


        indicestosynsets, self.synsetstoindices, synsetstoclassdescr = parsesynsetwords(synsetfile)


        for root, dirs, files in os.walk(self.root_dir):
            for ct,name in enumerate(files):
                nm=os.path.join(root, name)
                #print(nm)
                if (maxnum >0) and ct>= (maxnum):
                    break
                self.imgfilenames.append(nm)
                label,firstname=parseclasslabel(self.filenametoxml(nm) ,self.synsetstoindices)
                self.labels.append(label)

   
    def filenametoxml(self,fn):
        f=os.path.basename(fn)
        
        if not f.endswith(self.ending):
            print('not f.endswith(self.ending)')
            exit()
        
        f=f[:-len(self.ending)]+'.xml'
        f=os.path.join(self.xmllabeldir,f) 
        
        return f


    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')

        label=self.labels[idx]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

        return sample
