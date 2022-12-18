import os
import json
import PIL.Image
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from src.data.PascalVOC import dataset_voc, prepare_dataloaders_pascal_voc
from src.attacks.attacks import FastGradientSign, ProjectedGradientDescent, UniversalPerturbation
from src.training.Trainer import Trainer

def load_model(num_classes = 20, model_path = None, to_cuda = True):
    if not model_path:
        model = torchvision.models.resnet18(pretrained = True)
        input_feat = model.fc.in_features
        
        model.fc = nn.Linear(input_feat, num_classes)
        model.fc.reset_parameters()
        loaded_state_dict = False
    
    else:
        print("Loaded", model_path)
        model = torchvision.models.resnet18()
        input_feat = model.fc.in_features
        model.fc = nn.Linear(input_feat, num_classes)
        loaded_model = torch.load(model_path)
        model.load_state_dict(loaded_model['model_state_dict'])
        loaded_state_dict = True
        
    if to_cuda:
        model = model.to('cuda')
        
    return model, loaded_state_dict


if __name__ == "__main__":

    root_dir = 'Data/PascalVOC/VOCdevkit/VOC2012'

    resnet, loaded_state_dict = load_model(model_path = 'models/pascal_voc_resnet_fine_tuned.pt')

    dataloaders, classes = prepare_dataloaders_pascal_voc(root_dir, batch_size = 500)


    loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    up = UniversalPerturbation(resnet, loss_fn)

    for samp in dataloaders['train']:
        break


    img = samp['image'].to('cuda')
    label = samp['label'].to('cuda')


    v = up.compute_perturbation(inputs = img, targets = label, max_iter = 100)