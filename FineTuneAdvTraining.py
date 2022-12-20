import os
import json
import PIL.Image
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from src.data.ImageNet100ClassesDataset import ImageNet100ClassesDataset, prepare_dataloaders_ImageNet100ClassesDataset
from src.attacks.attacks import FastGradientSign, ProjectedGradientDescent
from src.training.Trainer import Trainer


def seed_everything(seed_value=4995):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(num_classes = 100, model_path = None, to_cuda = True):
    if not model_path:
        model = torchvision.models.resnet18(pretrained = True)
        input_feat = model.fc.in_features
        
        model.fc = nn.Linear(input_feat, num_classes)
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

    seed_everything()
    root_dir = 'Data/imagenet100classes'

    config = {
        'lr': 0.001,
        'batch_size': 64,
        'weight_decay': 0.01
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    resnet, loaded_state_dict = load_model(model_path = 'models/resnet_100_imagenet_adv_training.pt')

    for name, param in resnet.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False


    train_loader, val_loader = prepare_dataloaders_ImageNet100ClassesDataset(root_dir, batch_size = config['batch_size'])

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    loss_fn = nn.CrossEntropyLoss()

    train_params = [p for p in resnet.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(train_params, lr = config['lr'], weight_decay = config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5], gamma=0.1)

    training_params = {
        'dataloaders': dataloaders,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

    classes = list(train_loader.dataset.class_name_dict.values())


    pgd = ProjectedGradientDescent(resnet, loss_fn, iterations = 20, epsilon = 0.25, return_logits=False)


    trainer = Trainer(resnet, loss_fn, classes, training_params, DEVICE, num_epochs = 1, model_name = 'resnet_100_imagenet_adv_training', save_model = True, model_dir = 'models', adversarial_training = True, adversarial_attack = pgd)


    trainer.train()
    

    # Fine-tune all layers
    resnet, loaded_state_dict = load_model(model_path = 'models/resnet_100_imagenet_fine_tuned.pt')

    for name, param in resnet.named_parameters():
        param.requires_grad = True
    
    train_loader, val_loader = prepare_dataloaders_ImageNet100ClassesDataset(root_dir, batch_size = config['batch_size'])
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    loss_fn = nn.CrossEntropyLoss()

    train_params = [p for p in resnet.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(train_params, lr = 0.00001, weight_decay = config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5], gamma=0.1)

    training_params = {
        'dataloaders': dataloaders,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

    classes = list(train_loader.dataset.class_name_dict.values())
    pgd = ProjectedGradientDescent(resnet, loss_fn, iterations = 5, alpha = 0.02, epsilon = 0.01, return_logits=False)

    trainer_fine_tune = Trainer(resnet, loss_fn, classes, training_params, DEVICE, num_epochs = 6, model_name = 'resnet_100_imagenet_adv_training_from_scratch', save_model = True, model_dir = 'models', adversarial_training = True, adversarial_attack = pgd)

    trainer_fine_tune.train()
    
    
    print("Losses:",trainer_fine_tune.epoch_loss)
    print("Accuracies:",trainer_fine_tune.epoch_acc)