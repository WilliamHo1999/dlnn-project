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
from src.attacks.attacks import FastGradientSign, ProjectedGradientDescent
from src.training.Trainer import Trainer
from src.optim.scheduler import CustomScheduler


def seed_everything(seed_value=4995):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    seed_everything()
    root_dir = 'Data/PascalVOC/VOCdevkit/VOC2012'

    config = {
        'lr': 0.0001,
        'batch_size': 64,
        'weight_decay': 0.01
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = 'models/pascal_voc_resnet_fine_tuned.pt'

    # Fine-tune all layers
    resnet, loaded_state_dict = load_model(model_path = model_path)

    for name, param in resnet.named_parameters():
        param.requires_grad = True

    dataloaders, classes = prepare_dataloaders_pascal_voc(root_dir, batch_size = config['batch_size'])

    loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    train_params = [p for p in resnet.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(train_params, lr = config['lr'], weight_decay = config['weight_decay'])
    #scheduler = CustomScheduler(optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,6], gamma=0.1)

    training_params = {
        'dataloaders': dataloaders,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    

    pgd = ProjectedGradientDescent(resnet, loss_fn, iterations = 5, alpha = 2/255, epsilon = 2/255, return_logits=False)

    trainer = Trainer(
        model = resnet,
        loss_fn = loss_fn,
        classes = classes,
        num_epochs = 10,
        model_name = f'pascal_voc_resnet_adv_fine_Tune_small_ball_msls',
        training_params = training_params,
        multi_label = True,
        print_frequency = 10,
        adversarial_training = True,
        adversarial_attack = pgd,
        custom_scheduler = True
    )

    trainer.train()


    print("Losses:",trainer_fine_tune.epoch_loss)
    print("Accuracies:",trainer_fine_tune.epoch_mape)
