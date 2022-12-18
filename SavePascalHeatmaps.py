import os
import json
import PIL.Image
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from src.data.PascalVOC import dataset_voc, prepare_dataloaders_pascal_voc
from src.attacks.attacks import FastGradientSign, ProjectedGradientDescent
from src.training.Trainer import Trainer

from src.explainability.GradCam import GradCam
from src.utils.ImageDisplayerGradCam import ImageDisplayerGradCam



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


def exp_1(root_dir = 'Data/PascalVOC/VOCdevkit/VOC2012'):
    resnet, loaded_state_dict = load_model(model_path = 'models/pascal_voc_resnet_fine_tuned.pt')
    dataloaders, classes = prepare_dataloaders_pascal_voc(root_dir, batch_size = 1)
    val_loader = dataloaders['val']
    
    target_layer = resnet.layer4[-1].conv2
    cam = GradCam(resnet, target_layer, 20, multi_label = True, print_things = False)

    image_dispalyer = ImageDisplayerGradCam(resnet, 
            cam, 
            classes,
            reshape = transforms.Resize((256,256)), 
            multi_label = False, 
            image_dir = 'image_net_dir',
            pdf = False, suppress_show = True)
    
    normal_pred_heatmap = np.zeros((0, 256, 256))
    normal_true_heatmap = np.zeros((0, 256, 256))
    num_preds_pre_sample = np.zeros((0))
    num_labels_per_sample = np.zeros((0))
    all_preds = np.zeros((0))
    all_labels = np.zeros((0))
    
    for i, sample in enumerate(val_loader):
        all_heatmaps_pred, num_preds, preds, all_heatmaps_label, num_labels, labels = image_dispalyer.return_heatmap_all(sample, return_heatmap = True)

        normal_pred_heatmap = np.append(normal_pred_heatmap, all_heatmaps_pred, axis = 0)
        normal_true_heatmap = np.append(normal_true_heatmap, all_heatmaps_label, axis = 0)
        num_preds_pre_sample = np.append(num_preds_pre_sample, num_preds)
        num_labels_per_sample = np.append(num_labels_per_sample, num_labels)
        all_preds = np.append(all_preds, preds)
        all_labels = np.append(all_labels, labels)

            
        if i % 100 == 0:
            print(i, len(val_loader))
    
    
    with open('heatmaps/pascal_normal_net_normal_input.npy', 'wb') as f:
        np.save(f, normal_pred_heatmap)
        np.save(f, normal_true_heatmap)
        np.save(f, num_preds_pre_sample)
        np.save(f, num_labels_per_sample)
        np.save(f, all_preds)
        np.save(f, all_labels)
        
def exp_2(root_dir = 'Data/PascalVOC/VOCdevkit/VOC2012'):
    resnet, loaded_state_dict = load_model(model_path = 'models/pascal_voc_resnet_fine_tuned.pt')
    dataloaders, classes = prepare_dataloaders_pascal_voc(root_dir, batch_size = 1)
    val_loader = dataloaders['val']
    
    target_layer = resnet.layer4[-1].conv2
    cam = GradCam(resnet, target_layer, 20, multi_label = True, print_things = False)

    image_dispalyer = ImageDisplayerGradCam(resnet, 
            cam, 
            classes,
            reshape = transforms.Resize((256,256)), 
            multi_label = False, 
            image_dir = 'image_net_dir',
            pdf = False, suppress_show = True)
    
    loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    pgd = ProjectedGradientDescent(resnet, loss_fn, iterations = 5, alpha = 2/255, epsilon = 2/255, return_logits=False)

    normal_perturbed_pred_heatmap = np.zeros((0, 256, 256))
    normal_perturbed_true_heatmap = np.zeros((0, 256, 256))
    normal_perturbed_num_preds_pre_sample = np.zeros((0))
    normal_perturbed_num_labels_per_sample = np.zeros((0))
    normal_perturbed_all_preds = np.zeros((0))
    normal_perturbed_all_labels = np.zeros((0))
    
    for i, sample in enumerate(val_loader):
        pert_img = pgd(sample['image'], sample['label'], random_start = True, compute_original_prediction = False, compute_new_preds = False)
        pert_sample =  {'image': pert_img, 'label': sample['label'], 'filename': sample['filename']}
        all_heatmaps_pred, num_preds, preds, all_heatmaps_label, num_labels, labels = image_dispalyer.return_heatmap_all(pert_sample, return_heatmap = True)

        normal_perturbed_pred_heatmap = np.append(normal_perturbed_pred_heatmap, all_heatmaps_pred, axis = 0)
        normal_perturbed_true_heatmap = np.append(normal_perturbed_true_heatmap, all_heatmaps_label, axis = 0)
        normal_perturbed_num_preds_pre_sample = np.append(normal_perturbed_num_preds_pre_sample, num_preds)
        normal_perturbed_num_labels_per_sample = np.append(normal_perturbed_num_labels_per_sample, num_labels)
        normal_perturbed_all_preds = np.append(normal_perturbed_all_preds, preds)
        normal_perturbed_all_labels = np.append(normal_perturbed_all_labels, labels)

        
        if i % 100 == 0:
            print(i, len(val_loader))
    
    with open('heatmaps/pascal_normal_net_pert_input.npy', 'wb') as f:
        np.save(f, normal_perturbed_pred_heatmap)
        np.save(f, normal_perturbed_true_heatmap)
        np.save(f, normal_perturbed_num_preds_pre_sample)
        np.save(f, normal_perturbed_num_labels_per_sample)
        np.save(f, normal_perturbed_all_preds)
        np.save(f, normal_perturbed_all_labels)
        
def exp_3(root_dir = 'Data/PascalVOC/VOCdevkit/VOC2012'):
    resnet, loaded_state_dict = load_model(model_path = 'models/pascal_voc_resnet_adv_fine_Tune_small_ball_msls.pt')
    dataloaders, classes = prepare_dataloaders_pascal_voc(root_dir, batch_size = 1)
    val_loader = dataloaders['val']
    
    target_layer = resnet.layer4[-1].conv2
    cam = GradCam(resnet, target_layer, 20, multi_label = True, print_things = False)

    image_dispalyer = ImageDisplayerGradCam(resnet, 
            cam, 
            classes,
            reshape = transforms.Resize((256,256)), 
            multi_label = False, 
            image_dir = 'image_net_dir',
            pdf = False, suppress_show = True)
    
    adv_normal_pred_heatmap = np.zeros((0, 256, 256))
    adv_normal_true_heatmap = np.zeros((0, 256, 256))
    adv_normal_num_preds_pre_sample = np.zeros((0))
    adv_normal_mum_labels_per_sample = np.zeros((0))
    adv_normal_all_preds = np.zeros((0))
    adv_normal_all_labels = np.zeros((0))
    
    for i, sample in enumerate(val_loader):
        all_heatmaps_pred, num_preds, preds, all_heatmaps_label, num_labels, labels = image_dispalyer.return_heatmap_all(sample, return_heatmap = True)

        adv_normal_pred_heatmap = np.append(adv_normal_pred_heatmap, all_heatmaps_pred, axis = 0)
        adv_normal_true_heatmap = np.append(adv_normal_true_heatmap, all_heatmaps_label, axis = 0)
        adv_normal_num_preds_pre_sample = np.append(adv_normal_num_preds_pre_sample, num_preds)
        adv_normal_mum_labels_per_sample = np.append(adv_normal_mum_labels_per_sample, num_labels)
        adv_normal_all_preds = np.append(adv_normal_all_preds, preds)
        adv_normal_all_labels = np.append(adv_normal_all_labels, labels)


        if i % 100 == 0:
            print(i, len(val_loader))
            
    with open('heatmaps/pascal_robust_net_normal_input.npy', 'wb') as f:
        np.save(f, adv_normal_pred_heatmap)
        np.save(f, adv_normal_true_heatmap)
        np.save(f, adv_normal_num_preds_pre_sample)
        np.save(f, adv_normal_mum_labels_per_sample)
        np.save(f, adv_normal_all_preds)
        np.save(f, adv_normal_all_labels)
        
def exp_4(root_dir = 'Data/PascalVOC/VOCdevkit/VOC2012'):
    resnet, loaded_state_dict = load_model(model_path = 'models/pascal_voc_resnet_adv_fine_Tune_small_ball_msls.pt')
    dataloaders, classes = prepare_dataloaders_pascal_voc(root_dir, batch_size = 1)
    val_loader = dataloaders['val']
    
    target_layer = resnet.layer4[-1].conv2
    cam = GradCam(resnet, target_layer, 20, multi_label = True, print_things = False)

    image_dispalyer = ImageDisplayerGradCam(resnet, 
            cam, 
            classes,
            reshape = transforms.Resize((256,256)), 
            multi_label = False, 
            image_dir = 'image_net_dir',
            pdf = False, suppress_show = True)

    adv_pert_pred_heatmap = np.zeros((0, 256, 256))
    adv_pert_true_heatmap = np.zeros((0, 256, 256))
    adv_pert_num_preds_pre_sample = np.zeros((0))
    adv_pert_mum_labels_per_sample = np.zeros((0))
    adv_pert_all_preds = np.zeros((0))
    adv_pert_all_labels = np.zeros((0))
    
    loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    pgd = ProjectedGradientDescent(resnet, loss_fn, iterations = 5, alpha = 2/255, epsilon = 2/255, return_logits=False)

    
    for i, sample in enumerate(val_loader):
        pert_img = pgd(sample['image'], sample['label'], random_start = True, compute_original_prediction = False, compute_new_preds = False)
        pert_sample =  {'image': pert_img, 'label': sample['label'], 'filename': sample['filename']}
        all_heatmaps_pred, num_preds, preds, all_heatmaps_label, num_labels, labels = image_dispalyer.return_heatmap_all(pert_sample, return_heatmap = True)

        adv_pert_pred_heatmap = np.append(adv_pert_pred_heatmap, all_heatmaps_pred, axis = 0)
        adv_pert_true_heatmap = np.append(adv_pert_true_heatmap, all_heatmaps_label, axis = 0)
        adv_pert_num_preds_pre_sample = np.append(adv_pert_num_preds_pre_sample, num_preds)
        adv_pert_mum_labels_per_sample = np.append(adv_pert_mum_labels_per_sample, num_labels)
        adv_pert_all_preds = np.append(adv_pert_all_preds, preds)
        adv_pert_all_labels = np.append(adv_pert_all_labels, labels)

        
        if i % 100 == 0:
            print(i, len(val_loader))
            

    with open('heatmaps/pascal_robust_net_perturb_input.npy', 'wb') as f:
        np.save(f, adv_pert_pred_heatmap)
        np.save(f, adv_pert_true_heatmap)
        np.save(f, adv_pert_num_preds_pre_sample)
        np.save(f, adv_pert_mum_labels_per_sample)
        np.save(f, adv_pert_all_preds)
        np.save(f, adv_pert_all_labels)

        
if __name__ == "__main__":

    seed_everything()

    root_dir = 'Data/PascalVOC/VOCdevkit/VOC2012'
    print("exp_1")

   # exp_1()
    print("exp_2")
    
   # exp_2()
    print("exp_3")

    exp_3()
    print("exp_4")

    exp_4()
    
    