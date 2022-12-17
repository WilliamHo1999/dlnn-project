import numpy as np
import torch
import scipy.stats
from bs4 import BeautifulSoup as bs
import os

import matplotlib.pyplot as plt

# credit: https://nbviewer.org/gist/Teagum/460a508cda99f9874e4ff828e1896862
def get_hellinger_distance(dist1, dist2):
    """Hellinger distance between distributions"""
    return np.sqrt(sum([ (np.sqrt(p_i) - np.sqrt(q_i))**2 for p_i, q_i in zip(dist1, dist2) ]) / 2)

def hellinger_distance_vec(tensor1, tensor2):
    """ tensors are of type (num_samples X w X h) where (w X h) is the distribution """
    print(tensor1.size(), tensor2.size())
    if len(tensor1.size()) == 2:
        tensor1 = torch.unsqueeze(tensor1,0)
    if len(tensor2.size()) == 2:
        tensor2 = torch.unsqueeze(tensor2,0)
    squared_diff = torch.sub(torch.sqrt(tensor1), torch.sqrt(tensor2))**2
    #print(squared_diff)
    return torch.sqrt(torch.sum(squared_diff, dim=[-1,-2], keepdim=True) / 2) 

def get_bhattacharyya_distance(dist1,dist2):
    return -np.log(sum([np.sqrt(p_i*q_i) for p_i, q_i in zip(dist1, dist2) ] ))


def get_spearman_rank_coefficient(dist1, dist2):
    return scipy.stats.spearmanr(dist1, dist2, axis=None)

#tensor1 = torch.randint(5,(1,3,2,2))
#tensor2 = torch.randint(5,(1,3,2,2))


# for a single image
def get_bounding_boxes(filename, dir=None):
    if dir is not None:
        fp = os.path.join(dir, filename)
    with open(fp, 'r') as f:
        data = f.read()
    bs_data = bs(data, "xml")
    objects = bs_data.find_all("object")
    bounding_boxes = bs_data.find_all("bndbox")
    xmins = bs_data.find_all("xmin")
    ymins = bs_data.find_all("ymin")
    xmaxs = bs_data.find_all("xmax")
    ymaxs = bs_data.find_all("ymax")
    boxes = []
    for i in range(len(bounding_boxes)):
        box_i = {"xmin": int(xmins[i].text),
            "ymin": int(ymins[i].text),
            "xmax": int(xmaxs[i].text), 
            "ymax": int(ymaxs[i].text)}
        boxes.append(box_i)
    return boxes

# for a single image + heatmap
def get_weighted_iou(heatmap, bounding_boxes):
    bnds_map = np.zeros_like(heatmap)
    for bbox in bounding_boxes:
        bnds_map[bbox["ymin"]:bbox["ymax"],bbox["xmin"]:bbox["xmax"]] = 1
    #intersection = heatmap[bnds_map == 1] # mask heatmap to bounding box
    intersection = heatmap.copy()
    intersection[bnds_map != 1] = 0 # disable everything outside bounding box (0 inside bbox remain)
    print(intersection.shape)
    union = heatmap.copy()
    union[bnds_map == 1] = 1 # set 1 everyhwere where bounding boxes are, remaining entries remain (incl. > 0 ones)
    print(union.shape)
    weighted_iou = float(np.sum(intersection)) / np.sum(union)
    return weighted_iou, intersection, union

# for multiple images
def _get_bounding_maps(size, bounding_boxes_lists):
    '''bounding_boxes_lists: num_samples-sized list of lists of dicts with the four coordinates for a bounding box
        size: maximum size of all images'''
    bounding_maps = torch.zeros((len(bounding_boxes_lists),*size))
    for i_img in range(len(bounding_boxes_lists)):
        for bbox in bounding_boxes_lists[i_img]:
            print(bbox)
            bounding_maps[i_img, bbox["ymin"]:bbox["ymax"],bbox["xmin"]:bbox["xmax"]] = 1
    return bounding_maps

# for multiple images
def get_weighted_iou_vec(heatmaps, bounding_maps):
    ''' heatmaps: (num_samples X w X h)
        bounding_maps: (num_samples X 4) where 2nd dim is 1d array ymin,ymax,xmin,xmax'''
    intersections = torch.clone(heatmaps)
    intersections[bounding_maps != 1] = 0
    unions = torch.clone(heatmaps)
    unions[bounding_maps == 1] = 1
    weighted_ious = torch.div(torch.sum(intersections, [-1,-2], keepdim=True), torch.sum(unions, [-1,-2], keepdim=True))
    return weighted_ious


def _expand_with_zeros(tensor, dim, target_dim_size):
    '''expand four dimensional tensor to target size in one dimension by filling with zeros'''
    tens_size = tensor.size()
    target_size = tensor.size()
    target_size[dim] = target_dim_size
    out = torch.zeros(target_size)
    out[0:tens_size[0], 0:tens_size[1], 0:tens_size[2], 0:tens_size[3]] = tensor

# for multiple images each with multiple heatmaps (but only one bounding_map)
def get_weighted_iou_mult_class_vec(heatmaps, bounding_maps):
    ''' heatmaps: (num_samples X num_all_classes X w X h) or (num_samples X num_selected_classes X w X h)
        bounding_maps: (num_samples X 4) where 2nd dim is 1d array ymin,ymax,xmin,xmax'''
    bounding_maps = torch.unsqueeze(bounding_maps,1)
    bounding_maps = bounding_maps.expand_as(heatmaps)
    print("bounding_maps", bounding_maps.size())
    intersections = torch.clone(heatmaps)
    intersections[bounding_maps != 1] = 0
    print("intersections", intersections.size())
    unions = torch.clone(heatmaps)
    unions[bounding_maps == 1] = 1
    print("unions", unions.size())
    weighted_ious_per_heatmap = torch.div(torch.sum(intersections, [-1,-2], keepdim=True), torch.sum(unions, [-1,-2], keepdim=True))
    weighted_ious_per_heatmap = torch.squeeze(weighted_ious_per_heatmap)
    print("weighted_ious_per_heatmap", weighted_ious_per_heatmap.size())
    print(weighted_ious_per_heatmap)
    zero_mask = torch.zeros_like(weighted_ious_per_heatmap)
    print(weighted_ious_per_heatmap>zero_mask)
    print(weighted_ious_per_heatmap[...,0:]>0)
    print(weighted_ious_per_heatmap[...,0:]>0)
    weighted_ious_per_heatmap[weighted_ious_per_heatmap==0] = float('nan')
    print("weighted_ious_per_heatmap", weighted_ious_per_heatmap.size())
    print(weighted_ious_per_heatmap)
    #weighted_ious_per_img = torch.nanmean(weighted_ious_per_heatmap[weighted_ious_per_heatmap[...,-1:]>0], dim=-1)
    weighted_ious_per_img = torch.nanmean(weighted_ious_per_heatmap, dim=-1)
    print("weighted_ious_per_img", weighted_ious_per_img.size())
    return weighted_ious_per_img

dir = os.path.join(os.path.dirname( __file__ ),"../../Data/VOC2012/Annotations")
boxes1 = get_bounding_boxes("2007_000032.xml", dir= dir)
heatmap1 = np.random.rand(486,500)
iou, i, u = get_weighted_iou(heatmap1, boxes1)

#plt.imshow(heatmap)
print(iou)
#plt.imshow(i)
#plt.imshow(u)


boxes2 = get_bounding_boxes("2007_000027.xml", dir=dir)

bounding_maps = _get_bounding_maps((500,500), [boxes1, boxes2])
print(bounding_maps.size())

fig, ax = plt.subplots(len(bounding_maps))
for j_bounding_map in range(len(bounding_maps)):
    ax[j_bounding_map].imshow(bounding_maps[j_bounding_map])

heatmaps_org = torch.rand(2,2, 500,500)
zeros = torch.zeros(2,4,500,500)
heatmaps = zeros.clone()
heatmaps[0:2,0:2,:500,:500] = heatmaps_org
print("HEATMAPS", heatmaps)
ious = get_weighted_iou_mult_class_vec(heatmaps,bounding_maps)
print("ious", ious.size(), ious)

plt.show()

# TODO: different image sizes

