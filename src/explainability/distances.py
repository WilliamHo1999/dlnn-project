import numpy as np
import torch
import scipy.stats
from bs4 import BeautifulSoup as bs
import os
import albumentations as A

import PIL.Image
from torchvision import transforms
import matplotlib.pyplot as plt


def get_hellinger_distance(dist1, dist2):
    """Hellinger distance between distributions"""
    # credit: https://nbviewer.org/gist/Teagum/460a508cda99f9874e4ff828e1896862
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
def get_bounding_boxes(filename, dir=None, return_dict=True):
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
    size = (int(bs_data.find_all("height")[0].text), int(bs_data.find_all("width")[0].text))
    for i in range(len(bounding_boxes)):
        box_i = {"xmin": int(xmins[i].text),
            "ymin": int(ymins[i].text),
            "xmax": int(xmaxs[i].text), 
            "ymax": int(ymaxs[i].text)}
        if return_dict:
            boxes.append(box_i)
        else:
            boxes.append(list(box_i.values()))
    return boxes, size

# for a multiple image
def get_bounding_boxes_vec(filenames, dir=None, return_dict=True):
    bboxes_list = []
    size_list = []
    for filename in filenames:
        bboxes, size = get_bounding_boxes(filename, dir=dir, return_dict=return_dict)
        bboxes_list.append(bboxes)
        size_list.append(size)
    return bboxes_list, size_list

# for a multiple image
def resize_bounding_boxes(bboxes_list, size_list, target_width=256,target_height=256):
    ''' param: bboxes_list : list of (num_samples) lists each with (sample_i_num_bboxes) lists of 4 elements [x_min, y_min, x_max, y_max] '''
    res_bboxes_list = []
    transform = A.Compose([
        A.Resize(target_height,target_width),
        A.CenterCrop(target_height, target_width)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["class_labels"]))
    print(["x"]*len(bboxes_list))
    for i_bboxes in range(len(bboxes_list)):
        bboxes = bboxes_list[i_bboxes]
        size = list(size_list[i_bboxes])
        print("size", type(size), size)
        dummy_img = np.zeros((size))
        print(bboxes)
        transformed = transform(bboxes=bboxes, image=dummy_img, class_labels=["x"]*len(bboxes))["bboxes"]
        res_bboxes_list.append([[int(coord) for coord in bbox] for bbox in transformed])
    return res_bboxes_list

# for multiple images
def _get_bounding_maps(size, bounding_boxes_lists, input_dict=True):
    '''bounding_boxes_lists: num_samples-sized list of lists of dicts or arrays with the four coordinates for a bounding box
        size: maximum size of all images'''
    bounding_maps = torch.zeros((len(bounding_boxes_lists),*size))
    for i_img in range(len(bounding_boxes_lists)):
        for bbox in bounding_boxes_lists[i_img]:
            print(bbox)
            if input_dict:
                bounding_maps[i_img, bbox["ymin"]:bbox["ymax"],bbox["xmin"]:bbox["xmax"]] = 1
            else:
                bounding_maps[i_img, bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    return bounding_maps

# for multiple images
def _get_bounding_maps_vec_NEW(max_size, bounding_boxes_lists, max_num_bboxes, input_dict=True):
    '''bounding_boxes_lists: num_samples-sized list of lists of dicts or arrays with the four coordinates for a bounding box
        max_size: maximum size of all images
        max_num_boxes: maximum number of bboxes (classes) present in any image
        output : (num_samples) X (max_boxes) X (max_size[0] X max_size[1])'''
    bounding_maps = torch.zeros((len(bounding_boxes_lists),max_num_bboxes,*max_size))
    for i_img in range(len(bounding_boxes_lists)):
        for j_bbox in range(len(bounding_boxes_lists[i_img])):
            bbox = bounding_boxes_lists[i_img][j_bbox]
            print("bbox", bbox)
            if input_dict:
                bounding_maps[i_img, j_bbox, bbox["ymin"]:bbox["ymax"],bbox["xmin"]:bbox["xmax"]] = 1
            else:
                bounding_maps[i_img, j_bbox, bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    return bounding_maps


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
def get_weighted_iou_vec(heatmaps, bounding_maps):
    ''' heatmaps: (num_samples X w X h)
        bounding_maps: (num_samples X 4) where 2nd dim is 1d array ymin,ymax,xmin,xmax'''
    intersections = torch.clone(heatmaps)
    intersections[bounding_maps != 1] = 0
    unions = torch.clone(heatmaps)
    unions[bounding_maps == 1] = 1
    weighted_ious = torch.div(torch.sum(intersections, [-1,-2], keepdim=True), torch.sum(unions, [-1,-2], keepdim=True))
    return weighted_ious

# for multiple images (4d-tensor), used for heatmaps
def _expand_with_zeros(tensor, dim, target_dim_size):
    '''expand four dimensional tensor to target size in one dimension by filling with zeros'''
    tens_size = tensor.size()
    target_size = list(tensor.size())
    target_size[dim] = target_dim_size
    out = torch.zeros(target_size)
    out = out -0.00001
    out[0:tens_size[0], 0:tens_size[1], 0:tens_size[2], 0:tens_size[3]] = tensor
    #print(out)
    return out

# for multiple images each with multiple heatmaps (but only one bounding_map)
def get_weighted_iou_mult_class_vec(heatmaps, bounding_maps):
    ''' heatmaps: (num_samples X num_all_classes X w X h)
        bounding_maps: (num_samples X 4) where 2nd dim is 1d array ymin,ymax,xmin,xmax'''
    bounding_maps = torch.unsqueeze(bounding_maps,1)
    bounding_maps = bounding_maps.expand_as(heatmaps)
    print("bounding_maps", bounding_maps.size())
    print(bounding_maps)
    intersections = torch.clone(heatmaps)
    intersections[bounding_maps != 1] = 0
    print("intersections", intersections.size())
    print(intersections)
    unions = torch.clone(heatmaps)
    unions[bounding_maps == 1] = 1
    print("unions", unions.size())
    print(unions)
    weighted_ious_per_heatmap = torch.div(torch.sum(intersections, [-1,-2], keepdim=True), torch.sum(unions, [-1,-2], keepdim=True))
    weighted_ious_per_heatmap = torch.squeeze(weighted_ious_per_heatmap)
    print("weighted_ious_per_heatmap", weighted_ious_per_heatmap.size())
    print(weighted_ious_per_heatmap)
    #zero_mask = torch.zeros_like(weighted_ious_per_heatmap)
    #print(weighted_ious_per_heatmap>zero_mask)
    #print(weighted_ious_per_heatmap[...,0:]>0)
    #print(weighted_ious_per_heatmap[...,0:]>0)
    weighted_ious_per_heatmap[weighted_ious_per_heatmap==0] = float('nan')
    print("weighted_ious_per_heatmap", weighted_ious_per_heatmap.size())
    print(weighted_ious_per_heatmap)
    #weighted_ious_per_img = torch.nanmean(weighted_ious_per_heatmap[weighted_ious_per_heatmap[...,-1:]>0], dim=-1)
    weighted_ious_per_img = torch.nanmean(weighted_ious_per_heatmap, dim=-1)
    print("weighted_ious_per_img", weighted_ious_per_img.size())
    return weighted_ious_per_img

def get_weighted_iou_mult_class_NEW(heatmaps_list, bounding_maps_list):
    ''' heatmaps: (num_samples X num_all_classes X w X h)
        bounding_maps: (num_samples X 4) where 2nd dim is 1d array ymin,ymax,xmin,xmax'''
    #bounding_maps = torch.unsqueeze(bounding_maps,1)
    #bounding_maps = bounding_maps.expand_as(heatmaps)
    wious_per_image = []
    for i_img in range(len(heatmaps_list)):
        heatmaps = heatmaps_list[i_img]
        bounding_maps = bounding_maps_list[i_img]
        scores_per_heatmap = []
        for heatmap in heatmaps:
            scores_heatmap_per_bbox = []
            for bounding_map in bounding_maps:
                intersection = torch.clone(heatmap)
                intersection[bounding_map != 1] = 0
                union = torch.clone(heatmaps)
                union[bounding_map == 1] = 1
                hm_bm_weighted_iou = torch.div(torch.sum(intersection, dim=[-1,-2], keepdim=True), torch.sum(union, dim=[-1,-2], keepdim=True))
                scores_heatmap_per_bbox.append(hm_bm_weighted_iou)
            scores_per_heatmap.append(max(scores_heatmap_per_bbox))
        wiou_image = np.mean(scores_per_heatmap)
        wious_per_image.append(wiou_image)
    return wious_per_image

# for multiple images each with multiple heatmaps (but only one bounding_map)
def get_weighted_iou_mult_class_vec_NEW(heatmaps, bounding_maps):
    ''' heatmaps: (num_samples X num_all_classes X width X height)
        bounding_maps: (num_samples X num_max_objects X width X height) where 2nd dim is 1d array ymin,ymax,xmin,xmax'''
    #bounding_maps = torch.unsqueeze(bounding_maps,1)
    num_heatmaps = heatmaps.size()[1]
    num_bounding_maps = bounding_maps.size()[1]
    #lcm = np.lcm(num_heatmaps,num_bounding_maps)
    #print("lcm", lcm)
    #print(lcm/num_bounding_maps)
    #print(lcm/num_heatmaps)
    bounding_maps = bounding_maps.repeat(1,num_heatmaps, 1, 1)
    heatmaps = heatmaps.repeat(1,num_bounding_maps, 1, 1)
    
    #bounding_maps = bounding_maps.expand_as(heatmaps)
    intersections = torch.clone(heatmaps)
    intersections[bounding_maps != 1] = 0
    #print("intersections", intersections.size(), intersections)
    unions = torch.clone(heatmaps)
    unions[bounding_maps == 1] = 1
    #print("unions", unions.size(), unions)
    ###unions = unions.reshape(heatmaps.size()[0], num_heatmaps, num_bounding_maps, heatmaps.size()[-2], heatmaps.size()[-1])
    ###intersections = intersections.reshape(heatmaps.size()[0], num_heatmaps, num_bounding_maps, heatmaps.size()[-2], heatmaps.size()[-1])
    unions = unions.reshape(heatmaps.size()[0], num_bounding_maps, num_heatmaps, heatmaps.size()[-2], heatmaps.size()[-1])
    intersections = intersections.reshape(heatmaps.size()[0], num_bounding_maps, num_heatmaps, heatmaps.size()[-2], heatmaps.size()[-1])
    unions = torch.transpose(unions, 1, 2)
    intersections = torch.transpose(intersections, 1, 2)
    #print(intersections)
    #print(intersections.size())
    weighted_ious_per_bounding_map = torch.div(torch.sum(intersections, [-1,-2], keepdim=True), torch.sum(unions, [-1,-2], keepdim=True))
    #print("weighted_ious_per_bounding_map", weighted_ious_per_bounding_map.size(), weighted_ious_per_bounding_map)
    weighted_ious_per_bounding_map = torch.squeeze(weighted_ious_per_bounding_map)
    #print("weighted_ious_per_bounding_map is nan", weighted_ious_per_bounding_map[torch.isnan(weighted_ious_per_bounding_map)])
    weighted_ious_per_bounding_map[torch.isnan(weighted_ious_per_bounding_map)] = -9999 # fix nan issue with torch.max for nans coming by division by 0
    #print("weighted_ious_per_bounding_map -9999", weighted_ious_per_bounding_map.size(), weighted_ious_per_bounding_map)
    weighted_ious_per_heatmap = torch.max(weighted_ious_per_bounding_map, dim=-1, keepdim=True)[0] #max wiou across all bboxess for each heatmap
    #print(weighted_ious_per_heatmap[torch.min(weighted_ious_per_bounding_map, dim=-1, keepdim=True)[0] < 0])
    weighted_ious_per_heatmap[torch.min(weighted_ious_per_bounding_map, dim=-1, keepdim=True)[0] < 0] = -9999 # so that 0-expanded heatmaps get nan, not recognized in mean
    weighted_ious_per_heatmap[weighted_ious_per_heatmap<0] = float('nan') # set nan so ignored using nanmean
    print("weighted_ious_per_heatmap", weighted_ious_per_heatmap.size(), weighted_ious_per_heatmap)
    weighted_ious_per_heatmap = torch.squeeze(weighted_ious_per_heatmap)
    weighted_ious_per_img = torch.nanmean(weighted_ious_per_heatmap, dim=-1) # wiou per image as average of heatmaps wiou of that image
    #weighted_ious_per_img = torch.nanmean(weighted_ious_per_heatmap[weighted_ious_per_heatmap[...,-1:]>0], dim=-1)
    print("weighted_ious_per_img", weighted_ious_per_img.size(), weighted_ious_per_img)
    return weighted_ious_per_img

# wrapper function
def compute_weighted_iou_mult_class_vec(bb_filenames, bb_dir, heatmaps, transform=True):
    ''' wrapper function to compute the iou
        filenames: list of filenames of bbounding box files
        bb_dir: string directory in which filenames are stored
        heatmaps:  tensor containing heatmaps for all images (num_samples X num_all_classes X w X h)'''
    bboxes_list, size_list = get_bounding_boxes_vec(bb_filenames, dir=bb_dir, return_dict=False)
    if transform:
        bboxes_list = resize_bounding_boxes(bboxes_list, size_list=size_list)
    bounding_maps = _get_bounding_maps((heatmaps.size()[-2],heatmaps.size()[-1]), bboxes_list, input_dict=False)
    weighted_ious = get_weighted_iou_mult_class_vec(heatmaps, bounding_maps)
    return weighted_ious



dir = os.path.join(os.path.dirname( __file__ ),"../../Data/VOC2012/Annotations")
boxes1, _ = get_bounding_boxes("2007_000032.xml", dir= dir)
heatmap1 = np.random.rand(486,500)
iou, i, u = get_weighted_iou(heatmap1, boxes1)

#plt.imshow(heatmap)
print(iou)
#plt.imshow(i)
#plt.imshow(u)


boxes2, _ = get_bounding_boxes("2007_000027.xml", dir=dir)

bounding_maps = _get_bounding_maps((500,500), [boxes1, boxes2])
print(bounding_maps.size())


# try expanding
heatmaps_org = torch.rand(2,2, 500,500)
zeros = torch.zeros(2,4,500,500)
heatmaps = _expand_with_zeros(heatmaps_org, 1,4)
#heatmaps[0:2,0:2,:500,:500] = heatmaps_org
print("HEATMAPS", heatmaps)
ious = get_weighted_iou_mult_class_vec(heatmaps,bounding_maps)
print("ious", ious.size(), ious)

# try resizing
#boxes1, size1 = get_bounding_boxes("2007_000032.xml", dir= dir, return_dict=False)
#boxes2, size2 = get_bounding_boxes("2007_000027.xml", dir=dir, return_dict=False)
#res_bounding_boxes = resize_bounding_boxes([boxes1, boxes2], size_list=[size1, size2])
#print(res_bounding_boxes)
#res_bounding_maps = _get_bounding_maps((256,256), res_bounding_boxes, input_dict=False)

# -> try batch resizing
filenames = ["2007_000032.xml", "2007_000027.xml"]
bboxes_list, size_list = get_bounding_boxes_vec(["2007_000032.xml", "2007_000027.xml"], dir=dir, return_dict=False)
res_bounding_boxes = resize_bounding_boxes(bboxes_list, size_list=size_list)
res_bounding_maps = _get_bounding_maps_vec_NEW((256,256), res_bounding_boxes, 5, input_dict=False)

heatmaps_org = torch.rand(2,2, 256,256)
heatmaps = _expand_with_zeros(heatmaps_org, 1,4)
ious = compute_weighted_iou_mult_class_vec(filenames,dir, heatmaps)
print("FINAL IOUS:", ious)

image_transforms =  transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((256,256)),
    ])
img_dir = os.path.join(os.path.dirname( __file__ ),"../../Data/VOC2012/JPEGImages")
image1 = (PIL.Image.open(os.path.join(img_dir,"2007_000032.jpg")))
image2 = (PIL.Image.open(os.path.join(img_dir,"2007_000027.jpg")))
images = [image1, image2]
images_transformed = [image_transforms(image1), image_transforms(image2)]


# try NEW
heatmaps_org = torch.rand(2,3, 256,256)
heatmaps_org = heatmaps_org / heatmaps_org
heatmaps_org = _expand_with_zeros(heatmaps_org, 1,4)
print("res_bounding_maps", res_bounding_maps.size())
out = get_weighted_iou_mult_class_vec_NEW(res_bounding_maps,res_bounding_maps)
print("OUT", out.size(), out)


fig, ax = plt.subplots(len(bounding_maps),4)
for j_bounding_map in range(len(bounding_maps)):
    pass
    #ax[j_bounding_map][0].imshow(images[j_bounding_map])
    #ax[j_bounding_map][1].imshow(bounding_maps[j_bounding_map])
    #ax[j_bounding_map][2].imshow(images_transformed[j_bounding_map])
    #ax[j_bounding_map][2].imshow(res_bounding_maps[j_bounding_map], alpha=0.5)
    #ax[j_bounding_map][3].imshow(images_transformed[j_bounding_map])

# visualize each bounding box inidividually
fig, ax = plt.subplots(len(res_bounding_maps),5)
for j_bounding_map in range(len(res_bounding_maps)):
    
    for i_bbox in range(len(res_bounding_maps[j_bounding_map])):
        print(res_bounding_maps[j_bounding_map].shape)
        print(res_bounding_maps[j_bounding_map][i_bbox].shape)
        ax[j_bounding_map][i_bbox].imshow(images_transformed[j_bounding_map])
        ax[j_bounding_map][i_bbox].imshow(res_bounding_maps[j_bounding_map][i_bbox], alpha=0.5)

plt.show()

# TODO: fix zero expansion

