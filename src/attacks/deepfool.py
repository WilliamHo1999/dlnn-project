# SOURCE https://github.com/BXuan694/Universal-Adversarial-Perturbation/blob/master/deepfool.py
# ADAPTED TO MUCH CURRENT VERSION OF PYTORCH

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
###from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, num_classes, overshoot, max_iter):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            ###zero_gradients(x)
            ###x.zero_grad()
            x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image


def check_if_correct_label(pred, label, total_present):
    num_correct = 0
    for p,l in zip(pred.squeeze(), label.squeeze()):
        if l == 1 and p == l:
            num_correct += 1
            
    return num_correct > 0

def deepfool_multi_label(image, label, net, num_classes, overshoot, max_iter):
    
    out = net(image).data.cpu().flatten(start_dim=1)
    pred = out.sigmoid()
    pred = pred > 0.5
    
    input_shape = image.shape
    pert_image = image.clone()
    
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    
    loop_i = 0
    
    x = pert_image
    x.requires_grad = True
    fs = net(x)
    k_i = label
    
#    batch_size = x.shape[0]
    
    corrects = torch.nonzero(label.squeeze())
    num_correct = label.count_nonzero()
    while check_if_correct_label(k_i, label, num_correct) and loop_i < max_iter:
        
        """
        keep_samples = [] 
        for b in range(batch_size):
            # Simplify to only consider if all classes are correct.
            if (k_i[b] == label[b]).all():
                keep_samples.append(b)

        k_i = k_i[keep_samples, :]
        """
        
        pert = torch.inf
        # Only want gradients for true class
        masked_output = fs * label
        masked_output.sum().backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        
        for corr in corrects:
            for k in range(1, num_classes):
                if k in corrects:
                    continue
                x.grad.zero_()
                fs[:,k].sum().backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                w_k = cur_grad - grad_orig
                f_k = (fs[0, k] 
                       - fs[0, corr]).data.cpu().numpy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
        
        
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        
        x = pert_image
        x.requires_grad = True
        fs = net(x)
        out = net(image).data.cpu().flatten(start_dim=1)
        pred = out.sigmoid()
        k_i = (pred > 0.5).cuda()
        
        loop_i += 1
    
    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image