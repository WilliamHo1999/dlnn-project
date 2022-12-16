import numpy as np
import torch

# credit: https://nbviewer.org/gist/Teagum/460a508cda99f9874e4ff828e1896862
def get_hellinger_distance(dist1, dist2):
    """Hellinger distance between distributions"""
    return np.sqrt(sum([ (np.sqrt(p_i) - np.sqrt(q_i))**2 for p_i, q_i in zip(dist1, dist2) ]) / 2)

def hellinger_distance_vec(tensor1, tensor2):
    """ tensors are of type (num_samples X w X h) where (w X h) is the distribution """
    squared_diff = torch.sub(torch.sqrt(tensor1), torch.sqrt(tensor2))**2
    print(squared_diff)
    return torch.sqrt(torch.sum(squared_diff, dim=[-1,-2], keepdim=True) / 2) 

def get_bhattacharyya_distance(dist1,dist2):
    return -np.log(sum([np.sqrt(p_i*q_i) for p_i, q_i in zip(dist1, dist2) ] ))