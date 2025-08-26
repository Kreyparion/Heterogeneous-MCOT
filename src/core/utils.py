import math
import torch
import numpy as np

def choice_along_first_axis(weights,N,M):
    cumsumed_weights = torch.cumsum(weights,1)
    random_selection = torch.rand((N,1))
    possible_prob = cumsumed_weights > random_selection
    selection = torch.zeros((N,M), dtype=torch.bool)
    selection[:,1:] = (possible_prob[:,1:] != possible_prob[:,:-1])
    selection[:,0] = ~torch.any(selection,1)
    return selection

def choice_along_first_axis2(weights):
    cumsumed_weights = torch.cumsum(weights,1)
    random_selection = torch.rand((weights.shape[0],1))
    possible_prob = cumsumed_weights > random_selection
    return torch.argmax(possible_prob.int(),1)


def write_to_txt(name,X,Y):
    np.savetxt(name + ".txt", torch.stack([X,Y],1).cpu().numpy(),delimiter=" ")