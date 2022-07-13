import random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F


from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from wilds.common.data_loaders import get_train_loader, get_eval_loader

import torchvision
import torchvision.transforms as transforms
from torchvision import utils, models

from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from utils import progress_bar

# rohan: if parts of this code are lifted from another repo, let's put a link to that here
# so that we remember to give appropriate credit in the future

'''
Each query strategy below returns a list of len=query_size with indices of 
samples that are to be queried.

Arguments:
- model (torch.nn.Module): 
- device (torch.device): 
- dataloader (torch.utils.data.DataLoader)
- query_size (int): number of samples to be queried for labels (default=40)

Adapted from https://github.com/opetrova/Tutorials/blob/master/Active%20Learning%20image%20classification%20PyTorch.ipynb
'''

def least_confidence_query(model, device, data_loader, query_size=10):

    confidences = []

    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data, _, _ = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            
            # Keep only the top class confidence for each sample
            most_probable = torch.max(probabilities, dim=1)[0]
            confidences.extend(most_probable.cpu().tolist())

            progress_bar(batch_idx, len(data_loader), 'Querying least confidence')
            
    conf = np.asarray(confidences)
    sorted_pool = np.argsort(conf)
    # Return the relative indices corresponding to the lowest `query_size` confidences
    return sorted_pool[0:query_size]

def margin_query(model, device, data_loader, query_size=10):
    
    margins = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
        
            data, _, _ = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            
            # Select the top two class confidences for each sample
            toptwo = torch.topk(probabilities, 2, dim=1)[0]
            
            # Compute the margins = differences between the two top confidences
            differences = toptwo[:,0]-toptwo[:,1]
            margins.extend(torch.abs(differences).cpu().tolist())

    margin = np.asarray(margins)
    sorted_pool = np.argsort(margin)
    # Return the relative indices corresponding to the lowest `query_size` margins
    return sorted_pool[0:query_size]


'''
Queries the oracle (Change the unlabeled) for  labels for 'query_size' samples using 'query_strategy'

Arguments:
- unlabeled_mask (of the train set)
- model (torch.nn.Module)
- device: torch.device (CPU or GPU)
- dataset (torch.utils.data.DataSet)
- query_size (int): number of samples to be queried for labels (default=40)
- query_strategy (string): one of ['random', 'least_confidence', 'margin'], 
                           otherwise defaults to 'random'
- pool_size (int): when > 0, the size of the randomly selected pool from the unlabeled_loader to consider
                   (otherwise defaults to considering all of the associated data)
- batch_size (int): default=8
- num_workers (int): default=2

Modifies:
- dataset: edits the labels of samples that have been queried; updates dataset.unlabeled_mask
'''

def query_the_oracle(unlabeled_mask, model, device, dataset, grouper, query_size=40, query_strategy='least_confidence', 
                     pool_size=0, batch_size=8, num_workers=2):
    
    unlabeled_idx = np.nonzero(unlabeled_mask)[0]

    #WILDSSubset(train_data, labeled_idx, transform=None)
    
    # Select a pool of samples to query from
    if pool_size > 0:    
        pool_idx = random.sample(range(1, len(unlabeled_idx)), pool_size)
        pool_loader = DataLoader(Subset(dataset, unlabeled_idx[pool_idx]), shuffle = False, batch_size=batch_size, num_workers=num_workers)
    else:
        # rohan: use WildsSubset here to maintain consistency? you also might need the group information for the other AL schemes we experiment with
        pool_loader = DataLoader(Subset(dataset, unlabeled_idx), shuffle = False, batch_size=batch_size, num_workers=num_workers)
    
    print("Querying ...")
    if query_strategy == 'margin':
        sample_idx = margin_query(model, device, pool_loader, query_size)
    elif query_strategy == 'least_confidence':
        sample_idx = least_confidence_query(model, device, pool_loader, query_size)
    else:
        # 'random'
        if pool_size > 0:
            sample_idx = random.sample(pool_idx, query_size)
        else:
            sample_idx = random.sample(range(1, len(unlabeled_idx)), query_size)

    #print the group information of selected datapoints
    selected = WILDSSubset(dataset, unlabeled_idx[sample_idx], transform=None)
    meta_array = selected.metadata_array
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    # for i in range(len(group_counts)):
    #     print("group: {}, count: {} \n".format(grouper.group_str(i), group_counts[i]))
        
    # update the unlabeled mask, change sign from 1 to 0 for newly queried samples
    if pool_size > 0:
        unlabeled_mask[unlabeled_idx[pool_idx][sample_idx]] = 0
    else:
        unlabeled_mask[unlabeled_idx[sample_idx]] = 0
    
    return group_counts
    



# From https://github.com/google-research/big_transfer.git bit_hyperrule.py
def get_schedule(dataset_size):
  if dataset_size < 20_000:
    return [100, 200, 300, 400, 500]
  elif dataset_size < 500_000:
    return [500, 3000, 6000, 9000, 10_000]
  else:
    return [500, 6000, 12_000, 18_000, 20_000]


def get_lr(step, dataset_size, base_lr=0.003):
  """Returns learning-rate for `step` or None at the end."""
  supports = get_schedule(dataset_size)
  # Linear warmup
  if step < supports[0]:
    return base_lr * step / supports[0]
  # End of training
  elif step >= supports[-1]:
    return None
  # Staircase decays by factor of 10
  else:
    for s in supports[1:]:
      if s < step:
        base_lr /= 10
    return base_lr

            
            
