import random
import os
import numpy as np
import calibration as cal
import ATC as atc
from scipy.special import softmax
import pandas as pd

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

def calculate_probs(model, device, data_loader):
  probs = None
  with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
      data, _, _ = batch
      logits = model(data.to(device))
      probabilities = F.softmax(logits, dim=1)
      if probs is None:
        probs = probabilities
      else:
        probs = torch.cat((probs, probabilities))     
      progress_bar(batch_idx, len(data_loader), 'Calculating probabilities')
  return probs


def threshold(model, device, pool_data, val_data, query_size, batch_size, num_workers):
  
  print("Calculating val probabilities")
  val_loader = DataLoader(val_data, shuffle = False, batch_size=batch_size, num_workers=num_workers)
  source_probs = calculate_probs(model, device, val_loader).cpu().numpy()
  #print(f"Size of source_probs is {source_probs.size}")
  source_labels = val_data.y_array.cpu().numpy()
  calibration_error = cal.ece_loss(source_probs, source_labels)
  #print("Calibration error is {}".format(calibration_error))
  calibrator = cal.TempScaling(bias=False)
  calibrator.fit(source_probs, source_labels)
  calibrated_source_probs = calibrator.calibrate(source_probs)

  print("Calculating selection pool probabilities")
  pool_loader = DataLoader(pool_data, shuffle = False, batch_size=batch_size, num_workers=num_workers)
  test_probs = calculate_probs(model, device, pool_loader).cpu().numpy()
  #print(f"Size of test_probs is {test_probs.size}")
  calibrated_test_probs = calibrator.calibrate(test_probs)
  atc_acc, threshold = atc.ATC_accuracy(calibrated_source_probs, source_labels, calibrated_test_probs)
  print(f"ATC estimated accuracy on selection pool is {atc_acc} and threshold is {threshold}")
  
  scores = np.max(calibrated_test_probs, axis=-1)
  candidate_idx = np.nonzero(scores < threshold)[0]
  print(f"For testing, candidate_idx [10] has probability {scores[candidate_idx[10]]}")
  sample_idx = random.sample(list(candidate_idx), query_size)
  return sample_idx


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

def find_avg_c_group(model, device, data_loader, dataset, grouper):
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

            progress_bar(batch_idx, len(data_loader), 'Calculating confidence scores')
            
    confidences = np.asarray(confidences)
    group, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
    group = np.array(group)
    group_counts = np.array(group_counts)
    group_avg_c = np.zeros(len(group_counts))
    for i in range(len(group_counts)):
      group_avg_c[i] = np.mean(confidences[np.nonzero(group == i)[0]])
    worst_group = np.argmin(group_avg_c)
    wg_avg_c = group_avg_c[worst_group]
    # print(wg_avg_c)
    # print("Worst group is {}: {} with average confidence score {}".format(worst_group, grouper.group_str(worst_group), wg_avg_c))
    return worst_group



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

def query_the_oracle(unlabeled_mask, model, device, dataset, val_data, grouper, query_size=40, 
                     group_strategy=None, exclude=None, wg=None, query_strategy='least_confidence', 
                     replacement=False, pool_size=0, batch_size=8, num_workers=2):
    
    if replacement:
      candidate_mask = np.ones(len(unlabeled_mask))
    else:
      candidate_mask = unlabeled_mask.copy()

    group, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
    group = np.array(group)
    group_counts = np.array(group_counts)
    num_group = len(group_counts)
    
    # exclude some groups 
    if exclude is not None:
      for i in exclude:
        candidate_mask[np.nonzero(group == i)[0]] = 0
    
    group_idx = np.arange(len(dataset)) #used for creating group_mask, group_mask[group_idx] is set to 1; default is the entire dataset(no group_strategy)
    if group_strategy == "oracle" or group_strategy == "avg_c_val":
      assert wg != None and wg in range(num_group), "For group strategy = oracle or avg_c_val, a valid worst group is needed"
    elif group_strategy == "avg_c":
      data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
      wg = find_avg_c_group(model, device, data_loader, dataset, grouper)
    if group_strategy != None:
      assert wg != None and wg in range(num_group), "For group strategy != None, a valid worst group is needed"
      group_idx = np.nonzero(group == wg)[0]
    
    group_mask = np.zeros(len(dataset))
    group_mask[group_idx] = 1
    unlabeled_idx = np.nonzero(candidate_mask * group_mask)[0] #indices of datapoints available for sampling
    print("Number of selected unlabeled samples: {}, wg is {}".format(len(unlabeled_idx), wg))

    selected_idx = None
    if len(unlabeled_idx) < query_size:
      print(f"Selected unlabeled candidates of size {len(unlabeled_idx)} less than query size, sample the rest from all unlabeled data")
      selected_idx = unlabeled_idx.copy()
      query_size -= len(selected_idx)
      candidate_mask[selected_idx] = 0
      unlabeled_idx = np.nonzero(candidate_mask)[0]
    
    #WILDSSubset(train_data, labeled_idx, transform=None)
    
    # Select a pool of samples to query from
    use_pool = pool_size > 0 and len(unlabeled_idx) > pool_size
    if use_pool:    
        pool_idx = random.sample(range(0, len(unlabeled_idx)), pool_size)
        pool_data = Subset(dataset, unlabeled_idx[pool_idx])
        pool_loader = DataLoader(Subset(dataset, unlabeled_idx[pool_idx]), shuffle = False, batch_size=batch_size, num_workers=num_workers)
    else:
        # rohan: use WildsSubset here to maintain consistency? you also might need the group information for the other AL schemes we experiment with
        pool_data = Subset(dataset, unlabeled_idx)
        pool_loader = DataLoader(Subset(dataset, unlabeled_idx), shuffle = False, batch_size=batch_size, num_workers=num_workers)

    print("Querying ...")
    if query_strategy == 'margin':
      sample_idx = margin_query(model, device, pool_loader, query_size)
    elif query_strategy == 'least_confidence':
      sample_idx = least_confidence_query(model, device, pool_loader, query_size)
    elif query_strategy == 'threshold':
      sample_idx = threshold(model, device, pool_data, val_data, query_size, batch_size, num_workers)
    else:
        # 'random'
        if use_pool:
            sample_idx = random.sample(range(0, len(pool_idx)), query_size)
        else:
            sample_idx = random.sample(range(0, len(unlabeled_idx)), query_size)
    
    # update the unlabeled mask, change sign from 1 to 0 for newly queried samples
    if use_pool:
        selected = unlabeled_idx[pool_idx][sample_idx]
    else:
        selected = unlabeled_idx[sample_idx] 
    
    if selected_idx is None: 
      selected_idx = selected
    else:
      selected_idx = np.append(selected_idx, selected)
    
    unlabeled_mask[selected_idx] = 0
    
    return selected_idx
    



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

            
            
