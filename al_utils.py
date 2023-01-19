from curses.ascii import CAN
from itertools import filterfalse
import random
import os
from unittest.mock import NonCallableMock
import numpy as np
import calibration as cal
import ATC as ATC
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
from ffcv_dataloader import ffcv_train_loader, ffcv_val_loader, ffcv_train_val_loader
from ffcv.loader import Loader

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

def least_confidence_query(model, device, data_loader, query_size=10, calibrator=None, noise=0.0):
    model.eval()
    probs = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch[0]
            with torch.cuda.amp.autocast():
              logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            probs = torch.cat((probs, probabilities))
            progress_bar(batch_idx, len(data_loader), 'Querying least confidence')
    
    probs = np.asarray(probs.cpu())
    if calibrator is not None:
      print("Applying calibration")
      probs = calibrator.calibrate(probs)
    conf = np.max(probs, axis=1)
    if noise > 0:
      print(f"Adding noise {noise}")
      perturbation = np.random.normal(0, noise, conf.size)
      conf += perturbation
    sorted_pool = np.argsort(conf)
    # Return the relative indices corresponding to the lowest `query_size` confidences
    return sorted_pool[0:query_size]

def margin_query(model, device, data_loader, query_size=10, noise=0.0):
    margins = []
    model.eval() 
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):    
            data = batch[0]
            with torch.cuda.amp.autocast():
              logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            # Select the top two class confidences for each sample
            toptwo = torch.topk(probabilities, 2, dim=1)[0]
            # Compute the margins = differences between the two top confidences
            differences = toptwo[:,0]-toptwo[:,1]
            margins.extend(torch.abs(differences).cpu().tolist())
            progress_bar(batch_idx, len(data_loader), 'Querying margin')

    margin = np.asarray(margins)
    if noise > 0:
      print(f"Adding noise {noise}")
      perturbation = np.random.normal(0, noise, margin.size)
      margin += perturbation
    sorted_pool = np.argsort(margin)
    # Return the relative indices corresponding to the lowest `query_size` margins
    return sorted_pool[0:query_size]

def query(model, device, data_loader, query_size=10, query_strategy='margin_logit'):
  logits, probs = calculate_probs(model, device, data_loader, return_logit=True)
  probs = probs.cpu().numpy()
  logits = logits.cpu().numpy()
  if query_strategy == 'margin_logit':
    print("Calculating margin_logit")
    logits_top2 = np.take_along_axis(logits, np.argsort(logits, axis=1)[:, -2:], axis=1)
    margin = logits_top2[:, 1] - logits_top2[:, 0]
    sorted_idx = np.argsort(margin)
    return sorted_idx[0:query_size]
  elif query_strategy == 'entropy':
    print("Calculating entropy")
    entropy = np.sum(np.multiply(probs, np.log(probs + 1e-20)), axis=1)
    sorted_idx = np.argsort(entropy)
    return sorted_idx[0:query_size]
  elif query_strategy == 'margin':
    print("Calculating margin")
    top2 = np.take_along_axis(probs, np.argsort(probs, axis=1)[:, -2:], axis=1)
    margin = top2[:, 1] - top2[:, 0]
    sorted_idx = np.argsort(margin)
    return sorted_idx[0:query_size]

def calculate_probs(model, device, data_loader, return_logit=False):
  probs = torch.tensor([]).to(device)
  logs = torch.tensor([]).to(device)
  with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
      data = batch[0]
      with torch.cuda.amp.autocast():
        logits = model(data.to(device))
      logs = torch.cat((logs, logits))
      probabilities = F.softmax(logits, dim=1)
      probs = torch.cat((probs, probabilities))     
      progress_bar(batch_idx, len(data_loader), 'Calculating probabilities')
  if return_logit: 
    return logs, probs
  return probs

def threshold_query(model, device, pool_data, val_data, grouper, query_size, batch_size, num_workers, pool_loader=None,
                    val_probs=None, group_balance=False, group_spec=False, score_fn='MC', calibration=True):
  _, atc_groups, _, candidate_idx = atc(model, device, pool_data, val_data, grouper, batch_size, num_workers, target_loader=pool_loader,
                                     source_probs=val_probs, return_error_idx=True, group_spec=group_spec, score_fn=score_fn, calibration=calibration)

  if group_balance:
    weights = 100.0 - atc_groups
    weights = weights/np.sum(weights)
    # sample_idx = np.random.choice(candidate_idx, query_size, replace=False, p=weights)
    mask = np.zeros(len(pool_data))
    mask[candidate_idx] = 1
    sample_idx = sample_from_distribution(weights, pool_data, mask, query_size, grouper, 'random',
                                          batch_size, num_workers, model, device, val_data, ffcv_loader=pool_loader)
  else:
    if len(candidate_idx) < query_size: 
      sample_idx = candidate_idx
    else:
      sample_idx = np.random.choice(candidate_idx, query_size, replace=False)

  return sample_idx

def atc(model, device, target_data, source_val_data, grouper, batch_size, num_workers, target_loader=None,
        test_probs=None, source_probs=None, return_error_idx=False, group_spec=False, score_fn='MC', calibration=True):
  group, group_count = grouper.metadata_to_group(target_data.metadata_array, return_counts=True)
  group = group.cpu().numpy()
  group_count = group_count.cpu().numpy()
  num_groups = group_count.size
  
  if source_probs is None:
    print("Calculating source val probabilities")
    source_loader = DataLoader(source_val_data, shuffle = False, batch_size=batch_size, num_workers=num_workers)
    source_probs = calculate_probs(model, device, source_loader)
    # print(f"Shape of source_probs is {source_probs.shape}")
  source_probs = source_probs.cpu().numpy()
  source_labels = source_val_data.y_array.cpu().numpy()

  if test_probs is None:
    print("Calculating target probabilities")
    if target_loader is None:
      target_loader = DataLoader(target_data, shuffle = False, batch_size=batch_size, num_workers=num_workers)
    test_probs = calculate_probs(model, device, target_loader)
  # print(f"Size of test_probs is {test_probs.size}")
  test_probs = test_probs.cpu().numpy()

  # save_probs_info(source_probs, source_val_data, grouper, "domainnet_four_nopre_source")
  # save_probs_info(test_probs, target_data, grouper, "domainnet_four_nopre_target")

  group_threshold = np.zeros(num_groups)
  candidate_idx = np.array([]).astype(int)
  atc_groups = np.zeros(num_groups)
  if group_spec: 
    # if group_sec is set to True, source_data and target_data both need to contain all possible groups
    source_group, source_group_count = grouper.metadata_to_group(source_val_data.metadata_array, return_counts=True)
    source_group = source_group.cpu().numpy()
    source_group_count = source_group_count.cpu().numpy()
    source_num_groups = source_group_count.size
    for i in range(num_groups):
      source_group_idx = np.nonzero(source_group==i)[0]
      target_group_idx = np.nonzero(group==i)[0]
      _, atc_acc, threshold, error_idx = get_atc_threshold(source_probs[source_group_idx], source_labels[source_group_idx], test_probs[target_group_idx], 
                                                           score_fn=score_fn, calibration=calibration)
      group_threshold[i] = threshold
      candidate_idx = np.append(candidate_idx, target_group_idx[error_idx])
      atc_groups[i] = atc_acc
    atc_acc = np.sum(atc_groups * group_count)/len(target_data) # atc_acc is the weighted average estimated accuracy 
  else:
    _, atc_acc, threshold, error_idx = get_atc_threshold(source_probs, source_labels, test_probs, 
                                                         score_fn=score_fn, calibration=calibration)
    group_threshold = np.array([threshold]*num_groups)
    candidate_idx = error_idx
    for g in range(num_groups):
      if group_count[g] <= 0: 
        continue
      group_idx = np.nonzero(group == g)[0]
      error_in_group = np.intersect1d(error_idx, group_idx)
      atc_groups[g] = 100.0 - error_in_group.size / group_count[g] * 100.0

  print(f"ATC estimated accuracy on target data is {atc_acc}")
  print(f"ATC estimated threshold on target data is {group_threshold[0]}")
  # print(f"ATC estimated accuracy on target groups are {atc_groups}")
  # print(f"ATC estimated threshold on target groups are {group_threshold}")

  results = compare_actual_error(target_data, test_probs, candidate_idx, grouper)
  print("Precision and recall for ATC predicted errors are")
  print(results[0])
  atc_prec_recall = results[1:, :].flatten()

  #for analysis only 
  # _, atc_acc, threshold, error_idx = get_atc_threshold(source_probs, source_labels, test_probs, 
  #                                                        score_fn='MA', calibration=calibration)
  # group_threshold = np.array([threshold]*num_groups)
  # candidate_idx = error_idx
  # for g in range(num_groups):
  #   if group_count[g] <= 0: 
  #     continue
  #   group_idx = np.nonzero(group == g)[0]
  #   error_in_group = np.intersect1d(error_idx, group_idx)
  #   atc_groups[g] = 100.0 - error_in_group.size / group_count[g] * 100.0
  
  # print(f"ATC estimated accuracy on target data is {atc_acc}")
  # print(f"ATC estimated accuracy on target groups are {atc_groups}")
  # print(f"ATC estimated threshold on target groups are {group_threshold}")

  # results = compare_actual_error(target_data, test_probs, candidate_idx, grouper)
  # print("Precision and recall for ATC predicted errors are")
  # print(results)
  # atc_prec_recall = results[1:, :].flatten()


  if return_error_idx: return atc_acc, atc_groups, group_threshold, candidate_idx
  return atc_acc, atc_groups, group_threshold, atc_prec_recall

def save_probs_info(probs, data, grouper, name):
  labels = data.y_array.cpu().numpy()
  df = pd.DataFrame(labels, columns=['label'])
  groups = grouper.metadata_to_group(data.metadata_array, return_counts=False)
  df['group'] = groups.cpu().numpy()
  rank = np.argsort(probs, axis=1)
  df['predicted_label'] = rank[:, -1]
  top_3_probs = np.take_along_axis(probs, rank[:, -3:], axis=1)
  top1 = np.reshape(top_3_probs[:, 2], (-1, 1))
  top2 = np.reshape(top_3_probs[:, 1], (-1, 1))
  top3 = np.reshape(top_3_probs[:, 0], (-1, 1))
  margin = top1 - top2
  margin_logit = np.log(top1) - np.log(top2)
  entropy = np.sum(np.multiply(probs, np.log(probs + 1e-20)), axis=1).reshape(-1, 1)
  scores = np.concatenate((top1, top2, top3, margin, margin_logit, entropy), axis=1)
  scores_df = pd.DataFrame(scores, columns=['top1', 'top2', 'top3', 'margin', 'margin_logit', 'entropy'])
  result = pd.concat([df, scores_df], axis=1)
  if not os.path.isdir(f'./uncertainty_results/{name}'):
    os.mkdir(f'./uncertainty_results/{name}')
  print(f"Saving ./uncertainty_results/{name}/summary.csv")
  print(f"result is of shape {result.shape}")
  result.to_csv(f"./uncertainty_results/{name}/summary.csv")
  raw_probs = pd.DataFrame(probs)
  print(f"Saving ./uncertainty_results/{name}/probs.csv")
  print(f"probs is of shape {raw_probs.shape}")
  raw_probs.to_csv(f"./uncertainty_results/{name}/probs.csv")
  reversed_rank = pd.DataFrame(np.flip(rank, axis=1))
  print(f"Saving ./uncertainty_results/{name}/rank.csv")
  print(f"rank is of shape {reversed_rank.shape}")
  reversed_rank.to_csv(f"./uncertainty_results/{name}/rank.csv")

def get_atc_threshold(source_probs, source_labels, test_probs, score_fn='MC', calibration=True):
  calibration_error = -1
  if calibration:
    print("Applying calibration")
    # calibration_error = cal.ece_loss(source_probs, source_labels)
    # print("Calibration error is {}".format(calibration_error))
    calibrator = cal.TempScaling(bias=False)
    calibrator.fit(source_probs, source_labels)
    calibrated_source_probs = calibrator.calibrate(source_probs)
    calibrated_test_probs = calibrator.calibrate(test_probs)
  else:
    calibrated_source_probs = source_probs
    calibrated_test_probs = test_probs

  atc_acc, threshold, scores = ATC.ATC_accuracy(calibrated_source_probs, source_labels, calibrated_test_probs, return_score=True, score_function=score_fn)
  # print(f"ATC estimated accuracy on target data is {atc_acc} and threshold is {threshold}")
  error_idx = np.nonzero(scores < threshold)[0]
  return calibration_error, atc_acc, threshold, error_idx

def compare_actual_error(test_data, test_probs, candidate_idx, grouper):
  predicted = np.argmax(test_probs, axis=-1).squeeze()
  y_array = np.asarray(test_data.y_array.cpu())
  true_error_idx = np.nonzero(predicted != y_array)[0]
  prec, recall = precision_recall(candidate_idx, true_error_idx)
  group, group_counts = grouper.metadata_to_group(test_data.metadata_array, return_counts=True)
  group = np.asarray(group.cpu())
  group_counts = np.asarray(group_counts.cpu())
  results = np.zeros((group_counts.size+1, 2))
  results[0,0] = prec
  results[0,1] = recall
  true_acc =  np.zeros(group_counts.size+1)
  true_acc[0] = (1 - true_error_idx.size / len(test_data)) * 100.0
  # for i in range(group_counts.size):
  #   if group_counts[i] > 0:
  #     group_idx = np.nonzero(group==i)[0]
  #     group_real_error = np.intersect1d(true_error_idx, group_idx)
  #     true_acc[i+1] = (1 - group_real_error.size / group_counts[i])*100.0
  #     prec, recall = precision_recall(np.intersect1d(candidate_idx, group_idx), group_real_error)
  #     results[i+1,0] = prec
  #     results[i+1,1] = recall
  #   else:
  #     true_acc[i+1] = None
  #     results[i+1,0] = None
  #     results[i+1,1] = None
  print(f"Actual acc on target data is {true_acc[0]}")
  # print(f"Actual acc on target data groups are {true_acc[1:]}")
  return results
  
def precision_recall(predicted, true):
  overlapping = np.intersect1d(predicted, true)
  if predicted.size > 0:
    prec = overlapping.size/predicted.size * 100.0
  else: 
    prec = None
  if true.size > 0:
    recall = overlapping.size/true.size * 100.0
  else: 
    recall = None
  return prec, recall

def sample_from_distribution(distribution, dataset, unlabeled_mask, query_size, grouper, query_strategy,
                             batch_size, num_workers, model, device, val_data, ffcv_loader=None, dataset_name= None, val_probs=None, pool_size=-1,
                             score_fn='MC', calibration=True, addcal=False, noise=0.0):
  distribution = np.array(distribution)
  num_to_sample = [round(i) for i in (distribution * query_size)]
  group, group_count = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
  group = group.cpu().numpy()
  group_count = group_count.cpu().numpy()
  num_group = group_count.size
  assert len(num_to_sample) == num_group, f"To sample from distribution, {len(num_to_sample)} groups is given, {num_group} groups are required"
  selected = np.array([])
  if pool_size > 0:
     pool_sizes = [round(i) for i in (distribution * pool_size)]
  else: 
     pool_sizes = [-1] * num_group
  
  if addcal:
    calibrator = cal.TempScaling(bias=False)
    calibrator.fit(val_probs.cpu().numpy(), val_data.y_array.cpu().numpy())
  else:
    calibrator = None
    
  for g in range(num_group):
    num_select = num_to_sample[g]
    group_mask = (group == g)
    candidate_idx = np.nonzero(group_mask * unlabeled_mask)[0]

    # if unlabeled datapoints less than query_size, first label the available points and then sample from the group 
    if candidate_idx.size < num_select:
      selected.append(candidate_idx)
      num_select -= candidate_idx.size
      candidate_idx = np.nonzero(group_mask)[0] 
    
    # determine if to use pool sampling
    pool_select = pool_sizes[g]
    use_pool = pool_select > 0 and len(candidate_idx) > pool_select
    if use_pool:    
      pool_idx = random.sample(range(0, len(candidate_idx)), pool_select)
      subset_idx = candidate_idx[pool_idx]
    else:
      subset_idx = candidate_idx
    pool_data = WILDSSubset(dataset, subset_idx, transform=None)
    if ffcv_loader == 'ffcv': 
      # loader_args = {**ffcv_loader._args}
      # loader_args['indices'] = subset_idx
      # pool_loader = Loader(**loader_args)
      pool_loader = ffcv_train_val_loader(dataset_name, indices=subset_idx, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    else:
      pool_loader = DataLoader(pool_data, shuffle = False, batch_size=batch_size, num_workers=num_workers)
    
    print(f"Querying group {g} for {num_select} elements...")
    if query_strategy == 'random':
      sample_idx = random.sample(range(0, subset_idx.size), num_select)
    elif query_strategy == 'margin':
      sample_idx = margin_query(model, device, pool_loader, num_select, noise=noise)
    elif query_strategy == 'least_confidence':
      sample_idx = least_confidence_query(model, device, pool_loader, num_select, calibrator=calibrator, noise=noise)
    elif query_strategy == 'threshold':
      sample_idx = threshold_query(model, device, pool_data, val_data, grouper, num_select, batch_size, num_workers, 
                                  pool_loader=pool_loader, val_probs=val_probs, score_fn=score_fn, calibration=calibration)
    elif query_strategy == 'threshold_spec':
      val_group = grouper.metadata_to_group(val_data.metadata_array)
      val_group = val_group.cpu().numpy()
      val_group_idx = np.nonzero(val_group == g)[0]
      sample_idx = threshold_query(model, device, pool_data, WILDSSubset(val_data, val_group_idx, transform=None), grouper, num_select, batch_size, num_workers, 
                                  pool_loader=pool_loader, val_probs=val_probs[val_group_idx], score_fn=score_fn, calibration=calibration)
    else:
      sample_idx = query(model, device, pool_loader, num_select, query_strategy=query_strategy)
    
    selected = np.append(selected, subset_idx[sample_idx])

  selected = selected.astype(int)
  # unlabeled_mask[selected] = 0
  return selected


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
- batch_size (int): default=128
- num_workers (int): default=8

Modifies:
- dataset: edits the labels of samples that have been queried; updates dataset.unlabeled_mask
'''

def query_the_oracle(unlabeled_mask, model, device, dataset, val_data, grouper, query_size,
                     ffcv_loader=None, dataset_name=None, val_probs=None, sample_distribution=None, exclude=None, include=None, 
                     group_strategy=None, p=0.5, wg=None, query_strategy='least_confidence', 
                     score_fn='MC', calibration=True, addcal=False, noise=0.0,
                     group_losses=None, group_acc=None,
                     replacement=False, pool_size=0, batch_size=128, num_workers=8):
    group, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
    group = np.array(group)
    group_counts = np.array(group_counts)
    num_group = len(group_counts)
    
    if group_strategy == 'uniform':
      sample_distribution = np.ones(num_group)/num_group
    elif group_strategy == 'loss_proportional':
      assert group_losses is not None, "Query strategy is loss_proportional: require input of group_losses"
      assert group_losses.size == num_group, f"Query strategy is loss_proportional: input of group_losses has incorrect number of groups {group_losses.size}, required {num_group}"
      sample_distribution = group_losses / np.sum(group_losses)
      print(f"Loss proportional query: ")
      print(f"group losses are {group_losses}")
    elif group_strategy == 'error_proportional':
      group_errors = 1.0 - group_acc
      sample_distribution = group_errors / np.sum(group_errors)
      print(f"Error proportional query: ")
      print(f"group error rates are {group_errors}")
    elif group_strategy == 'loss_exp':
      exp_losses = np.exp(group_losses)
      sample_distribution = exp_losses / np.sum(exp_losses)
      print(f"Loss exponential query: ")
      print(f"group losses are {group_losses}")
    elif group_strategy == 'interpolate':
      wg_dis = np.zeros(num_group)
      wg_dis[np.argmin(group_acc)] = 1
      uniform_dis = np.ones(num_group)/num_group
      sample_distribution = p * wg_dis + p * uniform_dis
      print(f"Interpolation with alpha {p} query: ")
    
    if sample_distribution is not None:
      print(f"sample distribution is {sample_distribution}")
      selected_idx = sample_from_distribution(sample_distribution, dataset, unlabeled_mask, query_size, grouper, query_strategy, 
                                              batch_size, num_workers, model, device, val_data, 
                                              ffcv_loader=ffcv_loader, dataset_name=dataset_name, val_probs=val_probs, pool_size=pool_size,
                                              score_fn=score_fn, calibration=calibration, addcal=addcal, noise=noise)
      unlabeled_mask[selected_idx] = 0
      return selected_idx

    if replacement:
      candidate_mask = np.ones(len(unlabeled_mask))
    else:
      candidate_mask = unlabeled_mask.copy()
    
    # exclude some groups 
    mask = np.ones(candidate_mask.size)
    if include is not None: 
      mask = np.zeros(candidate_mask.size)
      for i in include:
        mask[np.nonzero(group == i)[0]] = 1
    elif exclude is not None:
      for i in exclude:
        mask[np.nonzero(group == i)[0]] = 0
    candidate_mask = candidate_mask * mask
    
    # sample according to group_strategy 
    group_idx = np.arange(len(dataset)) # used for creating group_mask, group_mask[group_idx] is set to 1; default is the entire dataset(no group_strategy)
    if group_strategy is not None:
      assert wg != None and wg in range(num_group), "For group strategy != None, a valid worst group is needed"
      group_idx = np.nonzero(group == wg)[0]
    group_mask = np.zeros(len(dataset))
    group_mask[group_idx] = 1
    unlabeled_idx = np.nonzero(candidate_mask * group_mask)[0] # indices of datapoints available for sampling
    print("Number of selected unlabeled samples: {}, wg is {}".format(len(unlabeled_idx), wg))

    selected_idx = None
    if len(unlabeled_idx) < query_size:
      print(f"Selected unlabeled candidates of size {len(unlabeled_idx)} less than query size, sample the rest from all unlabeled data")
      selected_idx = unlabeled_idx.copy()
      query_size -= len(selected_idx)
      candidate_mask[selected_idx] = 0
      unlabeled_idx = np.nonzero(candidate_mask)[0]
    
    # Select a pool of samples to query from
    use_pool = pool_size > 0 and len(unlabeled_idx) > pool_size
    if use_pool:    
      pool_idx = random.sample(range(0, len(unlabeled_idx)), pool_size)
      subset_idx = unlabeled_idx[pool_idx]
    else:
      subset_idx = unlabeled_idx
    pool_data = WILDSSubset(dataset, subset_idx, transform=None)
    if ffcv_loader == 'ffcv':
      # loader_args = {**ffcv_loader._args}
      # loader_args['indices'] = subset_idx
      # pool_loader = Loader(**loader_args)
      pool_loader = ffcv_train_val_loader(dataset_name, dataset=dataset, indices=subset_idx, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    else:
      pool_loader = DataLoader(pool_data, shuffle = False, batch_size=batch_size, num_workers=num_workers)
      
    if addcal:
      calibrator = cal.TempScaling(bias=False)
      calibrator.fit(val_probs.cpu().numpy(), val_data.y_array.cpu().numpy())
    else:
      calibrator = None
    
    print("Querying ...")
    if query_strategy == 'margin':
      sample_idx = margin_query(model, device, pool_loader, query_size, noise=noise)
    elif query_strategy == 'least_confidence':
      sample_idx = least_confidence_query(model, device, pool_loader, query_size, calibrator=calibrator, noise=noise)
    elif query_strategy == 'threshold':
      sample_idx = threshold_query(model, device, pool_data, val_data, grouper, query_size, batch_size, num_workers, 
                                  pool_loader=pool_loader, val_probs=val_probs, score_fn=score_fn, calibration=calibration)
    elif query_strategy == 'threshold_group':
      sample_idx = threshold_query(model, device, pool_data, val_data, grouper, query_size, batch_size, num_workers, 
                                  pool_loader=pool_loader, val_probs=val_probs, group_balance=True, score_fn=score_fn, calibration=calibration)
    elif query_strategy == 'threshold_spec':
      sample_idx = threshold_query(model, device, pool_data, val_data, grouper, query_size, batch_size, num_workers, 
                                  pool_loader=pool_loader, val_probs=val_probs, group_spec=True, score_fn=score_fn, calibration=calibration)
    elif query_strategy == 'threshold_group_spec':
      sample_idx = threshold_query(model, device, pool_data, val_data, grouper, query_size, batch_size, num_workers, 
                                  pool_loader=pool_loader, val_probs=val_probs, group_balance=True, group_spec=True, score_fn=score_fn, calibration=calibration)
    elif query_strategy == 'random':
      sample_idx = random.sample(range(0, subset_idx.size), query_size)
    else:
      sample_idx = query(model, device, pool_loader, query_size, query_strategy=query_strategy)

    # update the unlabeled mask, change sign from 1 to 0 for newly queried samples
    selected = subset_idx[sample_idx]
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
