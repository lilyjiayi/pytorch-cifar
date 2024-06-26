# to prevent multiprocess deadlock
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

'''Active learning with PyTorch.'''
from multiprocessing import AuthenticationError
from pickle import NONE
from xxlimited import Str
from sklearn.metrics import SCORERS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchmetrics
from cnn_finetune import make_model

import time

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset, Subset
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from al_utils import query_the_oracle, get_lr, sample_from_distribution

import os
import argparse

from models import *
from utils import progress_bar

import wandb

def main():
    parser = argparse.ArgumentParser(description='Active Learning Training')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for dataloader')
    parser.add_argument('--num_threads', default=8, type=int, help='number of threads for torch')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--iweight_decay', default=1e-4, type=float, help='weight decay for seed set training ')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for query rounds')
    parser.add_argument('--ilr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--inum_epoch', default=3, type=int, help='number of epochs to train for initial training')
    parser.add_argument('--num_epoch', default=3, type=int, help='number of epochs to train for each query round')
    parser.add_argument('--ischedule', default="cosine", type=str) # 'constant', 'cosine', 'find_lr'
    parser.add_argument('--schedule', default="cosine", type=str) # 'constant', 'cosine'
    parser.add_argument('--drop_last', '-d', action='store_true',
                        help='drop last batch')
    parser.add_argument('--new_model', '-n', action='store_true',
                        help='train a new model after each query round')
    parser.add_argument('--online', '-o', action='store_true',
                        help='only train newly selected datapoints for each query round')
    parser.add_argument('--no_al',  action='store_true',
                        help='train a resnet baseline')
    parser.add_argument('--dataset', default='celebA', type=str, choices=['waterbirds', 'celebA', 'domainnet', 'geo_yfcc']) 
    parser.add_argument('--use_four', action='store_true',
                        help='for domainnet dataset, only use "clipart", "painting", "real", "sketch"')
    parser.add_argument('--use_sentry', action='store_true',
                        help='for domainnet dataset, use sentry version')
    parser.add_argument('--frac', default=None, type=float, help='frac of val data to use')
    parser.add_argument('--val_size', default=None, type=int, help='validation size')
    parser.add_argument('--root_dir', default="/self/scr-sync/nlp", type=str) #root dir for accessing dataset
    # wandb params
    parser.add_argument('--wandb_group', default=None, type=str)
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_id', default=None, type=str)
    #checkpoints are saved at /nlp/scr/jiayili/pytorch-cifar/checkpoints
    parser.add_argument('--model', default="resnet18", type=str) 
    parser.add_argument('--pretrain', action='store_true',
                        help='load weights from model pretrained on IMAGENET1K_V1')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--load_train_set', action='store_true',
                        help='only load train idx from checkpoint, does not load model weights')
    parser.add_argument('--checkpoint', default="./checkpoint/al_waterbirds.pth", type=str) #checkpoint to load
    parser.add_argument('--save', '-s', action='store_true',
                        help='save checkpoint')
    parser.add_argument('--save_name', default="al_waterbirds", type=str) #checkpoint name to save
    parser.add_argument('--save_every', default=5, type=int, help='save checkpoint every # number of queries')
    # Uncertainty sampling parameters
    parser.add_argument('--seed_size', default=300, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--query_size', default=50, type=int)
    parser.add_argument('--pool_size', default=0, type=int)
    parser.add_argument('--query_strategy', default='least_confidence', type=str, 
                        choices=['least_confidence', 'margin', 'random', 'margin_logit', 'entropy',
                                 'threshold','threshold_group','threshold_group_spec','threshold_spec']) 
    parser.add_argument('--score_fn', default='MC', type=str, choices=['MC', 'NE', 'MA', 'MAL'])  
    parser.add_argument('--nocal', action='store_true', help='omit calibration when calculating threshold')
    parser.add_argument('--addcal', action='store_true', help='add calibration when doing uncertainty sampling')
    parser.add_argument('--noise', default=0.0, type=float)
    parser.add_argument('--group_strategy', default=None, type=str, 
                        choices=['oracle', 'avg_c_val', 'min', 'avg_c_train', 'label_oracle_full', 'uniform', 'loss_proportional', 'error_proportional', 'loss_exp', 'interpolate'])
    parser.add_argument('--alpha', default=0.5, type=float, help='weight to interpolate between uniform and worst group')
    parser.add_argument('--group_div', default='standard', type=str, choices=['standard', 'full', 'label']) # which group division to query
    parser.add_argument('--log_op', default=None, type=str, choices=['cal', 'log'], help='whether to cal/log full/label group info, standard group info is not affected') 
    parser.add_argument('--replacement', action='store_true', help='query with replacement')
    parser.add_argument('--exclude', default=None, type=str, help='exclude certain groups when choosing seed set')
    parser.add_argument('--uniform', action='store_true', help='uniform sample across groups for seed set')
    parser.add_argument('--distribution', default=None, type=str, help='distribution to sample across groups for seed set')

    args = parser.parse_args()

    #set default drop_last flag to True
    args.drop_last = True

    #torch.set_num_threads(args.num_threads)

    #wandb setup
    id = wandb.util.generate_id() if not args.wandb_id else args.wandb_id
    mode = 'disabled' if not (args.wandb_group and args.wandb_name) else 'online'
    name = f'{args.wandb_name}-{id}'
    wandb.init(project='al_group', entity='hashimoto-group', mode=mode, group=args.wandb_group, name=name, id=id, resume='allow', allow_val_change=True)
    def include_fn(path):
        include = False
        include_files = ["al.py", "al_utils.py", "transfer.py"]
        for f in include_files:
            if path.endswith(f): include = True
        return include
    wandb.run.log_code(root = ".", name = args.wandb_group, include_fn = include_fn) 
   
    if args.wandb_group and args.wandb_name: 
        args.save_name = f'{args.wandb_group}-{name}'
    else:
        args.save_name = f'{args.save_name}-{id}'

    wandb.config.update(vars(args), allow_val_change=True)
    #check if the current id already exists and find checkpoint to load 
    if args.wandb_id:
        checkpoint_dir = f'checkpoint/{args.save_name}'
        if os.path.isdir(checkpoint_dir):
            list = os.listdir(checkpoint_dir)
            if len(list) > 0: 
                args.resume = True
                # args.load_train_set = True
                queries = [int(file.split('.')[0]) for file in list]
                queries.sort(reverse=True)
                query = queries[0]
                args.checkpoint = f'./{checkpoint_dir}/{query}.pth'

    # Data
    print('==> Preparing data..')
    # Load the full dataset, and download it if necessary
    args.root_dir = f'/self/scr-sync/nlp/{args.dataset}'
    if args.dataset == 'geo_yfcc':
        dataset = get_dataset(dataset=args.dataset, download=False, root_dir = args.root_dir)
    elif args.dataset == 'celebA':
        target = 'Male'
        group = ['Black_Hair','Wavy_Hair']
        dataset = get_dataset(dataset=args.dataset, download=True, root_dir = args.root_dir, target=target, group=group)
        wandb.config.update({'target': target, 'group': group})
    elif args.dataset == 'domainnet':   
        dataset = get_dataset(dataset=args.dataset, download=False, root_dir = args.root_dir, use_four=args.use_four, use_sentry=args.use_sentry)   
    else:
        dataset = get_dataset(dataset=args.dataset, download=True, root_dir = args.root_dir)
   
    print(dataset.metadata_fields)
    #print(dataset.metadata_map)

    if args.dataset == 'geo_yfcc':
        grouper = CombinatorialGrouper(dataset, ['country'])
        label_grouper = CombinatorialGrouper(dataset, ['y'])
        full_grouper = CombinatorialGrouper(dataset, ['country', 'y'])
    elif args.dataset == 'celebA':
        grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-2])
        label_grouper = CombinatorialGrouper(dataset, ['y'])
        full_grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-1])
    elif args.dataset == 'waterbirds':
        grouper = CombinatorialGrouper(dataset, ['background'])
        label_grouper = CombinatorialGrouper(dataset, ['y'])
        full_grouper = CombinatorialGrouper(dataset, ['background','y'])
    elif args.dataset == 'domainnet':
        # DOMAIN_NET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        # SENTRY_DOMAINS = ["clipart", "painting", "real", "sketch"]
        grouper = CombinatorialGrouper(dataset, ['domain'])
        label_grouper = CombinatorialGrouper(dataset, ['y']) 
        full_grouper = CombinatorialGrouper(dataset, ['domain','y'])

    exclude = []
    if args.exclude:
        exclude = [int(i) for i in args.exclude.split(',')]
        print("When selecting seed set, the following groups are excluded")
        for i in exclude: 
            print(f"Group num {i}: {grouper.group_str(i)}")
        wandb.config.update({'exclude_group': exclude})
    
    if args.group_div == 'full':
        query_grouper = full_grouper 
    elif args.group_div == 'label':
        query_grouper = label_grouper
    else:
        query_grouper = grouper 

    # Get the training and validation set (transform config from https://github.com/kohpangwei/group_DRO/blob/f7eae929bf4f9b3c381fae6b1b53ab4c6c911a0e/data/cub_dataset.py#L78-L102)
    scale = 256.0/224.0
    target_resolution = (224, 224)

    train_transform = transforms.Compose([
        # let's just use default params? https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html
        # also why does the first scale parameter look so different from the default
        transforms.RandomResizedCrop(
            target_resolution,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        # note that providing both h and w will change the aspect ratio of the image
        # look here for docs: https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html
        # we can probably just pass 1 parameter to lett Resize preserve aspect ratio 
        transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
        # transforms.Resize(int(target_resolution[0]*scale)),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # train_transform = transforms.Compose(
    #     [transforms.Resize((224, 224)), transforms.ToTensor()]
    # )

    # eval_transform = transform=transforms.Compose(
    #     [transforms.Resize((224, 224)), transforms.ToTensor()]
    # )

    train_data = dataset.get_subset(
        "train",
        transform=train_transform,
    )
    train_val_data = dataset.get_subset(
        "train",
        transform=eval_transform,
    )
    raw_val_data = dataset.get_subset(
        "val",
        transform=eval_transform,
    )
    # test data not used 
    # test_data = dataset.get_subset(
    #     "test",
    #     transform=eval_transform,
    # )

    print('Train set size: ', len(train_data))
    print('Raw Eval set size: ', len(raw_val_data))
    #print('Test set size: ', len(test_data))

    _,_,_ = log_selection(np.arange(len(train_val_data)), train_val_data, 
                          grouper, full_grouper, label_grouper,                                                                        
                          0, 'data_train', args.log_op)

    counts, counts_full, counts_label = log_selection(np.arange(len(raw_val_data)), raw_val_data, 
                          grouper, full_grouper, label_grouper,                                                                        
                          0, 'raw_data_val', args.log_op)
    
    if args.group_div == 'standard':
        num_groups = counts.size
    elif args.group_div == 'full':
        num_groups = counts_full.size
    elif args.group_div == 'label':
        num_groups = counts_label.size
    
    if args.val_size is not None:
        # prepare a group_unifrom validation set 
        val_path = f"./val_data/{args.dataset}_{args.group_div}_{args.val_size}.csv"
        if os.path.exists(val_path):
            print(f"Validation file exists, loading from {val_path}")
            val_idx = pd.read_csv(val_path, header=None, index_col=False)
            val_idx = val_idx.to_numpy().squeeze()
        else:
            print(f"Validation file does not exist, sampling val indices")
            distribution = np.ones(num_groups) / num_groups
            val_idx = sample_from_distribution(distribution, raw_val_data, np.ones(len(raw_val_data)), args.val_size, query_grouper)
            print(f"Saving validation file to {val_path}")
            pd.DataFrame(val_idx).to_csv(val_path, header=False, index=False)
        val_data = WILDSSubset(raw_val_data, val_idx, transform=None)
        _,_,_ = log_selection(val_idx, raw_val_data, 
                              grouper, full_grouper, label_grouper,                                                                        
                              0, 'data_val', args.log_op)
    elif args.frac is not None:
        # prepare a validation set of size frac*len(raw_val_data) randomly sampled from raw_val_data
        val_data = dataset.get_subset(
            "val",
            transform=eval_transform,
            frac=args.frac,
        )
        _,_,_ = log_selection(np.arange(len(val_data)), val_data, 
                              grouper, full_grouper, label_grouper,                                                                        
                              0, 'data_val', args.log_op)
    else:
        val_data = raw_val_data

    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    # test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    epoch = 0  # start from epoch 0 or last checkpoint epoch
    unlabeled_mask = np.ones(len(train_data)) # We assume that in the beginning, the entire train set is unlabeled
    curr_query = 0
    curr_wg = None
    train_idx = None
    curr_test_acc = None

    num_classes = train_data.n_classes  # number of classes in the classification problem
    print(f"num_classes is {num_classes}")
    net = make_model(args.model, num_classes=num_classes, pretrained=args.pretrain)
    net = net.to(device)
    if ((args.dataset == 'waterbirds') or (args.dataset =='celebA')) and (not args.pretrain):
        print("Making old version of model")
        if args.model == 'resnet50':
            net = torchvision.models.resnet50(num_classes = num_classes)
        elif args.model == 'resnet18':
            net = torchvision.models.resnet18(num_classes = num_classes)
        net = net.to(device)
    if device == 'cuda':
        # is there a perf hit for using DataParallel on 1 gpu?
        net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss(reduction = 'none')

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        epoch = checkpoint['epoch'] 
        unlabeled_mask = checkpoint['unlabeled_mask']
        curr_query = checkpoint['curr_query']
        train_idx = checkpoint['train_idx']
        # curr_wg = checkpoint['curr_wg']
        print("Finish Loading model with accuracy {}, achieved from epoch {}, query {}, number of labeled samples {}".format(
            best_acc, epoch, curr_query, np.sum(unlabeled_mask == 0)))

        #evaluate and log
        group_counts, group_counts_full, group_counts_label = log_selection(train_idx, train_val_data, 
                                                                            grouper, full_grouper, label_grouper,                                                                        
                                                                            curr_query, 'selection_accumulating', args.log_op)
        
        curr_test_acc, val_probs, val_losses = test(epoch, net, val_data, val_loader, grouper, full_grouper, label_grouper, 
                                                                   args.log_op, criterion, device, True, curr_query)
        wg, wg_full, wg_label, g_acc, g_scores, g_losses = log_group_metrics(val_data.y_array, val_data.metadata_array, val_probs, val_losses, 
                                                  device, args.log_op, epoch, curr_query, 
                                                  grouper, full_grouper, label_grouper, args.group_div, 'val_per_query')

        # find worst group according to args.group_strategy and args.group_div
        curr_wg = find_target_group(args.group_strategy, args.group_div, val_probs, val_data.metadata_array, val_probs, val_data.metadata_array,
                                    wg, wg_full, wg_label, 
                                    group_counts, group_counts_full, group_counts_label, 
                                    query_grouper, label_grouper, full_grouper
                                    )
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    else:
        # Initialize counters
        round_step = 0 #keeps track of number of steps for this query round, not really used
        #query_start_epoch = np.zeros(args.num_queries + 1) # store the start epoch index for each query; the first query is the initial seed set with start epoch 0

        # Label the initial subset
        if args.no_al:
            unlabeled_mask = np.zeros(len(train_data))
            args.num_queries = 0
            train_idx = np.arange(len(train_data))
        elif args.load_train_set:
            print('==> Loading train idx from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(args.checkpoint)
            unlabeled_mask = checkpoint['unlabeled_mask']
            train_idx = checkpoint['train_idx']
            print(f'Finish Loading train idx of size {train_idx.size}')
        else:
            # sample the seed set based on given distribution
            distribution = None 
            if args.uniform:
                distribution = np.ones(num_groups) / num_groups
            if args.distribution is not None: 
                raw_prob = args.distribution.split(',')
                distribution = [float(i) for i in raw_prob]
                distribution = np.array(distribution)
                assert distribution.size == num_groups, "Distribution needs to have correct number of groups"
            if distribution is not None: 
                # idx = sample_from_distribution(distribution, train_val_data, unlabeled_mask, args.seed_size, query_grouper)
                idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, val_data, query_grouper, args.seed_size,
                                sample_distribution=distribution, group_strategy=None, exclude=exclude, wg=curr_wg, query_strategy='random', 
                                replacement=args.replacement, pool_size=0, batch_size=args.batch_size, num_workers = args.num_workers)
            else:
                # randomly sample the seed set
                idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, val_data, query_grouper, args.seed_size,
                                 group_strategy=None, exclude=exclude, wg=curr_wg, query_strategy='random', 
                                replacement=args.replacement, pool_size=0, batch_size=args.batch_size, num_workers = args.num_workers)
            
            _,_,_ = log_selection(idx, train_val_data, 
                                 grouper, full_grouper, label_grouper,                                                                        
                                 curr_query, 'selection_per_query', args.log_op)
            train_idx = idx

        # Prepare train loader
        group_counts, group_counts_full, group_counts_label = log_selection(train_idx, train_val_data, 
                                                                            grouper, full_grouper, label_grouper,                                                                        
                                                                            curr_query, 'selection_accumulating', args.log_op)        
        data_size = np.sum(unlabeled_mask == 0) #keeps track of number of distinct labeled datapoints
        curr_train_data = WILDSSubset(train_data, train_idx, transform=None)
        train_loader = get_train_loader("standard", curr_train_data, 
                                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
        
        # Initialize optimizer and scheduler
        optimizer = optim.SGD(net.parameters(), lr=args.ilr,
                              momentum=0.9, weight_decay=args.iweight_decay)
        scheduler = init_optimizer_scheduler(args.ischedule, args.inum_epoch, train_loader, optimizer)

        # if args.no_al and args.schedule == "cosine":
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.inum_epoch)

        # Pre-train on the initial subset
        for i in range(args.inum_epoch):
            query_end = (i==args.inum_epoch-1)
            train_group = 'train_per_query' if query_end else 'train'
            # val_group = 'val_per_query' if query_end else 'val'
            _, _, train_probs, train_losses, curr_y_array, curr_meta_array = train(epoch, round_step, net, train_loader, args.batch_size, data_size, 
                                                    optimizer, scheduler, criterion, device, query_end, curr_query, args.no_al) 
            log_group_metrics(curr_y_array, curr_meta_array, train_probs, train_losses, device, args.log_op, 
                              epoch, curr_query, grouper, full_grouper, label_grouper, args.group_div, train_group)
                                           
            # if args.no_al:
            #     scheduler.step()
            epoch += 1
            round_step += len(train_loader)
        curr_test_acc, val_probs, val_losses = test(epoch, net, val_data, val_loader, grouper, full_grouper, label_grouper, 
                                                    args.log_op, criterion, device, True, curr_query)
        wg, wg_full, wg_label, g_acc, g_scores, g_losses = log_group_metrics(val_data.y_array, val_data.metadata_array, val_probs, val_losses, device, args.log_op, 
                          epoch, curr_query, grouper, full_grouper, label_grouper, args.group_div, 'val_per_query')
                          
        
       # find worst group according to args.group_strategy and args.group_div
        curr_wg = find_target_group(args.group_strategy, args.group_div, val_probs, val_data.metadata_array, train_probs, curr_meta_array,
                                    wg, wg_full, wg_label, 
                                    group_counts, group_counts_full, group_counts_label, 
                                    query_grouper, label_grouper, full_grouper
                                    )

        save_checkpoint(args.save, curr_query % args.save_every == 0, net, curr_test_acc, epoch,
                        curr_query, unlabeled_mask, train_idx, curr_wg, args.save_name, wandb.run.name)

    # Start the query loop 
    for query in range(args.num_queries - curr_query):
        #print(query_start_epoch)
        #query_start_epoch[query + 1] = epoch
        curr_query += 1
        round_step = 0

        # Query the oracle for more labels
        idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, val_data, query_grouper, args.query_size, 
                               val_probs=val_probs, group_strategy=args.group_strategy, p=args.alpha, wg=curr_wg, query_strategy=args.query_strategy, 
                               group_losses=g_losses, group_acc=g_acc, score_fn=args.score_fn, calibration=(not args.nocal), addcal=args.addcal, noise=args.noise,
                               replacement=args.replacement, pool_size=args.pool_size, batch_size=args.batch_size, num_workers=args.num_workers)
        _,_,_ = log_selection(idx, train_val_data, 
                              grouper, full_grouper, label_grouper,                                                                        
                              curr_query, 'selection_per_query', args.log_op)
        train_idx = np.append(train_idx, idx)

        # If passed args.new_model, train a new model in each query round
        if args.new_model: 
            net = make_model(args.model, num_classes=num_classes, pretrained=args.pretrain)
            if ((args.dataset == 'waterbirds') or (args.dataset =='celebA')) and (not args.pretrain):
                if args.model == 'resnet50':
                    net = torchvision.models.resnet50(num_classes = num_classes)
                elif args.model == 'resnet18':
                    net = torchvision.models.resnet18(num_classes = num_classes)
            net = net.to(device)
            optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
            #     cudnn.benchmark = True
        else:
            for g in optimizer.param_groups:
                g['lr'] = args.lr

        # Prepare train loader
        group_counts, group_counts_full, group_counts_label = log_selection(train_idx, train_val_data, 
                                                                            grouper, full_grouper, label_grouper,                                                                        
                                                                            curr_query, 'selection_accumulating', args.log_op)
        data_size = np.sum(unlabeled_mask == 0)
        curr_train_data = WILDSSubset(train_data, train_idx, transform=None)
        train_loader = get_train_loader("standard", curr_train_data, 
                                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
        
        if args.online: #only train on newly labeled datapoints
            train_loader = get_train_loader("standard", WILDSSubset(train_data, idx, transform=None), 
                                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False) # to use the full data, drop_last is set to false and train() has been modified to handle partially full batch
        
        # Reinitialize scheduler
        scheduler = init_optimizer_scheduler(args.schedule, args.num_epoch, train_loader, optimizer)

        # Train the model on the data that has been labeled so far:
        for i in range(args.num_epoch):
            query_end = (i==args.num_epoch-1)
            _, _, train_probs, train_losses, curr_y_array, curr_meta_array = train(epoch, round_step, net, train_loader, args.batch_size, data_size, 
                                      optimizer, scheduler, criterion, device, query_end, curr_query, args.no_al)
            epoch += 1
            round_step += len(train_loader)
            if query_end:
                log_group_metrics(curr_y_array, curr_meta_array, train_probs, train_losses, device, args.log_op, 
                        epoch, curr_query, grouper, full_grouper, label_grouper, args.group_div, 'train_per_query')
        
        # To speed up training, only evaluate at the end of query 
        curr_test_acc, val_probs, val_losses = test(epoch, net, val_data, val_loader, grouper, full_grouper, label_grouper, 
                                                                   args.log_op, criterion, device, True, curr_query)

        wg, wg_full, wg_label, g_acc, g_scores, g_losses = log_group_metrics(val_data.y_array, val_data.metadata_array, val_probs, val_losses, device, args.log_op, 
                                                    epoch, curr_query, grouper, full_grouper, label_grouper, args.group_div, 'val_per_query')
                                                    
       
        # find worst group according to args.group_strategy and args.group_div
        curr_wg = find_target_group(args.group_strategy, args.group_div, val_probs, val_data.metadata_array, train_probs, curr_meta_array,
                                    wg, wg_full, wg_label, 
                                    group_counts, group_counts_full, group_counts_label, 
                                    query_grouper, label_grouper, full_grouper
                                    )

        save_checkpoint(args.save, curr_query % args.save_every == 0, net, curr_test_acc, epoch,
                        curr_query, unlabeled_mask, train_idx, curr_wg, args.save_name, wandb.run.name)


def find_target_group(group_strategy, group_div, val_prob, val_meta, train_prob, train_meta,
                      wg, wg_full, wg_label, 
                      group_ct, group_ct_full, group_ct_label,
                      query_grouper, label_grouper=None, full_grouper=None
                      ):
    curr_wg = None
    if group_strategy == 'oracle':
        curr_wg = wg
        if group_div == 'full': curr_wg = wg_full
        if group_div == 'label': curr_wg = wg_label 
    elif group_strategy == 'label_oracle_full':
        assert (label_grouper is not None) and (full_grouper is not None), "Group strategy is label_oracle_full, label_grouper or full_grouper is not given"
        assert wg_full is not None, "Group strategy is label_oracle_full, valid wg_full should be given"
        curr_wg = extract_label_group(wg_full, val_meta, label_grouper, full_grouper)
    elif group_strategy == 'min':
        curr_wg = np.argmin(group_ct)
        if group_div == 'full': curr_wg = np.argmin(group_ct_full)
        if group_div == 'label': curr_wg = np.argmin(group_ct_label)
        #print("Group strategy is min, the smallest group is {}".format(grouper.group_str(curr_wg)))
    elif group_strategy == 'avg_c_val':
        curr_wg = get_avgc_worst_group(val_prob, query_grouper, val_meta)
    elif group_strategy == 'avg_c_train':
        curr_wg = get_avgc_worst_group(train_prob, query_grouper, train_meta)
    return curr_wg

def extract_label_group(wg_full, meta_array, label_grouper, full_grouper):
    group = full_grouper.metadata_to_group(meta_array)
    group = np.asarray(group.cpu())
    idx = np.nonzero(group == wg_full)[0]
    label_group = label_grouper.metadata_to_group(meta_array)
    label_group = np.asarray(label_group.cpu())
    assert np.sum(label_group[idx]-np.mean(label_group[idx]))==0, f"Extracting label idx from group {query_grouper.group_str(wg_full)}, not all labels are same"
    label_idx = label_group[idx[0]]
    print(f"Extracting label idx from group {full_grouper.group_str(wg_full)}, label group is {label_grouper.group_str(label_idx)}")
    return label_idx

def get_avgc_worst_group(probabilities, grouper, meta_array):
    #start_time = time.time()
    confidences = torch.max(probabilities, dim=1)[0]
    confidences = np.asarray(confidences.cpu())
    meta_array = meta_array.cpu()
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    group = np.array(group)
    group_counts = np.array(group_counts)
    group_avg_c = np.zeros(len(group_counts))
    for i in range(len(group_counts)):
      group_avg_c[i] = np.mean(confidences[np.nonzero(group == i)[0]])
    worst_group = np.argmin(group_avg_c)
    wg_avg_c = group_avg_c[worst_group]
    #print("--- get_avgc_worst_group takes %s seconds ---" % (time.time() - start_time))
    #print(group_avg_c)
    #print("Worst group is {}: {} with average confidence score {}".format(worst_group, grouper.group_str(worst_group), wg_avg_c))
    return worst_group

def init_optimizer_scheduler(schedule, num_epoch, train_loader, optimizer):
    # optimizer = optim.SGD(net.parameters(), lr=lr,
    #             momentum=0.9, weight_decay=wd)
    if schedule == "constant":
        lambda_cons = lambda epoch: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_cons, last_epoch=-1, verbose=False)
    elif schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch*len(train_loader))
    elif schedule == "find_lr":
        lambda_cons = lambda epoch: 1.05**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_cons, last_epoch=-1, verbose=False)
    return scheduler

def log_selection(train_idx, train_val_data, grouper, full_grouper, label_grouper, curr_query, wandb_group, log_op):
    group_counts = print_log_selection_info(train_idx, train_val_data, grouper, curr_query, wandb_group, 'standard', log_op)
    group_counts_full = print_log_selection_info(train_idx, train_val_data, full_grouper, curr_query, wandb_group, 'full', log_op)
    group_counts_label = print_log_selection_info(train_idx, train_val_data, label_grouper, curr_query, wandb_group, 'label', log_op)
    return group_counts, group_counts_full, group_counts_label

def print_log_selection_info(idx, dataset, grouper, curr_query, wandb_name, group_type, log_op):
    if group_type != 'standard' and (not log_op): return None
    selected = WILDSSubset(dataset, idx, transform=None)
    meta_array = selected.metadata_array
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    group_counts = np.array(group_counts)
    query_info = dict()
    total = np.sum(group_counts)
    assert total == idx.size, "Sum of group counts does not match num of input datapoints"
    if (group_type == 'standard') or (log_op == 'log'): 
        for i in range(len(group_counts)):
            query_info[f"{wandb_name}_{group_type}/{grouper.group_str(i)}"] = 100.0 * group_counts[i]/total
            print("{}, group: {}, count: {}, percentage:{} \n".format(wandb_name, grouper.group_str(i), group_counts[i], 100.0 *group_counts[i]/total))
        wandb.log(dict(**query_info, **{"general/curr_query":curr_query}))
    return group_counts

def log_group_metrics(y_array, meta_array, probabilities, losses, device, log_op, epoch, curr_query,
                     grouper, full_grouper, label_grouper, group_div='standard', wandb_group='val'):
    worst_group, results, g_acc_s, g_scores_s, g_losses_s = get_worst_group(y_array, meta_array, grouper, probabilities, losses, device, log_op, prefix='standard') 
    worst_group_full, results_full, g_acc_f, g_scores_f, g_losses_f = get_worst_group(y_array, meta_array, full_grouper, probabilities, losses, device, log_op, prefix='full') 
    worst_group_label, results_label, g_acc_l, g_scores_l, g_losses_l = get_worst_group(y_array, meta_array, label_grouper, probabilities, losses, device, log_op, prefix='label') 
    if results_full is not None: results.update(results_full)
    if results_label is not None:results.update(results_label)
    target = y_array.to(device).int()
    auc = calculate_auc(probabilities, target)
    log_wandb(epoch, curr_query, auc, results, wandb_group=wandb_group)

    if group_div == 'standard':
        g_acc = g_acc_s
        g_scores = g_scores_s
        g_losses = g_losses_s
    elif group_div == 'full':
        g_acc = g_acc_f
        g_scores = g_scores_f
        g_losses = g_losses_f
    elif group_div == 'label':
        g_acc = g_acc_l
        g_scores = g_scores_l
        g_losses = g_losses_l
    print(f"Group_accuracies are {g_acc}")

    return worst_group, worst_group_full, worst_group_label, g_acc, g_scores, g_losses

# output worst performing group and group-wise metrics(for wandb log)
def get_worst_group(y_array, meta_array, grouper, probabilities, losses, device, log_op, prefix = 'standard'):
    if (prefix != 'standard') and (log_op is None): return None, None, None, None, None 
    probabilities = probabilities.to(device)
    confidences, predictions = probabilities.max(1)
    y_array = y_array.to(device).int()
    meta_array = meta_array.cpu()
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    results = {}
    num_group = torch.numel(group_counts)
    acc = np.zeros(num_group)
    auc = np.zeros(num_group)
    group_scores = np.zeros(num_group)
    group_losses = np.zeros(num_group)
    for i in range(num_group):
        group_idx = torch.nonzero(group == i).squeeze()
        if group_counts[i] > 0: 
            acc[i] = torch.sum(y_array[group_idx] == predictions[group_idx])/group_counts[i]
            group_scores[i] = torch.sum(confidences[group_idx])/group_counts[i]
            group_losses[i] = torch.sum(losses[group_idx])/group_counts[i]
        else: 
            acc[i] = 1.01
        auc[i] = calculate_auc(probabilities, y_array)
        if (prefix == 'standard') or (log_op == 'log'):
            results.update({f'{prefix}_acc_{grouper.group_str(i)}':acc[i]})
            results.update({f'{prefix}_score_{grouper.group_str(i)}':group_scores[i]})
            results.update({f'{prefix}_loss_{grouper.group_str(i)}':group_losses[i]})
            #results.update({f'{prefix}_auc_{grouper.group_str(i)}':auc[i]})
    # accuracy worst group
    worst_group = np.argmin(acc)
    wg_acc = acc[worst_group]
    results.update({f"{prefix}_wg_acc": wg_acc, f'{prefix}_wg': worst_group, f'{prefix}_mean_acc': np.mean(acc)})
    # auc worst group
    auc_worst_group = np.argmin(auc)
    wg_auc = auc[auc_worst_group]
    results.update({f"{prefix}_auc_wg_sc": wg_auc, f'{prefix}_auc_wg': auc_worst_group})
    # error/loss worst group
    loss_wg = np.argmax(group_losses)
    wg_loss = group_losses[loss_wg]
    results.update({f"{prefix}_wg_loss": wg_loss, f'{prefix}_loss_wg': loss_wg})
    print("Worst group is {}: {} with acc {}".format(worst_group, grouper.group_str(worst_group), wg_acc))
    print("AUC Worst group is {}: {} with auc {}".format(auc_worst_group, grouper.group_str(auc_worst_group), wg_auc))
    print("Error Worst group is {}: {} with error {}".format(loss_wg, grouper.group_str(loss_wg), wg_loss))
    return worst_group, results, acc, group_scores, group_losses

def log_wandb(epoch, curr_query, auc, results, wandb_group='val'):
    log_dict = {"general/epoch": epoch, "general/curr_query": curr_query, f'{wandb_group}/auc': auc}
    for key in results:
        group_type = key.split('_')[0]
        log_dict.update({f'{wandb_group}:{group_type}/{key}':results[key]})
    wandb.log(log_dict)

def calculate_auc(probabilities, target):
    n_classes = probabilities.shape[1]
    if n_classes == 2:
        preds = probabilities[:,1].squeeze()
        auc = torchmetrics.functional.auroc(preds, target)
    else:
        auc = torchmetrics.functional.auroc(probabilities, target, num_classes=n_classes, average='weighted')
    return auc


# Train
def train(epoch, step, net, dataloader, batch_size, data_size, optimizer, scheduler, criterion, device, query_end, curr_query, no_al):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_samples = 0
    probabilities = torch.tensor([]).to(device)
    losses = torch.tensor([]).to(device)
    y_array = torch.tensor([]).to(device)
    meta_array = torch.tensor([]).to(device)
    for batch_idx, (inputs, targets, metadata) in enumerate(dataloader):
        step += 1
        # For specific lr scheduling defined in al_utils.py
        # lr = get_lr(step, data_size, args.lr)
        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        inputs, targets = inputs.to(device), targets.to(device)
        y_array = torch.cat((y_array, targets))
        meta_array = torch.cat((meta_array, metadata.to(device)))
        num_samples += inputs.size()[0]
        optimizer.zero_grad()
        outputs = net(inputs)
        probs = F.softmax(outputs, dim=1)
        probabilities = torch.cat((probabilities, probs))
        loss = criterion(outputs, targets)
        losses = torch.cat((losses, loss))
        loss = torch.mean(loss)
        lr = scheduler.get_last_lr()[0]
        wandb.log({"general/epoch": epoch, "train/train_step_loss":loss.item(), "train/lr": lr})
        loss = loss * inputs.size()[0] / batch_size
        loss.backward()
        
        optimizer.step()
        #if not no_al:
        scheduler.step()
        train_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Learning rate: %.6f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (lr, train_loss/num_samples, 100.*correct/total, correct, total))

    wandb.log({"general/epoch": epoch, "train/train_epoch_loss": train_loss/num_samples, "train/train_acc":100.*correct/total})
    
    # if the current epoch is the last of the query round 
    if query_end:
        wandb.log({"general/epoch": epoch, "general/data_size": data_size, "general/curr_query": curr_query,
                "train_per_query/query_end_train_loss": train_loss/num_samples, "train_per_query/query_end_train_acc":100.*correct/total})
       
    probabilities = probabilities.detach()
    losses = losses.detach()
    y_array = y_array.detach().int()
    meta_array = meta_array.detach()
    return train_loss/len(dataloader), 100.*correct/total, probabilities, losses, y_array, meta_array

# Test
def test(epoch, net, dataset, dataloader, grouper, full_grouper, label_grouper, log_op, criterion, device, query_end, curr_query):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_samples = 0
    probabilities = torch.tensor([]).to(device)
    # predictions = None
    losses = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            num_samples += inputs.size()[0]
            outputs = net(inputs)

            probs = F.softmax(outputs, dim=1)
            # if probabilities is None:
            #     probabilities = probs
            # else:
            probabilities = torch.cat((probabilities, probs))

            loss = criterion(outputs, targets)
            losses = torch.cat((losses.to(device), loss))
            loss = torch.mean(loss)
            test_loss += loss * inputs.size()[0]
            _, predicted = outputs.max(1)

            # if predictions is None:
            #     predictions = predicted
            # else:
            #     predictions = torch.cat((predictions, predicted))

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/num_samples, 100.*correct/total, correct, total))
    wandb.log({"general/epoch": epoch, "val/epoch_loss":test_loss/num_samples, "val/test_acc":100.*correct/total})

    # if the current epoch is the last of the query round 
    if query_end:
        wandb.log({"general/epoch": epoch,  "general/curr_query": curr_query,
                 "val_per_query/test_loss": test_loss/num_samples, "val_per_query/test_acc":100.*correct/total})

    acc = 100.*correct/total
   
    # #oracle worst group
    # worst_group, results = get_worst_group(dataset, grouper, probabilities, device, log_op, prefix='standard') 
    # worst_group_full, results_full = get_worst_group(dataset, full_grouper, probabilities, device, log_op, prefix='full') 
    # worst_group_label, results_label = get_worst_group(dataset, label_grouper, probabilities, device, log_op, prefix='label') 
    # if results_full is not None: results.update(results_full)
    # if results_label is not None:results.update(results_label)

    # target = dataset.y_array.to(device)
    # auc = calculate_auc(probabilities, target)
    # #print(f"auc is {auc}")

    # # if dataset.dataset_name == 'waterbirds':
    # #     y_array = dataset.y_array.cpu()
    # #     meta_array = dataset.metadata_array.cpu()
    # #     results, result_str = dataset.eval(predictions.cpu(), y_array, meta_array)
    # #     #print(result_str)
    # #     log_waterbirds(epoch, curr_query, 100.*correct/total, auc, results, wandb_group='val')
    # #     acc = results['adj_acc_avg']
    # # else:
        
    # log_wandb(epoch, curr_query, auc, results, wandb_group='val')

    # # if the current epoch is the last of the query round
    # if query_end:
    # #    if dataset.dataset_name == 'waterbirds':
    # #        log_waterbirds(epoch, curr_query, 100.*correct/total, auc, results, wandb_group='val_per_query')
    # #    else:
    #     log_wandb(epoch, curr_query, auc, results, wandb_group='val_per_query')

    return acc, probabilities, losses

def save_checkpoint(save, condition, net, acc, epoch, curr_query, unlabeled_mask, train_idx, curr_wg, save_name, run_name):
    if save and condition:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            "curr_query": curr_query,
            'unlabeled_mask': unlabeled_mask,
            'train_idx': train_idx,
            'curr_wg': curr_wg
        }
        if not os.path.isdir(f'./checkpoint/{save_name}'):
            os.mkdir(f'./checkpoint/{save_name}')
        torch.save(state, f'./checkpoint/{save_name}/{curr_query}.pth')


if __name__ == "__main__":
    main()
