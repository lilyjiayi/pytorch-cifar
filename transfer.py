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

import numpy as np
import pandas as pd
import os
import argparse
import time
import random

from torch.utils.data import DataLoader, Dataset, Subset
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from al_utils import query_the_oracle, get_lr, atc, sample_from_distribution
from models import *
from utils import progress_bar

import wandb

def main():
    parser = argparse.ArgumentParser(description='Active Learning Training')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for dataloader')
    parser.add_argument('--num_threads', default=8, type=int, help='number of threads for torch')
    parser.add_argument('--iweight_decay', default=1e-4, type=float, help='weight decay for seed set training ')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for query rounds')
    parser.add_argument('--ilr', default=5e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--inum_epoch', default=100, type=int, help='number of epochs to train for initial training')
    parser.add_argument('--num_epoch', default=100, type=int, help='number of epochs to train for each query round')
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
    parser.add_argument('--dataset', default='domainnet', type=str, choices=['waterbirds', 'celebA', 'domainnet']) 
    parser.add_argument('--use_four', action='store_true',
                        help='for domainnet dataset, only use "clipart", "painting", "real", "sketch"')
    parser.add_argument('--use_sentry', action='store_true',
                        help='for domainnet dataset, use sentry version')
    parser.add_argument('--frac', default=1, type=float, help='frac of val data to use')
    parser.add_argument('--target', default='Male', type=str)
    parser.add_argument('--root_dir', default="/self/scr-sync/nlp/waterbirds", type=str) #root dir for accessing dataset
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
                        help='only load train idx from checkpoint')
    parser.add_argument('--checkpoint', default="./checkpoint/al_waterbirds.pth", type=str) #checkpoint to load
    parser.add_argument('--save', '-s', action='store_true',
                        help='save checkpoint')
    parser.add_argument('--save_name', default="al_waterbirds", type=str) #checkpoint name to save
    parser.add_argument('--save_every', default=1, type=int, help='save checkpoint every # number of queries')
    # Uncertainty sampling parameters
    parser.add_argument('--seed_size', default=10000, type=int)
    parser.add_argument('--num_queries', default=0, type=int)
    parser.add_argument('--query_size', default=1000, type=int)
    parser.add_argument('--pool_size', default=0, type=int)
    parser.add_argument('--query_strategy', default='least_confidence', type=str, choices=['least_confidence', 'margin', 'random', 'threshold']) 
    parser.add_argument('--group_strategy', default=None, type=str, choices=['oracle', 'avg_c_val', 'min']) # 'oracle', 'avg_c', 'avg_c_val', 'min'
    parser.add_argument('--group_div', default='standard', type=str, choices=['standard', 'full', 'label']) # which group division to query
    parser.add_argument('--log_op', default=None, type=str, choices=['cal', 'log'], help='whether to cal/log full/label group info, standard group info is not affected') 
    parser.add_argument('--replacement', action='store_true', help='query with replacement')
    parser.add_argument('--exclude', default=None, type=str, help='exclude certain groups when choosing seed set')
    parser.add_argument('--include', default=None, type=str, help='include certain groups when choosing seed set')
    parser.add_argument('--uniform', action='store_true', help='uniform sample across groups for seed set')
    parser.add_argument('--distribution', default=None, type=str, help='distribution to sample across groups for seed set')

    args = parser.parse_args()

    #set default drop_last flag to True
    args.drop_last = True
    # if group_div is 'full' or 'label', log_op is automatically set to true
    if args.group_div != 'standard': args.log_op = True
    if args.dataset == 'celebA': args.log_op = True

    #torch.set_num_threads(args.num_threads)

    #wandb setup
    id = wandb.util.generate_id() if not args.wandb_id else args.wandb_id
    mode = 'disabled' if not (args.wandb_group and args.wandb_name) else 'online'
    # mode = 'disabled'
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
    #check if the current id already exists and find checkpoint to load 
    if args.wandb_id:
        checkpoint_dir = f'checkpoint/{args.save_name}'
        if os.path.isdir(checkpoint_dir):
            # list = os.listdir(checkpoint_dir)
            # if len(list) > 0: 
            #     args.resume = True
            #     queries = [int(file.split('.')[0]) for file in list]
            #     queries.sort(reverse=True)
            #     query = queries[0]
            #     args.checkpoint = f'./{checkpoint_dir}/{query}.pth'
            args.resume = True
            args.checkpoint = f'./{checkpoint_dir}/0.pth'


    wandb.config.update(vars(args), allow_val_change=True)

    # Data
    print('==> Preparing data..')
    # Load the full dataset, and download it if necessary
    args.root_dir = f'/self/scr-sync/nlp/{args.dataset}'
    if args.dataset == 'celebA':
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

    if args.dataset == 'celebA':
        grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-2])
        label_grouper = CombinatorialGrouper(dataset, ['y'])
        full_grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-1])
        if args.group_div == 'full':
            query_grouper = full_grouper 
        elif args.group_div == 'label':
            query_grouper = label_grouper
        else:
            query_grouper = grouper
        exclude = []
        if args.exclude:
            exclude = [int(i) for i in args.exclude.split(',')]
            print("When selecting seed set, the following groups are excluded")
            for i in exclude: 
                print(f"Group num {i}: {grouper.group_str(i)}")
            wandb.config.update({'exclude_group': exclude})
    elif args.dataset == 'waterbirds':
        grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-1])
        query_grouper = grouper
        label_grouper = None
        full_grouper = None
        exclude = []
    elif args.dataset == 'domainnet':
        # DOMAIN_NET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        # SENTRY_DOMAINS = ["clipart", "painting", "real", "sketch"]
        grouper = CombinatorialGrouper(dataset, ['domain'])
        label_grouper = CombinatorialGrouper(dataset, ['y']) 
        full_grouper = CombinatorialGrouper(dataset, ['domain','y'])
        if args.group_div == 'full':
            query_grouper = full_grouper 
        elif args.group_div == 'label':
            query_grouper = label_grouper
        else:
            query_grouper = grouper 
        exclude = None
        if args.exclude:
            exclude = [int(i) for i in args.exclude.split(',')]
            print("When selecting seed set, the following groups are excluded")
            for i in exclude: 
                print(f"Group num {i}: {grouper.group_str(i)}")
            wandb.config.update({'exclude_group': exclude})
        include = None
        if args.include:
            include = [int(i) for i in args.include.split(',')]
            print("When selecting seed set, the following groups are included")
            for i in include: 
                print(f"Group num {i}: {grouper.group_str(i)}")
            wandb.config.update({'include_group': include})


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
        #transforms.Resize(int(target_resolution[0]*scale)),
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
        frac=1,
    )
    small_val_data = dataset.get_subset(
        "val",
        transform=eval_transform,
        frac=0.01,
    )
    
    # test data not used 
    # test_data = dataset.get_subset(
    #     "test",
    #     transform=eval_transform,
    # )

    print('Train set size: ', len(train_data))
    print('Eval set size: ', len(raw_val_data))
    print('Small eval set size: ', len(small_val_data))

    # _,_,_ = log_selection(np.arange(len(train_val_data)), train_val_data, 
    #                       grouper, full_grouper, label_grouper,                                                                        
    #                       0, 'data_train', args.log_op)

    # _,_,_ = log_selection(np.arange(len(raw_val_data)), raw_val_data, 
    #                       grouper, full_grouper, label_grouper,                                                                        
    #                       0, 'data_val', args.log_op)


    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
    val_loader = get_eval_loader("standard", raw_val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    small_val_loader = get_eval_loader("standard", small_val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    # test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # Read in the group-uniform val selection
    val_idx = pd.read_csv(f"./transfer_data/val.csv", header=None, index_col=False)
    val_idx = val_idx.to_numpy().squeeze()
    group_counts = print_log_selection_info(val_idx, raw_val_data, grouper, 0, f"Group-uniform val selection", "standard", args.log_op)
    val_data = WILDSSubset(raw_val_data, val_idx, transform=None)
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    num_groups = group_counts.size

      
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
    net = make_model(args.model, num_classes=num_classes, pretrained=args.pretrain)
    if ((args.dataset == 'waterbirds') or (args.dataset =='celebA')) and (not args.pretrain):
        if args.model == 'resnet50':
            net = torchvision.models.resnet50(num_classes = num_classes)
        elif args.model == 'resnet18':
            net = torchvision.models.resnet18(num_classes = num_classes)

    net = net.to(device)
    # if device == 'cuda':
    #     # is there a perf hit for using DataParallel on 1 gpu?
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss(reduction = 'none')

    # prepare validation set for each group
    # group_val_size = 10000
    # group, group_counts = grouper.metadata_to_group(val_data.metadata_array, return_counts=True)
    # group = np.array(group)
    # group_counts = np.array(group_counts)
    # val_idx = None 
    # for i in range(group_counts.size):
    #     group_idx = pd.read_csv(f"./transfer_data/val_{i}.csv", header=None, index_col=False)
    #     group_idx = group_idx.to_numpy().squeeze()
    #     if val_idx is None:
    #         val_idx = group_idx
    #     else:
    #         val_idx = np.append(val_idx, group_idx)
    #     print(f"group_idx shape is {group_idx.shape}")
    #     # group_idx = query_the_oracle(np.ones(group.size), net, device, val_data, val_data, grouper, query_size=group_val_size,
    #     #                              group_strategy=None, include=[i], query_strategy='random', batch_size=args.batch_size, num_workers = args.num_workers)
    #     print_log_selection_info(group_idx, val_data, grouper, curr_query, f"Group {i} val selection", "standard", args.log_op)
    #     # pd.DataFrame(group_idx).to_csv(f"./transfer_data/val_{i}.csv", header=False, index=False)
    # pd.DataFrame(val_idx).to_csv(f"./transfer_data/val.csv", header=False, index=False)

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
        # curr_test_acc, wg, wg_full, wg_label, probabilities, group_accuracies, group_scores, group_losses = test(epoch, net, val_data, val_loader, grouper, full_grouper, label_grouper, 
        #                                                            args.log_op, criterion_val, device, True, curr_query)

        # # find worst group according to args.group_strategy and args.group_div
        # curr_wg = find_target_group(args.group_strategy, args.group_div, probabilities, val_data, 
        #                             wg, wg_full, wg_label, 
        #                             group_counts, group_counts_full, group_counts_label, 
        #                             query_grouper
        #                             )
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
            # randomly sample the seed set
            distribution = None 
            if args.uniform:
                distribution = np.ones(num_groups) / num_groups
            if args.distribution is not None: 
                raw_prob = args.distribution.split(',')
                distribution = [float(i) for i in raw_prob]
                distribution = np.array(distribution)
                assert distribution.size == num_groups, "Distribution needs to have correct number of groups"
            if distribution is not None: 
                idx = sample_from_distribution(distribution, train_val_data, unlabeled_mask, args.seed_size, query_grouper)
            else:
                idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, val_data, query_grouper, query_size=args.seed_size,
                                group_strategy=None, exclude=exclude, include=include, wg=curr_wg, query_strategy='random', 
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
        train_loader = DataLoader(
                    curr_train_data,
                    shuffle=False, # Shuffle training dataset
                    collate_fn=curr_train_data.collate,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers, 
                    drop_last=False)
        # train_loader = get_train_loader("standard", curr_train_data, 
        #                                 batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
        
        # Initialize optimizer and scheduler
        optimizer, scheduler = init_optimizer_scheduler(net, args.ilr, args.ischedule, args.inum_epoch, train_loader, args.iweight_decay)

        # if args.no_al and args.schedule == "cosine":
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.inum_epoch)

        # Pre-train on the initial subset
        for i in range(args.inum_epoch):
            query_end = (i==args.inum_epoch-1)
            curr_train_loss, curr_train_acc = train(epoch, round_step, net, train_loader, args.batch_size, data_size, 
                                                    optimizer, scheduler, criterion, device, query_end, curr_query, args.no_al)
            # curr_test_acc, wg, wg_full, wg_label, probabilities, group_accuracies, group_scores, group_losses = test(epoch, net, small_val_data, small_val_loader, grouper, full_grouper, label_grouper, 
            #                                                        args.log_op, criterion_val, device, False, curr_query)
            # if args.no_al:
            #     scheduler.step()
            epoch += 1
            round_step += len(train_loader)
        
    curr_test_acc, wg, wg_full, wg_label, probabilities, group_accuracies, group_scores, group_losses = test(epoch, net, val_data, val_loader, grouper, full_grouper, label_grouper, 
                                                                   args.log_op, criterion_val, device, True, curr_query)

    # find worst group according to args.group_strategy and args.group_div
    curr_wg = find_target_group(args.group_strategy, args.group_div, probabilities, val_data, 
                                wg, wg_full, wg_label, 
                                group_counts, group_counts_full, group_counts_label, 
                                query_grouper
                                )
    if not args.resume: 
        save_checkpoint(args.save, curr_query % args.save_every == 0, net, curr_test_acc, epoch,
                        curr_query, unlabeled_mask, train_idx, curr_wg, args.save_name, wandb.run.name)

    group_dist = group_counts/np.sum(group_counts)
    val_sample = sample_from_distribution(group_dist, raw_val_data, np.ones(len(raw_val_data)), 1000, grouper)
    _, _, _ = log_selection(val_sample, raw_val_data, 
                            grouper, full_grouper, label_grouper,                                                                        
                            curr_query, None, args.log_op) 
    atc_acc, atc_groups, _, atc_prec_recall = atc(net, device, val_data, WILDSSubset(raw_val_data, val_sample, transform=None), 
                                grouper, args.batch_size, args.num_workers, test_probs=probabilities)     


    print(f"ATC accuracy predicted from source val data is {atc_acc}")

    print(f"For initial training, group counts are {group_counts}")
    print(f"For initial training, group accuracies are {group_accuracies}")
    print(f"For initial training, atc accuracies are {atc_groups}")
    print(f"For initial training, group scores are {group_scores}")
    print(f"For initial training, group losses are {group_losses}")
    print(f"For initial training, atc prec and recall are {atc_prec_recall}")
    num_groups = group_accuracies.size
    count_tb = np.zeros((num_groups+1, num_groups))
    accuracy_tb = np.zeros((num_groups+1, num_groups))
    score_tb = np.zeros((num_groups+1, num_groups))
    loss_tb = np.zeros((num_groups+1, num_groups))
    atc_tb = np.zeros((num_groups+1, num_groups))
    atc_prec_recall_tb = np.zeros((num_groups+1, 2*num_groups))

    count_tb[0,:] = group_counts
    accuracy_tb[0, :] = group_accuracies
    score_tb[0, :] = group_scores
    loss_tb[0, :] = group_losses
    atc_tb[0, :] = atc_groups
    atc_prec_recall_tb[0, :] = atc_prec_recall

    # Start the query loop 
    for group_idx in range(num_groups):
        curr_query += 1
        round_step = 0

        resume = False
        checkpoint_path = None
        if args.wandb_id:
            checkpoint_dir = f'checkpoint/{args.save_name}'
            if os.path.isdir(checkpoint_dir):
                list = os.listdir(checkpoint_dir)
                if len(list) > 0: 
                    queries = [int(file.split('.')[0]) for file in list]
                    group_ckp_idx = group_idx+1
                    if group_ckp_idx in queries:
                        resume = True
                        checkpoint_path = f'./{checkpoint_dir}/{group_ckp_idx}.pth'
                
        if resume == True:
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            epoch = checkpoint['epoch'] 
            unlabeled_mask = checkpoint['unlabeled_mask']
            curr_query = checkpoint['curr_query']
            curr_train_idx = checkpoint['train_idx']
            # curr_wg = checkpoint['curr_wg']
            print("Finish Loading model with accuracy {}, achieved from epoch {}, query {}, number of labeled samples {}".format(
                best_acc, epoch, curr_query, np.sum(unlabeled_mask == 0)))
            #evaluate and log
            group_counts, group_counts_full, group_counts_label = log_selection(curr_train_idx, train_val_data, 
                                                                                grouper, full_grouper, label_grouper,                                                                        
                                                                                curr_query, 'selection_accumulating', args.log_op)
        else:
            # Query the oracle for more labels
            idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, val_data, query_grouper, query_size=args.query_size, 
                                group_strategy=args.group_strategy, include=[group_idx], query_strategy=args.query_strategy, 
                                replacement=args.replacement, pool_size=args.pool_size, batch_size=args.batch_size, num_workers = args.num_workers)
            _,_,_ = log_selection(idx, train_val_data, 
                                grouper, full_grouper, label_grouper,                                                                        
                                curr_query, 'selection_per_query', args.log_op)
            curr_train_idx = np.append(train_idx, idx)

            # If passed args.new_model, train a new model in each query round
            if args.new_model: 
                net = make_model(args.model, num_classes=num_classes, pretrained=args.pretrain)
                if ((args.dataset == 'waterbirds') or (args.dataset =='celebA')) and (not args.pretrain):
                    if args.model == 'resnet50':
                        net = torchvision.models.resnet50(num_classes = num_classes)
                    elif args.model == 'resnet18':
                        net = torchvision.models.resnet18(num_classes = num_classes)
                net = net.to(device)
                # if device == 'cuda':
                #     net = torch.nn.DataParallel(net)
                #     cudnn.benchmark = True

            # Prepare train loader
            group_counts, group_counts_full, group_counts_label = log_selection(curr_train_idx, train_val_data, 
                                                                                grouper, full_grouper, label_grouper,                                                                        
                                                                                curr_query, 'selection_accumulating', args.log_op)
            data_size = np.sum(unlabeled_mask == 0)
            curr_train_data = WILDSSubset(train_data, curr_train_idx, transform=None)
            # train_loader = get_train_loader("standard", curr_train_data, 
            #                                 batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
            train_loader = DataLoader(
                        curr_train_data,
                        shuffle=False, # Shuffle training dataset
                        collate_fn=curr_train_data.collate,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers, 
                        drop_last=False)

            if args.online: #only train on newly labeled datapoints
                curr_train_data = WILDSSubset(train_data, idx, transform=None)
                train_loader = get_train_loader("standard", curr_train_data, 
                                            batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False) # to use the full data, drop_last is set to false and train() has been modified to handle partially full batch
            
            # Reinitialize optimizer and scheduler
            optimizer, scheduler = init_optimizer_scheduler(net, args.lr, args.schedule, args.num_epoch, train_loader, args.weight_decay)

            # Train the model on the data that has been labeled so far:
            for i in range(args.num_epoch):
                query_end = (i==args.num_epoch-1)
                _, curr_train_acc = train(epoch, round_step, net, train_loader, args.batch_size, data_size, 
                                        optimizer, scheduler, criterion, device, query_end, curr_query, args.no_al)
                # curr_test_acc, wg, wg_full, wg_label, probabilities, group_accuracies, group_scores, group_losses = test(epoch, net, small_val_data, small_val_loader, grouper, full_grouper, label_grouper, 
                #                                                     args.log_op, criterion_val, device, False, curr_query)
                epoch += 1
                round_step += len(train_loader)
        
        curr_test_acc, wg, wg_full, wg_label, probabilities, group_accuracies, group_scores, group_losses = test(epoch, net, val_data, val_loader, grouper, full_grouper, label_grouper, 
                                                                   args.log_op, criterion_val, device, True, curr_query)
        
        # find worst group according to args.group_strategy and args.group_div
        curr_wg = find_target_group(args.group_strategy, args.group_div, probabilities, val_data, 
                                    wg, wg_full, wg_label, 
                                    group_counts, group_counts_full, group_counts_label, 
                                    query_grouper
                                    )

        if not resume:
            save_checkpoint(args.save, curr_query % args.save_every == 0, net, curr_test_acc, epoch,
                            curr_query, unlabeled_mask, curr_train_idx, curr_wg, args.save_name, wandb.run.name)
        
        group_dist = group_counts/np.sum(group_counts)
        val_sample = sample_from_distribution(group_dist, raw_val_data, np.ones(len(raw_val_data)), 1100, query_grouper)
        _, _, _ = log_selection(val_sample, raw_val_data, 
                            grouper, full_grouper, label_grouper,                                                                        
                            curr_query, None, args.log_op) 
        atc_acc, atc_groups, _, atc_prec_recall = atc(net, device, val_data, WILDSSubset(raw_val_data, val_sample, transform=None), 
                                grouper, args.batch_size, args.num_workers, test_probs=probabilities)       
        print(f"ATC accuracy predicted from source val data is {atc_acc}")
        
        print(f"For training with group {group_idx} added, group counts are {group_counts}")
        print(f"For training with group {group_idx} added, group accuracies are {group_accuracies}")
        print(f"For training with group {group_idx} added, atc accuracies are {atc_groups}")
        print(f"For training with group {group_idx} added, group scores are {group_scores}")
        print(f"For training with group {group_idx} added, group losses are {group_losses}")
        print(f"For training with group {group_idx} added, atc prec and recall are {atc_prec_recall}")
        count_tb[group_idx+1,:] = group_counts
        accuracy_tb[group_idx+1, :] = group_accuracies
        score_tb[group_idx+1, :] = group_scores
        loss_tb[group_idx+1, :] = group_losses
        atc_tb[group_idx+1, :] = atc_groups
        atc_prec_recall_tb[group_idx+1, :] = atc_prec_recall

    print("Group counts table")
    print(count_tb)  
    print("Group accuracies table")
    print(accuracy_tb)
    print("ATC accuracies table")
    print(atc_tb)
    print("Group scores table")
    print(score_tb)
    print("Group losses table")
    print(loss_tb)
    print("ATC precision and recall table")
    print(atc_prec_recall_tb)

    if not os.path.isdir(f'./transfer_result/{args.wandb_group}/{name}'):
        os.mkdir(f'./transfer_result/{args.wandb_group}/{name}')
    pd.DataFrame(accuracy_tb).to_csv(f"./transfer_result/{args.wandb_group}/{name}/accuracy.csv", header=False, index=False)
    pd.DataFrame(atc_tb).to_csv(f"./transfer_result/{args.wandb_group}/{name}/atc.csv", header=False, index=False)
    pd.DataFrame(score_tb).to_csv(f"./transfer_result/{args.wandb_group}/{name}/score.csv", header=False, index=False)
    pd.DataFrame(loss_tb).to_csv(f"./transfer_result/{args.wandb_group}/{name}/loss.csv", header=False, index=False)
    pd.DataFrame(count_tb).to_csv(f"./transfer_result/{args.wandb_group}/{name}/count.csv", header=False, index=False)
    pd.DataFrame(atc_prec_recall_tb).to_csv(f"./transfer_result/{args.wandb_group}/{name}/atc_prec_recall.csv", header=False, index=False)

def find_target_group(group_strategy, group_div, probabilities, val_data,
                      wg, wg_full, wg_label, 
                      group_ct, group_ct_full, group_ct_label,
                      query_grouper
                      ):
    curr_wg = None
    if group_strategy == 'oracle':
        curr_wg = wg
        if group_div == 'full': curr_wg = wg_full
        if group_div == 'label': curr_wg = wg_label 
    elif group_strategy == 'min':
        curr_wg = np.argmin(group_ct)
        if group_div == 'full': curr_wg = np.argmin(group_ct_full)
        if group_div == 'label': curr_wg = np.argmin(group_ct_label)
        #print("Group strategy is min, the smallest group is {}".format(grouper.group_str(curr_wg)))
    elif group_strategy == 'avg_c_val':
        curr_wg = get_avgc_worst_group(probabilities, query_grouper, val_data)
    
    return curr_wg

def get_avgc_worst_group(probabilities, grouper, dataset):
    #start_time = time.time()
    confidences = torch.max(probabilities, dim=1)[0]
    confidences = np.asarray(confidences.cpu())
    print(f'Val set is of size {len(dataset)}')
    group, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
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

def init_optimizer_scheduler(net, lr, schedule, num_epoch, train_loader, wd):
    optimizer = optim.SGD(net.parameters(), lr=lr,
                momentum=0.9, weight_decay=wd)
    if schedule == "constant":
        lambda_cons = lambda epoch: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_cons, last_epoch=-1, verbose=False)
    elif schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch*len(train_loader))
    elif schedule == "find_lr":
        lambda_cons = lambda epoch: 1.05**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_cons, last_epoch=-1, verbose=False)
    return optimizer, scheduler

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
        if wandb_name is not None: 
            wandb.log(dict(**query_info, **{"general/curr_query":curr_query}))
    return group_counts


# Train
def train(epoch, step, net, dataloader, batch_size, data_size, optimizer, scheduler, criterion, device, query_end, curr_query, no_al):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_samples = 0
    for batch_idx, (inputs, targets, metadata) in enumerate(dataloader):
        step += 1
        # For specific lr scheduling defined in al_utils.py
        # lr = get_lr(step, data_size, args.lr)
        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        inputs, targets = inputs.to(device), targets.to(device)
        num_samples += inputs.size()[0]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
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
    
    return train_loss/len(dataloader), 100.*correct/total


# Test
def test(epoch, net, dataset, dataloader, grouper, full_grouper, label_grouper, log_op, criterion, device, query_end, curr_query):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_samples = 0
    probabilities = None
    predictions = None
    losses = None
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            num_samples += inputs.size()[0]
            outputs = net(inputs)

            probs = F.softmax(outputs, dim=1)
            if probabilities is None:
                probabilities = probs
            else:
                probabilities = torch.cat((probabilities, probs))

            loss = criterion(outputs, targets)
            if losses is None:
                losses = loss
            else:
                losses = torch.cat((losses, loss))
            mean_loss = torch.mean(loss)
            test_loss += mean_loss * inputs.size()[0]
            _, predicted = outputs.max(1)

            if predictions is None:
                predictions = predicted
            else:
                predictions = torch.cat((predictions, predicted))

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/num_samples, 100.*correct/total, correct, total))
    if not query_end: 
        wandb_label = 'val'
    else:
        wandb_label = 'val_per_query'
    
    # wandb.log({"general/epoch": epoch, "val/epoch_loss":test_loss/num_samples})
    wandb.log({"general/epoch": epoch, "general/curr_query": curr_query, f"{wandb_label}/epoch_loss":test_loss/num_samples})

    #oracle worst group
    worst_group, results, group_accuracies, group_scores, group_losses = get_worst_group(dataset, grouper, probabilities, losses, device, log_op, prefix='standard') 
    worst_group_full, results_full, _, _, _ = get_worst_group(dataset, full_grouper, probabilities, losses, device, log_op, prefix='full') 
    worst_group_label, results_label, _, _, _ = get_worst_group(dataset, label_grouper, probabilities, losses, device, log_op, prefix='label') 
    if results_full is not None: results.update(results_full)
    if results_label is not None:results.update(results_label)

    target = dataset.y_array.to(device)
    auc = calculate_auc(dataset.n_classes, probabilities, target)
    #print(f"auc is {auc}")
    
    if dataset.dataset_name == 'waterbirds':
        y_array = dataset.y_array.cpu()
        meta_array = dataset.metadata_array.cpu()
        results, result_str = dataset.eval(predictions.cpu(), y_array, meta_array)
        #print(result_str)
        log_waterbirds(epoch, curr_query, 100.*correct/total, auc, results, wandb_group=wandb_label)
        acc = results['adj_acc_avg']
    else:
        #print(results)
        log_test(epoch, curr_query, 100.*correct/total, auc, results, wandb_group=wandb_label)
        acc = 100.*correct/total

    # # if the current epoch is the last of the query round
    # if query_end:
    #    if dataset.dataset_name == 'waterbirds':
    #        log_waterbirds(epoch, curr_query, 100.*correct/total, auc, results, wandb_group='val_per_query')
    #    else:
    #        log_test(epoch, curr_query, 100.*correct/total, auc, results, wandb_group='val_per_query')

    return acc, worst_group, worst_group_full, worst_group_label, probabilities, group_accuracies, group_scores, group_losses

def calculate_auc(n_classes, probabilities, target):
    if n_classes == 2:
        preds = probabilities[:,1].squeeze()
        auc = torchmetrics.functional.auroc(preds, target)
    else:
        auc = torchmetrics.functional.auroc(probabilities, target, num_classes=n_classes, average='weighted')
    return auc

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

# output worst performing group and group-wise metrics(for wandb log)
def get_worst_group(dataset, grouper, probabilities, losses, device, log_op, prefix = 'standard'):
    if prefix != 'standard' and (not log_op): 
        return None, None, None, None, None
    confidences, predictions = probabilities.to(device).max(1)
    y_array = dataset.y_array.to(device)
    meta_array = dataset.metadata_array
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    results = {}
    num_group = torch.numel(group_counts)
    acc = np.zeros(num_group)
    #auc = np.zeros(num_group)
    scores = np.zeros(num_group)
    group_losses = np.zeros(num_group)
    for i in range(num_group):
        group_idx = torch.nonzero(group == i).squeeze()
        if group_counts[i] > 0: 
            acc[i] = torch.sum(y_array[group_idx] == predictions[group_idx])/group_counts[i]
            scores[i] = torch.sum(confidences[group_idx])/group_counts[i]
            group_losses[i] = torch.sum(losses[group_idx])/group_counts[i]
        else: 
            acc[i] = 1.01
        #auc[i] = calculate_auc(dataset.n_classes, probabilities, y_array)
        if (prefix == 'standard') or (log_op == 'log'):
            results.update({f'{prefix}_acc_{grouper.group_str(i)}':acc[i]})
            #results.update({f'{prefix}_auc_{grouper.group_str(i)}':auc[i]})
    worst_group = np.argmin(acc)
    wg_acc = acc[worst_group]
    results.update({f"{prefix}_wg_acc": wg_acc, f'{prefix}_wg': worst_group, f'{prefix}_mean_acc': np.mean(acc)})
    # auc_worst_group = np.argmin(auc)
    # wg_auc = auc[auc_worst_group]
    # results.update({f"{prefix}_auc_wg_sc": wg_auc, f'{prefix}_auc_wg': auc_worst_group})
    print("Worst group is {}: {} with acc {}".format(worst_group, grouper.group_str(worst_group), wg_acc))
    #print("AUC Worst group is {}: {} with auc {}".format(auc_worst_group, grouper.group_str(auc_worst_group), wg_auc))
    return worst_group, results, acc, scores, group_losses 

def log_waterbirds(epoch, curr_query, test_acc, auc, results, wandb_group='val'):
    if wandb_group == 'val':
        wandb.log({"general/epoch": epoch, "general/curr_query": curr_query, "val/test_acc": test_acc, "val/auc": auc,
                    "val/adj_acc":results['adj_acc_avg'],
                    "val/landbird_land_acc":results['acc_y:landbird_background:land'], 
                    "val/landbird_water_acc":results['acc_y:landbird_background:water'],
                    "val/waterbird_land_acc":results['acc_y:waterbird_background:land'],
                    "val/waterbird_water_acc":results['acc_y:waterbird_background:water'],
                    "val/wg_acc":results['acc_wg']
                    })
    elif wandb_group == 'val_per_query':
        wandb.log({"general/epoch": epoch, "general/curr_query": curr_query, 
                "val_per_query/query_end_test_acc": test_acc, "val_per_query/query_end_auc": auc,
                "val_per_query/query_end_adj_acc":results['adj_acc_avg'],
                "val_per_query/query_end_landbird_land_acc":results['acc_y:landbird_background:land'], 
                "val_per_query/query_end_landbird_water_acc":results['acc_y:landbird_background:water'],
                "val_per_query/query_end_waterbird_land_acc":results['acc_y:waterbird_background:land'],
                "val_per_query/query_end_waterbird_water_acc":results['acc_y:waterbird_background:water'],
                "val_per_query/query_end_wg_acc":results['acc_wg']
                })

def log_test(epoch, curr_query, test_acc, auc, results, wandb_group='val'):
    log_dict = {"general/epoch": epoch, "general/curr_query": curr_query, 
                f'{wandb_group}/test_acc': test_acc, f'{wandb_group}/auc': auc}
    for key in results:
        group_type = key.split('_')[0]
        log_dict.update({f'{wandb_group}:{group_type}/{key}':results[key]})
    wandb.log(log_dict)


if __name__ == "__main__":
    main()
