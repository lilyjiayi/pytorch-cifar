'''Active learning with PyTorch.'''
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
import time

import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from al_utils import query_the_oracle, get_lr

import os
import argparse

from models import *
from utils import progress_bar

import wandb

def main():
    parser = argparse.ArgumentParser(description='Active Learning Training')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for dataloader')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--ilr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--inum_epoch', default=3, type=int, help='number of epochs to train for initial training')
    parser.add_argument('--num_epoch', default=3, type=int, help='number of epochs to train for each query round')
    parser.add_argument('--ischedule', default="cosine", type=str) # 'constant', 'cosine'
    parser.add_argument('--schedule', default="cosine", type=str) # 'constant', 'cosine'
    parser.add_argument('--drop_last', '-d', action='store_true',
                        help='drop last batch')
    parser.add_argument('--new_model', '-n', action='store_true',
                        help='train a new model after each query round')
    parser.add_argument('--online', '-o', action='store_true',
                        help='only train newly selected datapoints for each query round')
    parser.add_argument('--no_al',  action='store_true',
                        help='train a resnet baseline')
    parser.add_argument('--dataset', default='celebA', type=str) #'waterbirds', 'celebA'
    parser.add_argument('--target', default='Male', type=str)
    parser.add_argument('--root_dir', default="/self/scr-sync/nlp/waterbirds", type=str) #root dir for accessing dataset
    parser.add_argument('--wandb_group', default=None, type=str)
    #checkpoints are saved at /nlp/scr/jiayili/pytorch-cifar/checkpoints
    parser.add_argument('--model', default="resnet18", type=str) #'resnet18', 'resnet50'
    parser.add_argument('--pretrain', action='store_true',
                        help='load weights from resnet50 pretrained on IMAGENET1K_V2')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--checkpoint', default="./checkpoint/al_waterbirds.pth", type=str) #checkpoint to load
    parser.add_argument('--save', '-s', action='store_true',
                        help='save checkpoint')
    parser.add_argument('--save_name', default="al_waterbirds", type=str) #checkpoint name to save
    parser.add_argument('--save_every', default=20, type=int, help='save checkpoint every # number of queries')
    # Uncertainty sampling parameters
    parser.add_argument('--seed_size', default=300, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--query_size', default=50, type=int)
    parser.add_argument('--query_strategy', default='least_confidence', type=str) # 'least_confidence', 'margin', 'random'
    parser.add_argument('--group_strategy', default=None, type=str) # 'oracle', 'avg_c', 'avg_c_val', 'min'
    parser.add_argument('--replacement', action='store_true', help='query with replacement')
    args = parser.parse_args()

    #set default drop_last flag to True
    args.drop_last = True

    #wandb setup
    mode = 'disabled' if not args.wandb_group else 'online'
    wandb.init(project='al_group', entity='hashimoto-group', mode=mode, group=args.wandb_group)
    wandb.config.update(vars(args))
    if args.wandb_group:
        args.save_name = args.wandb_group

    # Data
    print('==> Preparing data..')
    # Load the full dataset, and download it if necessary
    args.root_dir = f'/self/scr-sync/nlp/{args.dataset}'
    if args.dataset == 'celebA':
        dataset = get_dataset(dataset=args.dataset, download=True, root_dir = args.root_dir, target='Male', group=['Black_Hair','Wavy_Hair'])
    else:
        dataset = get_dataset(dataset=args.dataset, download=True, root_dir = args.root_dir)
    #print(dataset.metadata_fields)
    #print(dataset.metadata_map)
    if args.dataset == 'celebA':
        grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-2])
    else:
        grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-1])

    # Get the training and validation set (transform config from https://github.com/kohpangwei/group_DRO/blob/f7eae929bf4f9b3c381fae6b1b53ab4c6c911a0e/data/cub_dataset.py#L78-L102)
    scale = 256.0/224.0
    target_resolution = (224, 224)

    train_transform = transforms.Compose([
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
        transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
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
    val_data = dataset.get_subset(
        "val",
        transform=eval_transform,
    )
    # test data not used 
    test_data = dataset.get_subset(
        "test",
        transform=eval_transform,
    )

    print('Train set size: ', len(train_data))
    print('Eval set size: ', len(val_data))
    print('Test set size: ', len(test_data))

    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size, num_workers=args.num_workers)

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

    num_classes = 2  # number of classes in the classification problem
    if args.model == 'resnet50':
        net = torchvision.models.resnet50(num_classes = num_classes)
    elif args.model == 'resnet18':
        net = torchvision.models.resnet18(num_classes = num_classes)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        epoch = checkpoint['epoch'] + 1
        unlabeled_mask = checkpoint['unlabeled_mask']
        curr_query = checkpoint['curr_query']
        train_idx = checkpoint['train_idx']
        curr_wg = checkpoint['curr_wg']
        print("Finish Loading model with accuracy {}, achieved from epoch {}, query {}, number of labeled samples {}".format(
            best_acc, epoch, curr_query, np.sum(unlabeled_mask == 0)))
    else:
        # Initialize counters
        round_step = 0 #keeps track of number of steps for this query round, not really used
        #query_start_epoch = np.zeros(args.num_queries + 1) # store the start epoch index for each query; the first query is the initial seed set with start epoch 0
        #output_result = None

        # Label the initial subset
        if args.no_al:
            unlabeled_mask = np.zeros(len(train_data))
            args.num_queries = 0
            train_idx = np.arange(len(train_data))
        else:
            # randomly sample the seed set
            idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, grouper, query_size=args.seed_size,
                            group_strategy=None, wg=curr_wg, query_strategy='random', 
                            replacement=args.replacement, pool_size=0, batch_size=args.batch_size, num_workers = args.num_workers)
            print_log_selection_info(idx, train_val_data, grouper, curr_query, "selection_per_query")
            train_idx = idx

        # Prepare train loader
        group_counts = print_log_selection_info(train_idx, train_val_data, grouper, curr_query, "selection_accumulating")
        data_size = np.sum(unlabeled_mask == 0) #keeps track of number of distinct labeled datapoints
        train_loader = get_train_loader("standard", WILDSSubset(train_data, train_idx, transform=None), 
                                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
        
        # Initialize optimizer and scheduler
        optimizer, scheduler = init_optimizer_scheduler(net, args.ilr, args.ischedule, args.inum_epoch, train_loader, args.weight_decay)

        # if args.no_al and args.schedule == "cosine":
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.inum_epoch)

        # Pre-train on the initial subset
        for i in range(args.inum_epoch):
            query_end = (i==args.inum_epoch-1)
            curr_train_loss, curr_train_acc = train(epoch, round_step, net, train_loader, args.batch_size, data_size, 
                                                    optimizer, scheduler, criterion, device, query_end, curr_query, args.no_al)
            # curr_test_acc, curr_wg = test(epoch, net, val_data, val_loader, grouper, criterion, 
            #                               device, best_acc, args.group_strategy,
            #                               unlabeled_mask, train_idx, query_end, curr_query,  
            #                               args.save_name, wandb.run.name, save=args.save)
            # if args.no_al:
            #     scheduler.step()
            epoch += 1
            round_step += len(train_loader)
        
        # To speed up training, only evaluate at the end of query 
        curr_test_acc, curr_wg, confidences = test(epoch, net, val_data, val_loader, grouper, criterion, 
                                                   device, best_acc, args.group_strategy,
                                                   unlabeled_mask, train_idx, True, curr_query)
        # by default, curr_wg is the worst performing group on the val set
        if args.group_strategy == 'min':
            curr_wg = np.argmin(group_counts)
            #print("Group strategy is min, the smallest group is {}".format(grouper.group_str(curr_wg)))
        elif args.group_strategy == 'avg_c_val':
            curr_wg = get_avgc_worst_group(confidences, grouper, val_data)
        
        save_checkpoint(args.save, curr_query % args.save_every == 0, net, curr_test_acc, epoch,
                        curr_query, unlabeled_mask, train_idx, curr_wg, args.save_name, wandb.run.name)

    # Start the query loop 
    for query in range(args.num_queries - curr_query):
        #print(query_start_epoch)
        #query_start_epoch[query + 1] = epoch
        curr_query += 1
        round_step = 0

        # Query the oracle for more labels
        idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, grouper, query_size=args.query_size, 
                               group_strategy=args.group_strategy, wg=curr_wg, query_strategy=args.query_strategy, 
                               replacement=args.replacement, pool_size=0, batch_size=args.batch_size, num_workers = args.num_workers)
        print_log_selection_info(idx, train_val_data, grouper, curr_query, "selection_per_query")
        train_idx = np.append(train_idx, idx)

        # If passed args.new_model, train a new model in each query round
        if args.new_model: 
            if args.model == 'resnet50':
                net = torchvision.models.resnet50(num_classes = num_classes)
            elif args.model == 'resnet18':
                net = torchvision.models.resnet18(num_classes = num_classes)
            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

        # Prepare train loader
        group_counts = print_log_selection_info(train_idx, train_val_data, grouper, curr_query, "selection_accumulating")
        data_size = np.sum(unlabeled_mask == 0)
        train_loader = get_train_loader("standard", WILDSSubset(train_data, train_idx, transform=None), 
                                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
        if args.online: #only train on newly labeled datapoints
            train_loader = get_train_loader("standard", WILDSSubset(train_data, idx, transform=None), 
                                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False) # to use the full data, drop_last is set to false and train() has been modified to handle partially full batch
        
        # Reinitialize optimizer and scheduler
        optimizer, scheduler = init_optimizer_scheduler(net, args.lr, args.schedule, args.num_epoch, train_loader, args.weight_decay)

        # Train the model on the data that has been labeled so far:
        for i in range(args.num_epoch):
            query_end = (i==args.num_epoch-1)
            _, curr_train_acc = train(epoch, round_step, net, train_loader, args.batch_size, data_size, 
                                      optimizer, scheduler, criterion, device, query_end, curr_query, args.no_al)
            # curr_test_acc, curr_wg = test(epoch, net, val_data, val_loader, grouper, criterion, 
            #                               device, best_acc, args.group_strategy,
            #                               unlabeled_mask, train_idx, query_end, curr_query, 
            #                               args.save_name, wandb.run.name, save=args.save)
            epoch += 1
            round_step += len(train_loader)
        
        curr_test_acc, curr_wg, confidences = test(epoch, net, val_data, val_loader, grouper, criterion, 
                                                   device, best_acc, args.group_strategy,
                                                   unlabeled_mask, train_idx, True, curr_query)

        if args.group_strategy == 'min':
            curr_wg = np.argmin(group_counts)
            #print("Group strategy is min, the smallest group is {}".format(grouper.group_str(curr_wg)))
        elif args.group_strategy == 'avg_c_val':
            curr_wg = get_avgc_worst_group(confidences, grouper, val_data)
        save_checkpoint(args.save, curr_query % args.save_every == 0, net, curr_test_acc, epoch,
                        curr_query, unlabeled_mask, train_idx, curr_wg, args.save_name, wandb.run.name)

def init_optimizer_scheduler(net, lr, schedule, num_epoch, train_loader, wd):
    optimizer = optim.SGD(net.parameters(), lr=lr,
                momentum=0.9, weight_decay=wd)
    if schedule == "constant":
        lambda_cons = lambda epoch: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_cons, last_epoch=-1, verbose=False)
    elif schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch*len(train_loader))
    return optimizer, scheduler

def print_log_selection_info(idx, dataset, grouper, curr_query, wandb_name):
    selected = WILDSSubset(dataset, idx, transform=None)
    meta_array = selected.metadata_array
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    group_counts = np.array(group_counts)
    query_info = dict()
    for i in range(len(group_counts)):
        query_info["{}/{}".format(wandb_name, grouper.group_str(i))] = 100.0 * group_counts[i]/np.sum(group_counts)
        print("{}, group: {}, count: {} \n".format(wandb_name, grouper.group_str(i), group_counts[i]))
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
def test(epoch, net, dataset, dataloader, grouper, criterion, device, best_acc, group_strategy,
        unlabeled_mask, train_idx, query_end, curr_query):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = None
    num_samples = 0
    confidences = None
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            num_samples += inputs.size()[0]
            outputs = net(inputs)

            if group_strategy=='avg_c_val':
                probabilities = F.softmax(outputs, dim=1)
                conf = torch.max(probabilities, dim=1)[0]
                if confidences is None:
                    confidences = conf
                else:
                    confidences = torch.cat((confidences, conf))

            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size()[0]
            _, predicted = outputs.max(1)

            if predictions is None:
                predictions = predicted
            else:
                predictions = torch.cat((predictions, predicted))

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/num_samples, 100.*correct/total, correct, total))

    worst_group, results = get_worst_group(dataset, grouper, predictions) #oracle worst group

    if dataset.dataset_name == 'waterbirds':
        y_array = dataset.y_array.cpu()
        meta_array = dataset.metadata_array.cpu()
        results, result_str = dataset.eval(predictions.cpu(), y_array, meta_array)
        #print(result_str)
        log_waterbirds(epoch, curr_query, 100.*correct/total, results, wandb_group='val')
        acc = results['adj_acc_avg']
    else:
        #print(results)
        log_test(epoch, curr_query, 100.*correct/total, results, wandb_group='val')
        acc = 100.*correct/total

    # if the current epoch is the last of the query round
    if query_end:
       if dataset.dataset_name == 'waterbirds':
           log_waterbirds(epoch, curr_query, 100.*correct/total, results, wandb_group='val_per_query')
       else:
           log_test(epoch, curr_query, 100.*correct/total, results, wandb_group='val_per_query')

    return acc, worst_group, confidences

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
        if not os.path.isdir(f'./checkpoint/{save_name}_{run_name}'):
            os.mkdir(f'./checkpoint/{save_name}_{run_name}')
        torch.save(state, f'./checkpoint/{save_name}_{run_name}/{curr_query}.pth')

def get_avgc_worst_group(confidences, grouper, dataset):
    start_time = time.time()
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

# output worst performing group and group-wise metrics(for wandb log)
def get_worst_group(dataset, grouper, predictions):
    predictions = np.array(predictions.cpu())
    y_array = np.array(dataset.y_array.cpu())
    meta_array = dataset.metadata_array.cpu()
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    group = np.array(group)
    group_counts = np.array(group_counts)
    results = {}
    acc = np.zeros(len(group_counts))
    for i in range(len(group_counts)):
        group_idx = np.nonzero(group == i)[0]
        acc[i] = np.sum(y_array[group_idx] == predictions[group_idx])/group_counts[i]
        results.update({grouper.group_str(i):acc[i]})
    worst_group = np.argmin(acc)
    wg_acc = acc[worst_group]
    results.update({"wg_acc": wg_acc, 'wg': worst_group, 'mean_acc': np.mean(acc)})
    #print("Worst group is {}: {} with acc {}".format(worst_group, grouper.group_str(worst_group), wg_acc))
    return worst_group, results

def log_waterbirds(epoch, curr_query, test_acc, results, wandb_group='val'):
    if wandb_group == 'val':
        wandb.log({"general/epoch": epoch, "general/curr_query": curr_query, "val/test_acc": test_acc,
                    "val/adj_acc":results['adj_acc_avg'],
                    "val/landbird_land_acc":results['acc_y:landbird_background:land'], 
                    "val/landbird_water_acc":results['acc_y:landbird_background:water'],
                    "val/waterbird_land_acc":results['acc_y:waterbird_background:land'],
                    "val/waterbird_water_acc":results['acc_y:waterbird_background:water'],
                    "val/wg_acc":results['acc_wg']
                    })
    elif wandb_group == 'val_per_query':
        wandb.log({"general/epoch": epoch, "general/curr_query": curr_query, "val_per_query/query_end_test_acc": test_acc,
                "val_per_query/query_end_adj_acc":results['adj_acc_avg'],
                "val_per_query/query_end_landbird_land_acc":results['acc_y:landbird_background:land'], 
                "val_per_query/query_end_landbird_water_acc":results['acc_y:landbird_background:water'],
                "val_per_query/query_end_waterbird_land_acc":results['acc_y:waterbird_background:land'],
                "val_per_query/query_end_waterbird_water_acc":results['acc_y:waterbird_background:water'],
                "val_per_query/query_end_wg_acc":results['acc_wg']
                })

def log_test(epoch, curr_query, test_acc, results, wandb_group='val'):
    log_dict = {"general/epoch": epoch, "general/curr_query": curr_query, f'{wandb_group}/test_acc': test_acc}
    for key in results:
        log_dict.update({f'{wandb_group}/{key}':results[key]})
    wandb.log(log_dict)


if __name__ == "__main__":
    main()
