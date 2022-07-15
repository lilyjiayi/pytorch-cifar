'''Active learning with PyTorch.'''
from xxlimited import Str
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

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
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=4, type=int, help='number of epochs to train for each query round')
    parser.add_argument('--inum_epoch', default=3, type=int, help='number of epochs to train for initial training')
    parser.add_argument('--schedule', default="cosine", type=str) # 'constant', 'cosine'
    parser.add_argument('--new_model', '-n', action='store_true',
                        help='train a new model after each query round')
    parser.add_argument('--no_al',  action='store_true',
                        help='train a resnet baseline')
    parser.add_argument('--root_dir', default="/self/scr-sync/nlp/waterbirds", type=str) #root dir for accessing dataset
    parser.add_argument('--wandb_group', default=None, type=str)
    #checkpoints are saved at /nlp/scr/jiayili/pytorch-cifar/checkpoints
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--checkpoint', default="./checkpoint/al_waterbirds.pth", type=str) #checkpoint to load
    parser.add_argument('--save', '-s', action='store_true',
                        help='save checkpoint')
    parser.add_argument('--save_name', default="al_waterbirds", type=str) #checkpoint name to save
    # Uncertainty sampling parameters
    parser.add_argument('--seed_size', default=40, type=int)
    parser.add_argument('--num_queries', default=56, type=int)
    parser.add_argument('--query_size', default=5, type=int)
    parser.add_argument('--query_strategy', default='least_confidence', type=str) # 'least_confidence', 'margin', 'random'
    args = parser.parse_args()

    #wandb setup
    mode = 'disabled' if not args.wandb_group else 'online'
    wandb.init(project='al_group', entity='hashimoto-group', mode=mode, group=args.wandb_group)
    wandb.config.update(vars(args))
    if args.wandb_group:
        args.save_nmae = args.wandb_group

    # Data
    print('==> Preparing data..')
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="waterbirds", download=True, root_dir = args.root_dir)
    grouper = CombinatorialGrouper(dataset, dataset.metadata_fields[:-1])
    #print(dataset.metadata_fields)
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
    test_data = dataset.get_subset(
        "test",
        transform=eval_transform,
    )

    print('Train set size: ', len(train_data))
    print('Eval set size: ', len(val_data))
    print('Test set size: ', len(test_data))

    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size)
    test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)

    # Model
    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    unlabeled_mask = np.ones(len(train_data)) # We assume that in the beginning, the entire train set is unlabeled
    
    num_classes = 2  # number of classes in the classification problem
    net = torchvision.models.resnet50(num_classes = num_classes)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        unlabeled_mask = checkpoint['unlabeled_mask']
        curr_query = checkpoint['curr_query']
        print("Finish Loading model with accuracy {}, achieved from epoch {}, query {}, number of samples {}".format(
            best_acc, start_epoch, curr_query, np.sum(unlabeled_mask == 0)))

    criterion = nn.CrossEntropyLoss()

    # Initialize counters
    curr_query = 0
    epoch = start_epoch # keeps track of overall epoch 
    round_step = 0 #keeps track of number of steps for this query round
    #query_start_epoch = np.zeros(args.num_queries + 1) # store the start epoch index for each query; the first query is the initial seed set with start epoch 0

     # Label the initial subset
    if args.no_al:
        unlabeled_mask = np.zeros(len(train_data))
        args.num_queries = 0
    else:
        idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, grouper, query_size=args.seed_size, 
                        query_strategy='random', pool_size=0, batch_size=args.batch_size)
        print_log_selection_info(idx, train_val_data, grouper, curr_query, "selection_per_query")

    # Prepare train loader
    labeled_idx = np.where(unlabeled_mask == 0)[0]
    print_log_selection_info(labeled_idx, train_val_data, grouper, curr_query, "selection_accumulating")
    data_size = np.sum(unlabeled_mask == 0)
    train_loader = get_train_loader("standard", WILDSSubset(train_data, labeled_idx, transform=None), 
                                    batch_size=args.batch_size, num_workers=2)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = init_optimizer_scheduler(net, args.lr, args.schedule, args.inum_epoch, train_loader)

    if args.no_al and args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.inum_epoch)

    # Pre-train on the initial subset
    for i in range(args.inum_epoch):
        curr_train_loss, curr_train_acc = train(epoch, round_step, net, train_loader, data_size, optimizer, scheduler, criterion, device, i==args.inum_epoch-1, curr_query, args.no_al)
        curr_test_acc, curr_wg = test(epoch, net, val_data, val_loader, grouper, criterion, device, best_acc,
                            unlabeled_mask, i==args.inum_epoch-1, curr_query, args.save_name, wandb.run.name, save=args.save)
        if args.no_al:
            scheduler.step()
        epoch += 1
        round_step += len(train_loader)

    # Start the query loop 
    for query in range(args.num_queries):
        #print(query_start_epoch)
        #query_start_epoch[query + 1] = epoch
        curr_query += 1
        round_step = 0

        # Query the oracle for more labels
        idx = query_the_oracle(unlabeled_mask, net, device, train_val_data, grouper, query_size=args.query_size, 
                        query_strategy=args.query_strategy, pool_size=0, batch_size=args.batch_size)
        print_log_selection_info(idx, train_val_data, grouper, curr_query, "selection_per_query")

        # If passed args.new_model, train a new model in each query round
        if args.new_model: 
            net = torchvision.models.resnet50(num_classes = num_classes)
            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

        # Prepare train loader
        labeled_idx = np.where(unlabeled_mask == 0)[0]
        print_log_selection_info(labeled_idx, train_val_data, grouper, curr_query, "selection_accumulating")
        data_size = np.sum(unlabeled_mask == 0)
        train_loader = get_train_loader("standard", WILDSSubset(train_data, labeled_idx, transform=None), 
                                        batch_size=args.batch_size, num_workers=2)

        # Reinitialize optimizer and scheduler
        optimizer, scheduler = init_optimizer_scheduler(net, args.lr, args.schedule, args.num_epoch, train_loader)

        # Train the model on the data that has been labeled so far:
        for i in range(args.num_epoch):
            _, curr_train_acc = train(epoch, round_step, net, train_loader, data_size, optimizer, scheduler, criterion, device, i==args.num_epoch-1, curr_query, args.no_al)
            curr_test_acc, curr_wg = test(epoch, net, val_data, val_loader, grouper, criterion, device, best_acc,
                            unlabeled_mask, i==args.num_epoch-1, curr_query, args.save_name, wandb.run.name, save=args.save)
            epoch += 1
            round_step += len(train_loader)


def init_optimizer_scheduler(net, lr, schedule, num_epoch, train_loader):
    optimizer = optim.SGD(net.parameters(), lr=lr,
                momentum=0.9, weight_decay=1e-4)
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

# Training
def train(epoch, step, net, dataloader, data_size, optimizer, scheduler, criterion, device, query_end, curr_query, no_al):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, metadata) in enumerate(dataloader):
        step += 1
        # For specific lr scheduling defined in al_utils.py
        # lr = get_lr(step, data_size, args.lr)
        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        lr = scheduler.get_last_lr()[0]
        wandb.log({"train/train_step_loss":loss.item(), "train/lr": lr})
        optimizer.step()
        if not no_al:
            scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Learning rate: %.6f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (lr, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({"general/epoch": epoch, "train/train_epoch_loss": train_loss/len(dataloader), "train/train_acc":100.*correct/total})
    
    # if the current epoch is the last of the query round 
    if query_end:
        wandb.log({"general/epoch": epoch, "general/data_size": data_size, "general/curr_query": curr_query,
                "train_per_query/query_end_train_loss": train_loss/len(dataloader), "train_per_query/query_end_train_acc":100.*correct/total})
    
    return train_loss/len(dataloader), 100.*correct/total

def test(epoch, net, dataset, dataloader, grouper, criterion, device, best_acc, 
        unlabeled_mask, query_end, curr_query, save_name, run_name, save=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = None
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            if predictions is None:
                predictions = predicted
            else:
                predictions = torch.cat((predictions, predicted))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    y_array = dataset.y_array.cpu()
    meta_array = dataset.metadata_array.cpu()
    results, result_str = dataset.eval(predictions.cpu(), y_array, meta_array)
    print(result_str)

    worst_group = get_worst_group(dataset, grouper, predictions)

    wandb.log({"general/epoch": epoch, "general/curr_query": curr_query, "val/test_acc": 100.*correct/total,
                "val/adj_acc":results['adj_acc_avg'],
                "val/landbird_land_acc":results['acc_y:landbird_background:land'], 
                "val/landbird_water_acc":results['acc_y:landbird_background:water'],
                "val/waterbird_land_acc":results['acc_y:waterbird_background:land'],
                "val/waterbird_water_acc":results['acc_y:waterbird_background:water'],
                "val/wg_acc":results['acc_wg']
                })

    # if the current epoch is the last of the query round
    if query_end:
       wandb.log({"general/epoch": epoch, "general/curr_query": curr_query, "val_per_query/query_end_test_acc": 100.*correct/total,
                "val_per_query/query_end_adj_acc":results['adj_acc_avg'],
                "val_per_query/query_end_landbird_land_acc":results['acc_y:landbird_background:land'], 
                "val_per_query/query_end_landbird_water_acc":results['acc_y:landbird_background:water'],
                "val_per_query/query_end_waterbird_land_acc":results['acc_y:waterbird_background:land'],
                "val_per_query/query_end_waterbird_water_acc":results['acc_y:waterbird_background:water'],
                "val_per_query/query_end_wg_acc":results['acc_wg']
                })

    # Save checkpoint.
    acc = results['adj_acc_avg']
    if acc > best_acc and save:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            "curr_query": curr_query,
            'unlabeled_mask': unlabeled_mask
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{save_name}_{run_name}.pth')
        best_acc = acc

    return 100.*correct/total, worst_group

def get_worst_group(dataset, grouper, predictions):
    predictions = np.array(predictions.cpu())
    y_array = np.array(dataset.y_array.cpu())
    meta_array = dataset.metadata_array.cpu()
    group, group_counts = grouper.metadata_to_group(meta_array, return_counts=True)
    group = np.array(group)
    group_counts = np.array(group_counts)
    acc = np.zeros(len(group_counts))
    for i in range(len(group_counts)):
        group_idx = np.nonzero(group == i)[0]
        acc[i] = np.sum(y_array[group_idx] == predictions[group_idx])/group_counts[i]
    worst_group = np.argmin(acc)
    wg_acc = acc[worst_group]
    #print("Worst group is {}: {} with acc {}".format(worst_group, grouper.group_str(worst_group), wg_acc))
    return worst_group


 



##deprecated test method
# def test(epoch, query_start_epoch, curr_query):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, metadata) in enumerate(eval_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(eval_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#     wandb.log({"epoch": epoch, "curr_query": curr_query, "test_acc": 100.*correct/total})        

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#             'query_start_epoch': query_start_epoch
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, args.save_name)
#         best_acc = acc

#     return 100.*correct/total


if __name__ == "__main__":
    main()
