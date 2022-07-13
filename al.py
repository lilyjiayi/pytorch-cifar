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
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=10, type=int, help='number of epochs to train for each query round')
    parser.add_argument('--schedule', default="cosine", type=str) # 'constant', 'cosine'
    parser.add_argument('--new_model', '-n', action='store_true',
                        help='train a new model after each query round')
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
    parser.add_argument('--query_strategy', default='least_confidence', type=Str) # 'least_confidence', 'margin', 'random'
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
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=1e-4)
    if args.schedule == "constant":
        lambda_cons = lambda epoch: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_cons, last_epoch=-1, verbose=False)
    elif args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch*len(train_loader))

    # Label the initial subset
    group_counts = query_the_oracle(unlabeled_mask, net, device, train_val_data, grouper, query_size=args.seed_size, 
                    query_strategy='random', pool_size=0, batch_size=args.batch_size)
    query_info = dict()
    group_counts = np.array(group_counts)
    for i in range(len(group_counts)):
        query_info[grouper.group_str(i)] = 100.0 * group_counts[i]/np.sum(group_counts)
        print("group: {}, count: {} \n".format(grouper.group_str(i), group_counts[i]))
    wandb.log(query_info)

    # Pre-train on the initial subset
    epoch = start_epoch
    #query_start_epoch = np.zeros(args.num_queries + 1) # store the start epoch index for each query; the first query is the initial seed set with start epoch 0
    curr_query = 0
    step = 0 

    labeled_idx = np.where(unlabeled_mask == 0)[0]
    data_size = np.sum(unlabeled_mask == 0)
    train_loader = get_train_loader("standard", WILDSSubset(train_data, labeled_idx, transform=None), 
                                    batch_size=args.batch_size, num_workers=2)
    if args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch*len(train_loader))
    # previous_test_acc = 0
    # current_test_acc = 1
    # previous_train_acc = 0
    # current_train_acc = 0
    # while not(current_train_acc > previous_train_acc and previous_test_acc > current_test_acc + 5.0):
        # previous_train_acc = current_train_acc
        # previous_test_acc = current_test_acc
    for i in range(args.num_epoch):
        _, current_train_acc = train(epoch, step, net, train_loader, data_size, optimizer, scheduler, criterion, device)
        current_test_acc = test(epoch, net, val_data, val_loader, criterion, device, best_acc,
                            unlabeled_mask, curr_query, args.save_name, wandb.run.name, save=args.save)
        epoch += 1
        step += len(train_loader)
        

    # Start the query loop 
    for query in range(args.num_queries):
        #print(query_start_epoch)
        #query_start_epoch[query + 1] = epoch
        curr_query += 1

        # Query the oracle for more labels
        group_counts = query_the_oracle(unlabeled_mask, net, device, train_val_data, grouper, query_size=args.query_size, 
                        query_strategy=args.query_strategy, pool_size=0, batch_size=args.batch_size)
        group_counts = np.array(group_counts)
        query_info = dict()
        for i in range(len(group_counts)):
            query_info[grouper.group_str(i)] = 100.0 * group_counts[i]/np.sum(group_counts)
            print("group: {}, count: {} \n".format(grouper.group_str(i), group_counts[i]))
        wandb.log(query_info)

        if args.new_model: 
            net = torchvision.models.resnet50(num_classes = num_classes)
            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

        # Train the model on the data that has been labeled so far:
        labeled_idx = np.where(unlabeled_mask == 0)[0]
        train_loader = get_train_loader("standard", WILDSSubset(train_data, labeled_idx, transform=None), 
                                        batch_size=args.batch_size, num_workers=2)
        data_size = np.sum(unlabeled_mask == 0)
        # previous_test_acc = 0
        # current_test_acc = 1
        # previous_train_acc = 0
        # current_train_acc = 0
        # while not(current_train_acc > previous_train_acc and previous_test_acc > current_test_acc + 5.0):
        #     previous_train_acc = current_train_acc
        #     previous_test_acc = current_test_acc
        for i in range(args.num_epoch):
            _, current_train_acc = train(epoch, step, net, train_loader, data_size, optimizer, scheduler, criterion, device)
            current_test_acc = test(epoch, net, val_data, val_loader, criterion, device, best_acc,
                            unlabeled_mask, curr_query, args.save_name, wandb.run.name, save=args.save)
            epoch += 1
            step += len(train_loader)



# Training
def train(epoch, step, net, dataloader, data_size, optimizer, scheduler, criterion, device):
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
        wandb.log({"step_loss":loss.item(), "lr": lr})
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Learning rate: %.6f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (lr, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({"epoch": epoch, "data_size": data_size, "batch_loss": train_loss/len(dataloader), "train_acc":100.*correct/total})
    
    return train_loss/len(dataloader), 100.*correct/total

def test(epoch, net, dataset, dataloader, criterion, device, best_acc, 
        unlabeled_mask, curr_query, save_name, run_name, save=True):
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

    wandb.log({"epoch": epoch, "curr_query": curr_query, "test_acc": 100.*correct/total,
    "adj_acc":results['adj_acc_avg'],
    "landbird_land_acc":results['acc_y:landbird_background:land'], 
    "landbird_water_acc":results['acc_y:landbird_background:water'],
    "waterbird_land_acc":results['acc_y:waterbird_background:land'],
    "waterbird_water_acc":results['acc_y:waterbird_background:water'],
    "wg_acc":results['acc_wg']
    })

    # Save checkpoint.
    acc = 100.*correct/total
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

    return 100.*correct/total



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
