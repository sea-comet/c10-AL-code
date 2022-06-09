'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
#from influence import *

# Utils
#import visdom
#from tqdm import tqdm

# Custom
import models.resnet as resnet
from config import *
from models.discriminator import *
from models.lossnet import *
from data.sampler import SubsetSequentialSampler
import copy




##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cifar10_train = CIFAR10('/mnt/home/xiaoxiang/feat_drop/cifar10/data', train=True, download=False, transform=train_transform)
cifar10_unlabeled = CIFAR10('/mnt/home/xiaoxiang/feat_drop/cifar10/data', train=True, download=False, transform=test_transform)
cifar10_test = CIFAR10('/mnt/home/xiaoxiang/feat_drop/cifar10/data', train=False, download=False, transform=test_transform)


##
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


##
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

##
# Train Utils
iters = 0
criterion_module = nn.BCELoss()
criterion_unsupervised = nn.CrossEntropyLoss()

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['ema'].train()
    models['student'].train()
    global iters

    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['student'].zero_grad()

        scores, cons_scores, features, features_list = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        scores_stu, _, _, _ = models['student'](inputs)
        target_loss_stu = criterion(scores_stu, labels)

        u_inputs, _ = next(iter(dataloaders['extra']))
        u_inputs = u_inputs.cuda()
        u_scores, cons_u_scores, _, _ = models['backbone'](u_inputs)

        '''ema_scores, _, _, _ = models['ema'](inputs)
        ema_u_scores, _, _, _ = models['ema'](u_inputs)
        res_loss = F.mse_loss(scores, cons_scores) + F.mse_loss(u_scores, cons_u_scores)
        consistency_loss = F.mse_loss(cons_scores, ema_scores) + F.mse_loss(cons_u_scores, ema_u_scores)'''

        u_scores_aug = models['ema'](u_inputs)[0]
        kl_loss = F.kl_div(F.log_softmax(u_scores, dim=1), F.softmax(u_scores_aug, dim=1), reduction='batchmean')

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss = m_backbone_loss + 0.02 * kl_loss          #0.02 * (res_loss + consistency_loss)
        loss.backward(retain_graph=True)
        optimizers['backbone'].step()
        update_ema_variables(models['backbone'], models['ema'], 0.999, iters)

        backbone_loss_stu = torch.sum(target_loss_stu) / target_loss_stu.size(0)
        loss_stu = backbone_loss_stu
        loss_stu.backward()
        optimizers['student'].step()


#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, cycle):
    print('>> Train a Model...')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)
        schedulers['backbone'].step()
        schedulers['student'].step()

        # Save a checkpoint
        if epoch % 20 == 0 or epoch == 199:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                    #'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Cycle:', cycle, 'Epoch:', epoch, '---', 'Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc), flush=True)
    print('>> Finished.')

#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['student'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _, _ = models['backbone'](inputs)
            scores_stu, _, _, _ = models['student'](inputs)
            pred_loss = F.kl_div(F.log_softmax(scores_stu, dim=1), F.softmax(scores, dim=1), reduction='batchmean')
            pred_loss = pred_loss.unsqueeze(0)

            uncertainty = torch.cat((uncertainty, pred_loss), dim=0)

    return uncertainty.cpu()


##
# Main
if __name__ == '__main__':

    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        START = 2 * ADDENDUM
        labeled_set = indices[:START]
        unlabeled_set = indices[START:]

        train_loader = DataLoader(cifar10_train, batch_size=BATCH,     # BATCH
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(cifar10_test, batch_size=BATCH)
        extra_loader = DataLoader(cifar10_train, batch_size=BATCH,
                                  sampler=SubsetSequentialSampler(unlabeled_set),
                                  pin_memory=True)

        dataloaders = {'train': train_loader, 'test': test_loader, 'extra': extra_loader}

        # Model
        backbone_net = resnet.ResNet18().cuda()
        student = resnet.ResNet18_student().cuda()

        # create ema model (exponential moving average)
        ema_model = resnet.ResNet18().cuda()
        for param in ema_model.parameters():
            param.detach_()

        models = {'backbone': backbone_net, 'student': student, 'ema': ema_model}
        torch.backends.cudnn.benchmark = True

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            optim_student = optim.SGD(models['student'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_student = lr_scheduler.MultiStepLR(optim_student, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone, 'student': optim_student}
            schedulers = {'backbone': sched_backbone, 'student': sched_student}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, cycle)
            acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc), flush=True)

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=1,  # 1 for our kt-v2
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())       # select largest loss
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            #labeled_set += list(torch.tensor(subset)[arg][:ADDENDUM].numpy())  # select smallest influence
            #unlabeled_set = list(torch.tensor(subset)[arg][ADDENDUM:].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH,    # BATCH
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
            dataloaders['extra'] = DataLoader(cifar10_train, batch_size=BATCH,  # BATCH
                                              sampler=SubsetRandomSampler(unlabeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                    #'state_dict_module': models['module'].state_dict()
                },
                './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))

        print('---------------------------Current Trial is done-----------------------------', flush=True)
