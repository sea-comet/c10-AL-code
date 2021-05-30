# -*- coding: UTF-8 -*-
# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from numpy.core._multiarray_umath import ndarray
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F


# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
# from influence import *

# Utils
# import visdom
# from tqdm import tqdm

# Custom
import models.resnet as resnet
from config import *
from data.sampler import SubsetSequentialSampler

# import copy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

#
# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Cuda available?: {}, device now: {}".format(torch.cuda.is_available(), device))
print("CUDA_VISIBLE_DEVICE: cuda ", os.environ["CUDA_VISIBLE_DEVICES"])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cifar10_train = CIFAR10('../cifar10', train=True, download=False, transform=train_transform)  # specify data path here
cifar10_unlabeled = CIFAR10('../cifar10', train=True, download=False, transform=test_transform)
cifar10_test = CIFAR10('../cifar10', train=False, download=False, transform=test_transform)

##
# Train Utils
iters = 0


#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    global iters

    for data in dataloaders['train']:
        inputs = data[0].to(device)
        labels = data[1].to(device)
        iters += 1

        optimizers['backbone'].zero_grad()

        scores, _, _, features_list = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss = m_backbone_loss
        loss.backward()
        optimizers['backbone'].step()


#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.to(device)
            labels = labels.to(device)

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

        # Save a checkpoint
        if epoch % 20 == 0 or epoch == 199:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                    # 'state_dict_module': models['module'].state_dict()
                },
                    '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Cycle:', cycle, 'Epoch:', epoch, "---", 'Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc), flush=True)
            # 这里去掉了一个 ,flush=True
    print('>> Finished.')


#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    # uncertainty = torch.tensor([]).to(device) #这是原来版本，聚类先不用GPU，用CPU可以用numpy
    uncertainty = torch.tensor([])

    with torch.no_grad():
        print(">> Start clustering:")
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.to(device)

            scores, _, total_feature, features = models['backbone'](inputs)
            # total_feature: [SUBSET,512], SUBSET是行，是batch_size，也就是图片个数，512是列，是resnet展平的像素feature个数

            # TODO Use clustering to determine data uncertainty
            # input_x = np.array(total_feature) # 这是原来版本，计算关键点 因为KMeans.fix(X[,y) X是需要2D 而不是1D
            input_x = np.array(total_feature.cpu())
            # input_x: shape: [BATCH_SIZE,512]
            # 利用别人写好的简单Kmeans, 可以知道每个点离最近两个cluster中心点的距离差

            """k-means聚类算法
               input_x      - ndarray(line_num, column_num)，line_num个样本的数据集，每个样本column_num个属性值
               """

            line_num, column_num = input_x.shape  # line_num：样本数量，column_num：每个样本的属性值个数
            # print("line_num: ",line_num)
            # print("column_num: ", column_num)

            result = np.empty(line_num, dtype=np.int)  # line_num个样本的聚类结果
            distance = np.empty((line_num, CLUSTER_NUMS), dtype=np.float32)  # 我加的
            # 从line_num个数据样本中不重复地随机选择k个样本作为质心
            cores = input_x[np.random.choice(np.arange(line_num), CLUSTER_NUMS, replace=False)]
            min_distance_differ = np.empty((line_num, 1), dtype=np.float32)  # 对吗？？
            iter = 0

            while True:  # 迭代聚类计算,这是10000次，这里也可以用while True，质心不变时停止
                iter = iter + 1
                d = np.square(
                    np.repeat(input_x, CLUSTER_NUMS, axis=0).reshape(line_num, CLUSTER_NUMS, column_num) - cores)
                distance = np.sqrt(np.sum(d, axis=2))  # ndarray(line_num, k)，每个样本距离k个质心的距离，共有line_num行

                index_min = np.argmin(distance, axis=1)  # 每个样本距离最近的质心索引序号 [line_num, 1]
                if (index_min == result).all() or iter == CLUSTER_MAX_ITER:
                    break;

                # 这里在 while True 时使用
                # if (index_min == result).all():  # 如果样本聚类没有改变
                #    return result, cores  # 则返回聚类结果和质心数据

                result[:] = index_min  # 重新分类 [line_num, 1] 这些图片分别属于哪个cluster, index
                for i in range(CLUSTER_NUMS):  # 遍历质心集
                    if len(input_x[result == i]) != 0:
                        items = input_x[result == i]  # 找出对应当前质心的子样本集
                        cores[i] = np.mean(items, axis=0)  # 以子样本集的均值作为当前质心的位置

            sorted_distance = np.sort(distance)  # 将矩阵的每一行升序排列
            # print(">> Index_sort shape: ")
            # print(sorted_distance.shape)

            min_distance_differ = np.abs(sorted_distance[:, 0].reshape(line_num, 1)
                                         - sorted_distance[:, 1].reshape(line_num, 1))

            # print(">> Min_distance_differ shape: ")
            # print(min_distance_differ.shape)

            # return result, cores 先不需要知道聚类的结果

            # 利用sklearn 自带的方法, 待看怎么取出每个点离cluster的距离
            # # print(input_x)
            # # print("x的值为：")
            # print(input_x.shape)
            # k_means = KMeans(n_clusters=CLUSTER_NUMS, max_iter=CLUSTER_MAX_ITER).fit(input_x)  # 关键点聚类
            # cluster_labels = k_means.labels_  # 返回标签以及聚类中心
            # cluster_center = k_means.cluster_centers_

            min_distance_differ_torch = torch.from_numpy(min_distance_differ)
            # print("Min_distance_differ_torch:")
            # print(min_distance_differ_torch)
            uncertainty = torch.cat((uncertainty, 10.00 / min_distance_differ_torch), 0)
            # print("get_uncertainty uncertainty: ", uncertainty)

    print(">> Clustering Over:")
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

        train_loader = DataLoader(cifar10_train, batch_size=BATCH,  # BATCH
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(cifar10_test, batch_size=BATCH)

        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        backbone_net = resnet.ResNet18().to(device)   #记住这才是原本pycharm里面的
        # backbone_net = ResNet18().to(device)   #这是Jupyter里面的

        models = {'backbone': backbone_net}
        torch.backends.cudnn.benchmark = True

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, cycle)
            acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc),
                  flush=True)

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)
            # print("main uncertainty: ", uncertainty)
            uncertainty = uncertainty.T
            # Index in ascending order
            arg = np.argsort(uncertainty).numpy().tolist()
            # print("main arg: ", arg)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())  # select largest loss
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # labeled_set += list(torch.tensor(subset)[arg][:ADDENDUM].numpy())  # select smallest influence
            # unlabeled_set = list(torch.tensor(subset)[arg][ADDENDUM:].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH,  # BATCH
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        torch.save({
            'trial': trial + 1,
            'state_dict_backbone': models['backbone'].state_dict()
            # 'state_dict_module': models['module'].state_dict()
        },
            './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))

        print('---------------------------Current Trial is done-----------------------------', flush=True)
