# -*- coding: utf-8 -*-
# import necessary dependencies
import argparse
import os, sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import datasets
from torchsummary import summary
from torch.autograd import Variable
import PIL
from PIL import Image
from torch.utils.data import Subset
from tools.cutout.util.cutout import Cutout
from torch.utils.tensorboard import SummaryWriter
import random
import pandas as pd

print('Architecture List:')
print(torch.cuda.get_arch_list())
random.seed(10)
print(random.random())
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore')



#############################################################################
# ------------- Model Architecture----------------#

# Residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x.to(device))))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, numBlocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.layer1  = self.make_layer(block, 16, numBlocks[0], stride=1)
        self.layer2  = self.make_layer(block, 32, numBlocks[1], stride=2)
        self.layer3  = self.make_layer(block, 64, numBlocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1     = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, numBlocks, stride):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, numBlocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        return out


#############################################################################
# GPU check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Run on GPU...")
else:
    print("Run on CPU...")

net = ResNet(ResBlock, [3, 3, 3]).to(device)
net = net.to(device)
#summary(net, (3,32,32))


#####----------Mixup Data----#############
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
#######################Twesting function######################


def test_CIFAR10(net,criterion,root_data_dir,batch_size):
    transform_test = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    testset = torchvision.datasets.CIFAR10(root=root_data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = criterion

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    return test_loss / (num_val_steps), val_acc


'''
This part of the code is adopted from
https://github.com/tanimutomo/cifar10-c-eval/blob/b4b30257ddbd737c64bc3d2e961a2c47238b5249/src/dataset.py
'''


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(root, transform=transform,target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


#'/home/saman/sharedFolder/ECE661/Project/data/CIFAR10-C/'
def test_CIFAR10C(net,criterion,root_data_dir,batch_size,corruption_to_use='glass_blur'):
    transform_test = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    testset = CIFAR10C(root=root_data_dir,name=corruption_to_use, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = criterion

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    return test_loss / (num_val_steps), val_acc

##############################################
# Parameters

# a few arguments, do NOT change these
DATA_ROOT           = "./data"
CORRUPTED_DATA_ROOT = "/data/usr/ft42/nobackup/CIFAR_10_C/CIFAR-10-C/"
# the folder where the trained model is saved
CHECKPOINT_FOLDER   = "savedFiles/saved_models/"
# the folder where the figures are saved
FIGURES_FOLDER      = "savedFiles/saved_figures/"
# the folder where the csvs are saved
CSVS_FOLDER         = "savedFiles/saved_csvs/"
# name of the saved model
MODELNAME_REG       = "resnet20_mixup02_cutout12"

# hyperparameters, do NOT change right now
TRAIN_BATCH_SIZE = 256    # training batch size
VAL_BATCH_SIZE   = 100    # validation batch size
INITIAL_LR       = 0.1    # initial learning rate
MOMENTUM         = 0.9    # momentum for optimizer
REG              = 1e-3   # L2 regularization strength
EPOCHS           = 200    # total number of training epochs
DECAY_EPOCHS     = 20     # parameter for LR schedule (decay after XX epochs)
DECAY            = 0.5    # parameter for LR schedule (decay multiplier)
MIXUP_ALPHA_RANGE = [0.2]

# start the training/validation process
best_val_acc          = 0
current_learning_rate = INITIAL_LR
epochs                = np.linspace(1,EPOCHS,EPOCHS)



for Alpha_i in MIXUP_ALPHA_RANGE:

    # lists for saving training and validation accuracy and loss
    base_train_avg_loss = []
    base_train_avg_acc  = []
    base_valid_avg_loss = []
    base_valid_avg_acc  = []

    # Step 0: Model Definition

    # GPU check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    net = ResNet(ResBlock, [3, 3, 3]).to(device)
    net = net.to(device)
    net = torch.nn.DataParallel(net, device_ids = [0]).cuda()

    # Step 1: Preprocessing Function

    # specify preprocessing function
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # as per lecture instructions, data augmentation is only enabled in the
    # training process. It is recommended not to perform data augmentation on
    # validation or test dataset.
    transform_train.transforms.append(Cutout(n_holes=1, length=2))
    transform_val = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Step 2: Set up dataset and dataloader

    # construct dataset
    train_set = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    val_set = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform_val,
    )

    # construct dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_set,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    #Step 3: Set up the loss function and optimizer

    # create loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # Add optimizer
    optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)

    # Step 4: Start the training process.
    writer = SummaryWriter()
    start = time.time()
    print("==> Training starts!")
    print('using Alpha={} for Mixup regularization'.format(Alpha_i))
    print("="*50)
    MODELNAME = MODELNAME_REG+'_{}'.format(Alpha_i)
    for i in range(0, EPOCHS):
        # handle the learning rate scheduler.
        if i % DECAY_EPOCHS == 0 and i != 0:
            current_learning_rate = current_learning_rate * DECAY
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_learning_rate
            print("Current learning rate has decayed to %f" %current_learning_rate)

        #######################
        # switch to train mode
        net.train()
        #######################

        print("Epoch %d:" %i)
        # this help you compute the training accuracy
        total_examples = 0
        correct_examples = 0

        train_loss = 0 # track training loss if you want

        # Train the model for 1 epoch.
        for batch_idx, (images, targets) in enumerate(train_loader):
            ####################################

            # copy inputs to device
            images = images.to(device)
            targets = targets.to(device)
            images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=Alpha_i, use_cuda=True)
            images, targets_a, targets_b = map(Variable, (images,targets_a, targets_b))

            # compute the output and loss
            outputs = net(images)
            loss    = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # zero the gradient
            optimizer.zero_grad()

            # backpropagation
            loss.backward()

            # apply gradient and update the weights
            optimizer.step()

            # count the number of correctly predicted samples in the current batch
            _, predicted = torch.max(outputs, 1)
            correct = (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

            # Append to totals
            total_examples += targets.shape[0]
            train_loss += loss
            correct_examples += correct.item()
            ####################################

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = correct_examples / total_examples
        print("Training loss: %.4f, Training accuracy: %.4f" %(avg_train_loss, avg_train_acc))

        # append to list
        base_train_avg_loss.append(avg_train_loss)
        base_train_avg_acc.append(avg_train_acc)

        # Validate on the validation dataset
        #######################
        # switch to eval mode
        net.eval()

        #######################

        # this help you compute the validation accuracy
        total_examples = 0
        correct_examples = 0

        val_loss = 0 # again, track the validation loss if you want

        # disable gradient during validation, which can save GPU memory
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                ####################################
                # copy inputs to device
                images = images.to(device)
                targets = targets.to(device)

                # compute the output and loss
                outputs = net(images)
                loss = criterion(outputs, targets)
                # count the number of correctly predicted samples in the current batch
                _, predicted = torch.max(outputs, 1)
                correct = predicted.eq(targets).sum()

                # Append to totals
                total_examples += targets.shape[0]
                val_loss += loss
                correct_examples += correct.item()

                ####################################

        avg_validation_loss = val_loss / len(val_loader)
        avg_validation_acc = correct_examples / total_examples
        print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_validation_loss, avg_validation_acc))

        writer.add_scalars('loss', {'Loss/train'     : avg_train_loss,
                                    'Loss/validation': avg_validation_loss},i)
        writer.add_scalars('accuracy', {'Acc/train'      : avg_train_acc,
                                        'Acc/validation' :avg_validation_acc}, i)

        # append to list
        base_valid_avg_loss.append(avg_validation_loss)
        base_valid_avg_acc.append(avg_validation_acc)

        # save the model checkpoint
        if avg_validation_acc > best_val_acc:
            best_val_acc = avg_validation_acc
            if not os.path.exists(CHECKPOINT_FOLDER):
               os.makedirs(CHECKPOINT_FOLDER)
            print("Saving ...")
            state = {'state_dict': net.state_dict(),
                    'epoch': i,
                    'lr': current_learning_rate}
            torch.save(state, os.path.join(CHECKPOINT_FOLDER, MODELNAME+".pth"))
            torch.save(net.module.state_dict(), CHECKPOINT_FOLDER+MODELNAME+'.pt')

        print('')

    print("="*50)
    print(f"==> Optimization finished! Best validation accuracy: {best_val_acc:.4f}")
    end = time.time()
    print("Total Executation Time :",(end-start) * 10**3, "ms")

    timer_df = pd.DataFrame(data={'Alpha':Alpha_i,'training_time_ms':(end-start) * 10**3},index=[0])
    timer_df.to_csv(CSVS_FOLDER+MODELNAME+"mixup_tr_val_time.csv",index=False,encoding='utf-8')



    torch.save(net.module.state_dict(), CHECKPOINT_FOLDER+MODELNAME+'.pt')
    #############################################
    base_train_avg_loss = [np.float(base_train_avg_loss[i].cpu().detach().numpy()) for i in range(len(base_train_avg_loss))]
    base_valid_avg_loss = [np.float(base_valid_avg_loss[i].cpu().detach().numpy()) for i in range(len(base_valid_avg_loss))]
    #############################################
    plt.figure(figsize=(8,4))
    plt.plot(epochs, base_train_avg_loss, color='red')
    plt.plot(epochs, base_valid_avg_loss, color='blue')
    plt.legend(['train','validation'])
    plt.grid()
    plt.title('Loss vs. Number of Epochs')
    plt.savefig(FIGURES_FOLDER+MODELNAME+"mixup_loss.png", quality=95, dpi=500)
    #lt.show()

    plt.figure(figsize=(8,4))
    plt.plot(epochs, base_train_avg_acc, color='red')
    plt.plot(epochs, base_valid_avg_acc, color='blue')
    plt.legend(['train','validation'])
    plt.grid()
    plt.title('Accuracy vs. Number of Epochs')
    plt.savefig(FIGURES_FOLDER+MODELNAME+"mixup_acc.png", quality=95, dpi=500)
    #plt.show()

    info_df = pd.DataFrame(list(zip(base_train_avg_loss,base_valid_avg_loss,base_train_avg_acc,base_valid_avg_acc)),columns=['train_loss','valid_loss','train_acc','valid_acc'])
    info_df.to_csv(CSVS_FOLDER+MODELNAME+"mixup_tr_val.csv",index=False,encoding='utf-8')







    #-------------- Testingh---------------#
    test_data_list=[]
    test_acc_list =[]
    test_loss_list=[]


    net = ResNet(ResBlock, [3, 3, 3]).to(device)
    net = net.to(device)
    net.load_state_dict(torch.load(CHECKPOINT_FOLDER+MODELNAME+'.pt',map_location='cuda:0'))
    test_loss,test_acc=test_CIFAR10(net,criterion,root_data_dir=DATA_ROOT,batch_size=VAL_BATCH_SIZE)

    test_data_list.append('Original')
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)




    corrupted_data_type = ['brightness','contrast','defocus_blur','elastic_transform','fog','frost','gaussian_blur',
                           'gaussian_noise','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate',
                           'saturate','shot_noise','snow','spatter','speckle_noise','zoom_blur']



    for corruption_i in corrupted_data_type:
        print('Corruption type:{}'.format(corruption_i))
        test_loss,test_acc=test_CIFAR10C(net,criterion,root_data_dir=CORRUPTED_DATA_ROOT,batch_size=VAL_BATCH_SIZE,corruption_to_use=corruption_i)

        test_data_list.append(corruption_i)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

    info_df_test = pd.DataFrame(list(zip(test_data_list,test_loss_list,test_acc_list)),columns=['Data','test_loss','test_acc'])
    info_df_test.to_csv(CSVS_FOLDER+MODELNAME+"mixup_test.csv",index=False,encoding='utf-8')

