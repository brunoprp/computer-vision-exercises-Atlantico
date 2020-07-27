#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:56:15 2020

@author: bruno
"""
import datasets
import decorator
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import numpy as np

# Data parameters
data_folder = '/media/bruno/HD-Arquivos2/Data_Object_Detect/'  # folder with data files

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keep_difficult = True  # use objects considered difficult to detect?
# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 15  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
epocas = 20 # epochs train
cudnn.benchmark = True


class TrainObjDetect():
    def __init__(self, keep_difficult, checkpoint, batch_size, 
                 iterations, workers, print_freq, lr, decay_lr_at, decay_lr_to, 
                momentum, weight_decay, grad_clip):
        # Learning parameters
        self.keep_difficult = keep_difficult  # use objects considered difficult to detect
        self.checkpoint = checkpoint  # path to model checkpoint, None if none
        self.batch_size = batch_size  # batch size
        self.iterations = iterations  # number of iterations to train
        self.workers = workers  # number of workers for loading data in the DataLoader
        self.print_freq = print_freq  # print training status every __ batches
        self.lr = lr  # learning rate
        self.decay_lr_at = decay_lr_at  # decay learning rate after these many iterations
        self.decay_lr_to = decay_lr_to  # decay learning rate to this fraction of the existing learning rate
        self.momentum = momentum  # momentum
        self.weight_decay = weight_decay  # weight decay
        self.grad_clip = grad_clip  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
    
    def trainModel(self, train_loader, model, criterion, optimizer, epoch):
        """
        :param train_loader: DataLoader for training data
        :param model: model
        :param criterion: MultiBox loss
        :param optimizer: optimizer
        :param epoch: epoch number
        """
        model.train()  # training mode enables dropout

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss

        start = time.time()
        #loss_interation = []
            # Batches
        for i, (images, boxes, labels, _) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients, if necessary
            if self.grad_clip is not None:
                clip_gradient(optimizer, self.grad_clip)

            # Update model
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()
            #loss_interation.append(loss) ## Loss interation

            # Print status
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses))
        
        del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
        #return loss_interation 

    
    
    def startTrain(self, epochs , data_folder):
        """
        Training.
        """
        
        global start_epoch, label_map
        # Initialize model or load checkpoint
        if self.checkpoint is None:
            start_epoch = 0
            model = SSD300(n_classes=n_classes)
            # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
            biases = list()
            not_biases = list()
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)
            optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                        lr= self.lr, momentum= self.momentum, weight_decay= self.weight_decay)
    
        else:
            self.checkpoint = torch.load(self.checkpoint)
            start_epoch = self.checkpoint['epoch'] + 1
            print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
            model = self.checkpoint['model']
            optimizer = self.checkpoint['optimizer']
    
        # Move to default device
        model = model.to(device)
        criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    
        # Custom dataloaders
        train_dataset = PascalVOCDataset(data_folder,
                                         split='train',
                                         keep_difficult=self.keep_difficult)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True,
                                                   collate_fn=train_dataset.collate_fn, num_workers= self.workers,
                                                   pin_memory=True)  # note that we're passing the collate function here
    
        # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
        # To convert iterations to epochs, divide iterations by the number of iterations per epoch
        # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
        #epochs = self.iterations // (len(train_dataset) // 32)
        self.decay_lr_at = [it // (len(train_dataset) // 32) for it in self.decay_lr_at]
    
        # Epochs
        #los_epochs = []
        for epoch in range(start_epoch, epochs):
    
            # Decay learning rate at particular epochs
            if epoch in self.decay_lr_at:
                adjust_learning_rate(optimizer, self.decay_lr_to)
    
            # One epoch's training
            self.trainModel(train_loader=train_loader,
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=epoch)
            #los_epochs.append(loss)
    
            # Save checkpoint
            save_checkpoint(epoch, model, optimizer)
        #return los_epochs

train_obj = TrainObjDetect(keep_difficult, checkpoint, batch_size, iterations, 
                          workers, print_freq, lr, decay_lr_at, decay_lr_to, momentum,
                          weight_decay, grad_clip)

train_obj.startTrain(epocas,data_folder)