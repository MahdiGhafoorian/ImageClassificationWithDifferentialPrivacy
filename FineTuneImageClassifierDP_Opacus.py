# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:09:06 2024

@author: Mahdi
"""

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Dataset
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

from torchvision.models import resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

import opacus
from opacus.validators import ModuleValidator

from statistics import mean
import matplotlib.pyplot as plt
import copy
import math

#---------------------------------------------------------------------------
import warnings
warnings.simplefilter("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu" 

print('device=',device)


MAX_GRAD_NORM = 1.2
EPSILON = 2.0
DELTA = 1e-5
EPOCHS = 20

BATCH_SIZE = 64

LR = 1e-3

deleted_item_idx = 1   

#---------------------------------------------------------------------------

class ParallelDataset(Dataset):
    def __init__(self, root, train=True, transform=None, remove_list=None):
        self.data = CIFAR10(root, train=train, download=True, transform=transform)
        self.data.data, self.data.targets = self.__remove__(remove_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __remove__(self, remove_list):
        data = np.delete(self.data.data, remove_list, axis=0)
        targets = np.delete(self.data.targets, remove_list, axis=0)
        return data, targets

#---------------------------------------------------------------------------

def train_withDP(backbone, head, criterion, optimizer, train_loader, device, epoch):
  accs = []
  losses = []
  for images, target in tqdm(train_loader):
    images = images.to(device)
    target = target.to(device)

    with torch.no_grad():
      images = backbone(images)

    logits = head(images)
    loss = criterion(logits, target)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    
    preds = logits.argmax(-1)
    n_correct = float(preds.eq(target).sum())
    batch_accuracy = n_correct / len(target)

    accs.append(batch_accuracy)
    losses.append(float(loss))
    
    # torch.cuda.empty_cache()  ## Enable to use bigger mini-batch

  epsilon = privacy_engine.get_epsilon(DELTA)  # Pointer to privacy_engine into optimizer

  print(
      f"\tTrain Epoch: {epoch} \t"
      f" Train Accuracy: {mean(accs):.6f}"
      f" Train Loss: {mean(losses):.6f}"
      f" (ε = {epsilon:.2f}, δ = {DELTA})"
  ) 
  return

def test_withDP(backbone, head, test_loader, privacy_engine, device):
  accs = []
  with torch.no_grad():
    for images, target in tqdm(test_loader):
      images = images.to(device)
      target = target.to(device)

      preds = head(backbone(images)).argmax(-1)
      n_correct = float(preds.eq(target).sum())
      batch_accuracy = n_correct / len(target)

      accs.append(batch_accuracy)
  epsilon = privacy_engine.get_epsilon(DELTA)
  print(
      f" Test Accuracy: {mean(accs):.6f}"
      f" (ε = {epsilon:.2f}, δ = {DELTA})"
  )
  return

#---------------------------------------------------------------------------

def train(backbone, head, criterion, optimizer, train_loader, device, epoch):
  accs = []
  losses = []
  for images, target in tqdm(train_loader):
    images = images.to(device)
    target = target.type(torch.LongTensor)
    target = target.to(device)

    with torch.no_grad():
      images = backbone(images)

    logits = head(images)
    loss = criterion(logits, target)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    
    preds = logits.argmax(-1)
    n_correct = float(preds.eq(target).sum())
    batch_accuracy = n_correct / len(target)

    accs.append(batch_accuracy)
    losses.append(float(loss))
    
    # torch.cuda.empty_cache()  ## Enable to use bigger mini-batch

  print(
      f"\tTrain Epoch: {epoch} \t"
      f" Train Accuracy: {mean(accs):.6f}"
      f" Train Loss: {mean(losses):.6f}"
  ) 
  return

def test(backbone, head, test_loader, device):
  accs = []
  with torch.no_grad():
    for images, target in tqdm(test_loader):
      images = images.to(device)
      target = target.to(device)

      preds = head(backbone(images)).argmax(-1)
      n_correct = float(preds.eq(target).sum())
      batch_accuracy = n_correct / len(target)

      accs.append(batch_accuracy)
  print(
      f" Test Accuracy: {mean(accs):.6f}"
  )
  return

#============================================================================
#============================================================================

if __name__ == '__main__':
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    # Main dataset **********************************************************
    
    train_ds = CIFAR10('.', 
                   train=True, 
                   download=True, 
                   transform=transform
                   # transform=Compose([ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    )       
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
    )
    
    test_ds = CIFAR10('.', 
                      train=False, 
                      download=True, 
                      transform=transform
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )    
    
    # Prallel Dataest ********************************************************
    
    
         

    # Create instances of the custom dataset for training and testing
    Parallel_train_ds = ParallelDataset(root='.', train=True, transform=transform, remove_list = deleted_item_idx)
    Parallel_test_ds = ParallelDataset(root='.', train=False, transform=transform, remove_list = deleted_item_idx)

    # Create data loaders
    Parallel_train_loader = torch.utils.data.DataLoader(Parallel_train_ds, batch_size=BATCH_SIZE, shuffle=False)
    Parallel_test_loader = torch.utils.data.DataLoader(Parallel_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # ************************************************************************
    
    # Now, you can iterate over the data loaders to get batches of CIFAR-10 data
    images, labels = next(iter(train_loader))
    deleted_item_image = images[deleted_item_idx]
    deleted_item_label = labels[deleted_item_idx]
    
    deleted_item_image = deleted_item_image.to(device)
    deleted_item_label = deleted_item_label.to(device)
          
    # data = x[1, :] # get a row data
    # single_img_reshaped = np.transpose(np.reshape(data,(3, 32,32)), (1,2,0))    
    # plt.imshow(single_img_reshaped)
    # plt.show()    
    
    # Define models **********************************************************
    
    # resnet18(pretrained=True)    
    
    resnet_modules = list(resnet18(pretrained=True).children())

    backbone = nn.Sequential(*resnet_modules[:-3])
    head = nn.Sequential(*resnet_modules[-3:-1], nn.Flatten(), nn.Linear(512, 10))
    
    Parallel_backbone = copy.deepcopy(backbone)
    Parallel_head = copy.deepcopy(head)
    
    backbone = backbone.eval().to(device)
    head = head.train().to(device)
    
    Parallel_head = Parallel_head.to(device)
    Parallel_backbone = Parallel_backbone.to(device)
    
    ## Sanity Check
    # with torch.no_grad():
    #     representation = backbone(x)

    # head(representation).shape
    
    
    ## converting batch norm to group norm    
    head = ModuleValidator.fix(head)
    head = head.to(device)
    backbone = backbone.to(device)
    #ModuleValidator.validate(head, strict=False)  ## validate if it works
    
    
    ## train
    criterion = nn.CrossEntropyLoss()
    
    ## The optimizer needs to point to the new model
    # optimizer = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9, nesterov=True) 
    optimizer = torch.optim.RMSprop(head.parameters(), lr=0.1)
    Parallel_optimizer = torch.optim.RMSprop(Parallel_head.parameters(), lr=0.1)
    
    privacy_engine = opacus.PrivacyEngine()

    head, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=head,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )
    
    # ************************************************************************
    
    for epoch in range(EPOCHS):
      train(Parallel_backbone, Parallel_head, criterion, Parallel_optimizer, Parallel_train_loader, device, epoch)
        
    # Test once 
    test(Parallel_backbone, Parallel_head, Parallel_test_loader, device)
        
    print('*'*100+'\n'+'*'*100)    
    
    for epoch in range(EPOCHS):
      train_withDP(backbone, head, criterion, optimizer, train_loader, device, epoch)
        
    # Test once 
    test_withDP(backbone, head, test_loader, privacy_engine, device)
    
    
    Parallel_ds_Model = torch.nn.Sequential(Parallel_backbone, Parallel_head)
    Parallel_ds_Model.eval()
    
    DP_Model = torch.nn.Sequential(backbone, head)
    DP_Model.eval()
    
    DifferntData = deleted_item_image.unsqueeze(dim=0)
    
    logits_Parallel_ds = Parallel_ds_Model(DifferntData)
    Probability_Parallel_ds = torch.nn.functional.softmax(logits_Parallel_ds)
    Pr_D = Probability_Parallel_ds[0][deleted_item_label.item()].item()
    
    logits_ds = DP_Model(DifferntData)
    Probability_ds = torch.nn.functional.softmax(logits_ds)
    Pr_D_prime = Probability_ds[0][deleted_item_label.item()].item()
    
    exp_eps = math.exp(EPSILON)
    
    print(
        f"\n\t Pr[M(D)∈S] ≤ (e^ε) × Pr[M(D')∈S] + δ \n"
        f"\t ------------------------------------ \n"
        f"\t {Pr_D:.5f} ≤ ({exp_eps:.5f}) * {Pr_D_prime:.5f} + {DELTA}"
    ) 
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

