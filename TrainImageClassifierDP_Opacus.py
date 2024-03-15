# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:36:38 2024

@author: Mahdi
"""
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import models

import torch.nn as nn
import torch.optim as optim

from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

# from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm

import warnings
warnings.simplefilter("ignore")

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20

LR = 1e-3

BATCH_SIZE = 128
MAX_PHYSICAL_BATCH_SIZE = 64

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

#---------------------------------------------------------------------------

def accuracy(preds, labels):
    return (preds == labels).mean()

#---------------------------------------------------------------------------

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):   
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

#---------------------------------------------------------------------------

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)
    epsilon = privacy_engine.get_epsilon(DELTA)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
        f"(ε = {epsilon:.2f}, δ = {DELTA})"
    )
    return np.mean(top1_acc)

#---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

DATA_ROOT = '../cifar10'
# DATA_ROOT = '../cifar-10-batches-py'

train_dataset = CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)

test_dataset = CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = models.resnet18(num_classes=10)

model = ModuleValidator.fix(model)  ## find best repacement for incompatible modules
ModuleValidator.validate(model, strict=False) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)


privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)


print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")


for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
# for epoch in tqdm(range(EPOCHS)):
        train(model, train_loader, optimizer, epoch + 1, device)

top1_acc = test(model, test_loader, device)





