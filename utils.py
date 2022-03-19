import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import argparse
import sys
from dataset import get_dataset
import os
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class Net(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512,7)
    def forward(self, x):
        x = self.backbone(x).view(-1,512)
        return self.fc(x)
    

def train(model, criterion, optimizer, dataloader):
    
    epoch_loss, total,correct = 0,0,0
    model.train()
    for x,y in dataloader:
        B = y.size(0)
        
        x = x.to(device)
        y = y.to(device)
        y = y.view(B).long()

        logits = model(x)
        preds = torch.argmax(logits,1)

        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += (loss.item()/len(dataloader))
        total += y.size(0)
        correct += (preds == y).sum().item()
    return (correct/total), epoch_loss

def  val(model, criterion,dataloader):
    
    epoch_loss, total,correct = 0,0,0
    model.eval()
    for x,y in dataloader:
        B = y.size(0)
        
        x = x.to(device)
        y = y.to(device)
        y = y.view(B).long()

        logits = model(x)
        preds = torch.argmax(logits,1)

        loss = criterion(logits,y)
        epoch_loss += (loss.item()/len(dataloader))
        total += y.size(0)
        correct += (preds == y).sum().item()
    return (correct/total), epoch_loss
   