import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import argparse
import sys
sys.path.append('../')
from dataset import get_dataset
import os
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_config():
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--lr',default = 0.01,type = float, help = 'Learning Rate')
    parser.add_argument('--wd',default = 0,type = float, help = 'Weight Decay')
    
    parser.add_argument('--epochs',default = 100, type = int)
    
    return parser.parse_args()



def train(model, criterion, optimizer, dataloader):
    
    epoch_loss, total,correct = 0,0,0
    model.train()
    for x,y in dataloader:
        B = y.size(0)
        
        x = x.to(device)
        x = x.view(B,-1)
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
        x = x.view(B,-1)
        y = y.to(device)
        y = y.view(B).long()

        logits = model(x)
        preds = torch.argmax(logits,1)

        loss = criterion(logits,y)
        epoch_loss += (loss.item()/len(dataloader))
        total += y.size(0)
        correct += (preds == y).sum().item()
    return (correct/total), epoch_loss

def main():
    
    config = vars(get_config())
    print(config)
    epochs = config["epochs"]
    dataset = get_dataset("../data")
    train_loader =  DataLoader(dataset['train_set'], batch_size=256,shuffle = True, pin_memory = True,num_workers = 4)
    val_loader =  DataLoader(dataset['val_set'], batch_size=256,shuffle = False, pin_memory = True,num_workers = 4)
    test_loader =  DataLoader(dataset['test_set'], batch_size=256,shuffle = False, pin_memory = True,num_workers = 4)
    model = nn.Linear(28*28*3,7)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= config["lr"],weight_decay = config["wd"],momentum=0.9)
    
    best_acc = -1
    for epoch in range(epochs):
        train_acc,train_loss = train(model,criterion,optimizer,train_loader)
        val_acc,val_loss = val(model,criterion,val_loader)
        if(best_acc < val_acc):
            best_acc = val_acc
            torch.save(model.state_dict(),"results/checkpoints/lin.pth")
            
        print(f"Epoch:{epoch} Train Acc:{train_acc} Loss: {train_loss} VAL ACC: {val_acc} loss:{val_loss}")
    model.load_state_dict(torch.load("results/checkpoints/lin.pth"))
    test_acc,test_loss = val(model,criterion,test_loader)
    print(f"Test Acc:{test_acc} Loss: {test_loss}")
if __name__ == "__main__":
    main()
