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
from SimSiam import simsiam
import os
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config():
    parser = argparse.ArgumentParser(description='SimSiam Training')
    parser.add_argument('--lr',default = 0.01,type = float, help = 'Learning Rate')
    parser.add_argument('--wd',default = 0.0008,type = float, help = 'Weight Decay')
    parser.add_argument('--batch_size',default = 512, type = int)
    parser.add_argument('--start_epoch',default = 0, type = int)
    parser.add_argument('--epochs',default = 700, type = int)
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint')
    
    return parser.parse_args()

def save_checkpoint(model,optimizer,epoch ,filename = 'last.pth.tar'):
    torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, f"results/checkpoints/{filename}")
def train(model,optimizer,dataloader, config):
    
    criterion = nn.CosineSimilarity(dim=1)
    for epoch in range(config["start_epoch"], config["epochs"]):
        epoch_loss = 0
        losses = []
        stds = []
        for X,Y in dataloader:
            img0,img1= X[0],X[1]
            img0,img1,Y = img0.to(device),img1.to(device),Y.to(device)
            
            
            out0,out1= model(img0,img1)
            z0,p0 = out0
            z1,p1 = out1
            
            loss = -0.5*(criterion(p0,z1).mean() + criterion(p1,z0).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += (loss/len(dataloader))
            
            losses.append(str(loss.detach().cpu().numpy().item()))
            
            t = z0/(torch.norm(z0))
            stds.append(t.std(axis = 1).mean().detach().cpu().numpy().item())
            
        print(f"Epoch: {epoch}, Loss: {epoch_loss} Std: {np.array(stds).mean()}")
        save_checkpoint(model,optimizer,epoch)
        with open("ssl_loss.txt", "a") as file:
            file.write('\n'.join(losses))
            file.write('\n')
        
            

def main():
    
    dirs = ["results","results/checkpoints"]
    for d in dirs:
        if(not os.path.isdir(d)):
            os.makedirs(d)
            print(f"Creating directory: {d}")
    config = vars(get_config())
    print(config)
    
    epochs,lr = config["epochs"], config["lr"] * (config["batch_size"]/256)
    dataset = get_dataset("../data")
    
    ssl_loader =  DataLoader(dataset['unlabel_set'], batch_size= config["batch_size"],shuffle = True, pin_memory = True,num_workers = 4)
    
    model = simsiam()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr ,weight_decay = config["wd"],momentum=0.9)
    if(config["resume"] != ""):
        #load checkpoint and state_dicts, update start_epoch
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        config["start_epoch"] = epoch = checkpoint['epoch']
        
    train(model,optimizer,ssl_loader,config)
    

    
if __name__ == "__main__":
    main()