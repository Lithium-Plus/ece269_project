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
from VicReg import vicreg,inv_loss,cov_loss,var_loss
import os
import json
from optimizer import LARS,adjust_learning_rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config():
    parser = argparse.ArgumentParser(description='SimSiam Training')
    parser.add_argument('--lr',default = 0.2,type = float, help = 'Learning Rate')
    parser.add_argument('--wd',default = 1e-6,type = float, help = 'Weight Decay')
    parser.add_argument('--batch_size',default = 512, type = int)
    parser.add_argument('--start_epoch',default = 0, type = int)
    parser.add_argument('--epochs',default = 1000, type = int)
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint')
    parser.add_argument('--ld',default = 25, type = int)
    parser.add_argument('--mu',default = 25, type = int)
    parser.add_argument('--nu',default = 1, type = int)
    
    
    return parser.parse_args()

def save_checkpoint(model,optimizer,epoch,filename = 'last.pth.tar'):
    torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, f"results/checkpoints/{filename}")
    
def train(model,optimizer,dataloader, config):
    
    for epoch in range(config["start_epoch"], config["epochs"]):
        epoch_loss = 0
        losses = []
        for step,(X,Y) in enumerate(dataloader,start = epoch*len(dataloader)):
            adjust_learning_rate(config, optimizer, dataloader, step)
            optimizer.zero_grad()
            
            
            img0,img1= X[0],X[1]
            img0,img1 = img0.to(device),img1.to(device)
            
            
            z0,z1= model(img0,img1)

            
            loss = config["ld"]*inv_loss(z0,z1) + config["mu"]*var_loss(z0,z1) + config["nu"]*cov_loss(z0,z1)
            loss.backward()
            optimizer.step()
            
            epoch_loss += (loss.item()/len(dataloader))
            
            losses.append(str(loss.detach().cpu().numpy().item()))

            
        print(f"Epoch: {epoch}, Loss: {epoch_loss},step: {step}, LR: {optimizer.param_groups[0]['lr']}")
        save_checkpoint(model,optimizer,epoch)
        with open("ssl_loss.txt", "a") as file:
            file.write('\n'.join(losses))
            file.write('\n')
        
            

def main():
    
    step = 0
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
    
    model = vicreg()
    model.to(device)
    

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay= config["wd"],
                 weight_decay_filter=True,
                 lars_adaptation_filter=True)

    
    #optimizer = optim.SGD(model.parameters(), lr = lr ,weight_decay = config["wd"],momentum=0.9)
    
    if(config["resume"] != ""):
        #load checkpoint and state_dicts, update start_epoch
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        config["start_epoch"] =  checkpoint['epoch']
        
        st = config["start_epoch"]
        print(f"Resuming at : {st}")
    train(model,optimizer,ssl_loader,config)
    

    
if __name__ == "__main__":
    main()