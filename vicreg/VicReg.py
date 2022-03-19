import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



def get_backbone():
    network = models.resnet18(False) #not pretrain
    backbone = torch.nn.Sequential(*(list(network.children())[:-1]))
    return backbone

#in_dim = output feature dimension of backbone
def expander(in_dim = 512, hidden_dim = 8912):
    l1 = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                   nn.BatchNorm1d(hidden_dim),
                   nn.ReLU(inplace=True))
    l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                   nn.BatchNorm1d(hidden_dim),
                   nn.ReLU(inplace=True))
    l3 = nn.Linear(hidden_dim,hidden_dim)
    return nn.Sequential(l1, l2, l3)


'''
Implements VicReg with ResNet18 Backbone.
Note: Dimensions will change with other network
'''
class vicreg(nn.Module):
    def __init__(self):
        super(vicreg, self).__init__()
        self.backbone = get_backbone()
        self.expander = expander()
        
    def forward(self, x0,x1):
        b = x0.shape[0]
        y0,y1 = self.backbone(x0).view(b,-1),self.backbone(x1).view(b,-1)
        z0,z1 = self.expander(y0),self.expander(y1)
        return z0,z1

    
inv_loss = nn.MSELoss() #invariance loss

#variance regularization
def var_loss(x,y,eps = 0.0001):
    std_x,std_y = torch.sqrt(x.var(0)+eps), torch.sqrt(y.var(0)+eps)
    return torch.mean(F.relu(1-std_x)) + torch.mean(F.relu(1-std_y))



#off diagonal term taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cov_loss(x,y):
    n,d = x.shape
    x,y = (x-x.mean(0)), (y-y.mean(0))
    covx,covy = (x.T@x)/(n-1), (y.T@y)/(n-1)
    
    lx = off_diagonal(covx).pow_(2).sum() / d
    ly = off_diagonal(covy).pow_(2).sum() / d
    return lx + ly

def main():
    vr = vicreg()
    img = torch.rand(2,3,28,28)
    z0,z1 = vr(img,img)

    loss = var_loss(z0,z1) + cov_loss(z0,z1) + inv_loss(z0,z1)
    
    
    print(z0[0].shape, z1[1].shape, loss.item())
    
if __name__ == "__main__":
    main()