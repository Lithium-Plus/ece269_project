import torch
import torch.nn as nn
import torchvision.models as models




def get_backbone():
    network = models.resnet18(False) #not pretrain
    backbone = torch.nn.Sequential(*(list(network.children())[:-1]))
    return backbone

#in_dim = output feature dimension of backbone
def projection_mlp(in_dim = 512, hidden_dim = 2048):
    l1 = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                   nn.BatchNorm1d(hidden_dim),
                   nn.ReLU(inplace=True))
    l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                   nn.BatchNorm1d(hidden_dim),
                   nn.ReLU(inplace=True))
    l3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
               nn.BatchNorm1d(hidden_dim))
    return nn.Sequential(l1, l2, l3)

#in_dim = outdim of proj mlp
def prediction_mlp(in_dim = 2048,hidden_dim = 512):
    l1 = nn.Sequential(nn.Linear(in_dim, hidden_dim),
               nn.BatchNorm1d(hidden_dim),
               nn.ReLU(inplace=True))
    l2 = nn.Linear(hidden_dim, in_dim)
    return nn.Sequential(l1,l2)
'''
Implements SimSiam with ResNet18 Backbone.
Note: Dimensions will change with other network
'''
class simsiam(nn.Module):
    def __init__(self):
        super(simsiam, self).__init__()
        self.backbone = get_backbone()
        self.proj_mlp = projection_mlp()
        self.pred_mlp = prediction_mlp()
        
    def forward(self, x0,x1):
        b = x0.shape[0]
        z0,z1 = self.backbone(x0).view(b,-1),self.backbone(x1).view(b,-1)
        z0,z1 = self.proj_mlp(z0),self.proj_mlp(z1)
        p0,p1 = self.pred_mlp(z0),self.pred_mlp(z1)
        out0,out1 = (z0.detach(),p0), (z1.detach(),p1)
        return out0,out1
    
    
def main():
    ss_ = simsiam()
    img = torch.rand(2,3,28,28)
    out0,out1 = ss_(img,img)
    z0,p0 = out0
    z1,p1 = out1
    cos = nn.CosineSimilarity(dim = 1)
    L = (-0.5*(cos(p0,z1) + cos(p1,z0))).mean()
    
    
    print(out0[0].shape, out0[1].shape, L)
    
    
    
if __name__ == "__main__":
    main()