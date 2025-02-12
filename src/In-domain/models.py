import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.dense.linear import Linear

class GCNBackbone(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(GCNBackbone,self).__init__()
        self.conv1 = GCNConv(input_dim,hidden_dim)
        self.conv2 = GCNConv(hidden_dim,hidden_dim)

    def forward(self, data):
        x, edge_index = data.x.to(dtype=torch.float32), data.edge_index
        if (hasattr(data,'train_pos_edge_index')):
            edge_index = data.train_pos_edge_index
        x = self.conv1(x , edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
    
    def decode(self,z,edge_index):
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)

    
class SageBackbone(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(SageBackbone,self).__init__()
        self.conv1 = SAGEConv(input_dim,hidden_dim)
        self.conv2 = SAGEConv(hidden_dim,hidden_dim)

    def forward(self, data):
        x, edge_index = data.x.to(dtype=torch.float32), data.edge_index
        if (hasattr(data,'train_pos_edge_index')):
            edge_index = data.train_pos_edge_index
        x = self.conv1(x , edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
    
    def decode(self,z,edge_index):
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)

class GNNMLP(nn.Module):
    def __init__(self,hidden_dim, output_dim):
        super(GNNMLP,self).__init__()
        self.mlp = Linear(hidden_dim,output_dim)

    def forward(self, x):
        return self.mlp(x)
    
class GNNComplete(nn.Module):
    def __init__(self,backbone,mlp):
        super(GNNComplete,self).__init__()
        self.backbone = backbone
        self.mlp = mlp
    def forward(self,data):
        x = self.backbone(data)
        return self.mlp(x)
