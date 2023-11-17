from torch_geometric.nn import GCNConv 
import torch

MANUAL_SEED = 1234

# define a simple 2-layer GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(MANUAL_SEED)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        head = torch.nn.Linear(hidden_channels, num_classes)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.head(x)
        return x