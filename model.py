import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GCNNet(torch.nn.Module):
    def __init__(self,graph_output_dim=128,graph_features_dim=78,dropout=0.2):
        super(GCNNet,self).__init__()
        self.conv1 = GCNConv(graph_features_dim, graph_features_dim)
        self.conv2 = GCNConv(graph_features_dim, graph_features_dim*2)
        self.conv3 = GCNConv(graph_features_dim*2, graph_features_dim * 4)
        self.fc_g1 = torch.nn.Linear(graph_features_dim*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        
        return x

class GINConvNet(torch.nn.Module):
    def __init__(self,graph_output_dim=128,graph_features_dim=78,dropout=0.2):
        super(GINConvNet,self).__init__()
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # convolution layers
        nn1 = Sequential(Linear(graph_features_dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, graph_output_dim)
    
    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return x 

class GAT_GCN(torch.nn.Module):
    def __init__(self,graph_output_dim=128,graph_features_dim=78,dropout=0.2):
        super(GAT_GCN,self).__init__()
        self.conv1 = GATConv(graph_features_dim, graph_features_dim, heads=10)
        self.conv2 = GCNConv(graph_features_dim*10, graph_features_dim*10)
        self.fc_g1 = torch.nn.Linear(graph_features_dim*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        return x


class GATNet(torch.nn.Module):
    def __init__(self,graph_output_dim=128,graph_features_dim=78,dropout=0.2):
        super(GATNet,self).__init__() 
        self.gcn1 = GATConv(graph_features_dim, graph_features_dim, heads=10, dropout=dropout)
        self.gcn2 = GATConv(graph_features_dim * 10, graph_output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(graph_output_dim, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        return x 

class SeqNet(torch.nn.Module):
    def __init__(self,seq_embed_dim=1024,n_filters=256,seq_output_dim=1024,dropout=0.2):
        super(SeqNet,self).__init__()
        
        self.conv_xt_1 = nn.Conv1d(in_channels=seq_embed_dim, out_channels=n_filters, kernel_size=5)
        self.pool_xt_1 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters,out_channels=seq_embed_dim,kernel_size=5)
        self.pool_xt_2 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.conv_xt_3 = nn.Conv1d(in_channels=seq_embed_dim,out_channels=n_filters,kernel_size=5)
        self.pool_xt_3 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.conv_xt_4 = nn.Conv1d(in_channels=n_filters,out_channels=int(n_filters/2),kernel_size=3)
        self.pool_xt_4 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.fc1_xt = nn.Linear(128*61, seq_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
   
    def forward(self,seq_embed,seq_mask=None):
        # 1d conv layers
        xt = self.conv_xt_1(seq_embed.transpose(1,2))
        xt = self.relu(xt)
        xt = self.pool_xt_1(xt)
        xt = self.conv_xt_2(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_2(xt)
        xt = self.conv_xt_3(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_3(xt)
        xt = self.conv_xt_4(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_4(xt)

        # flatten
        xt = xt.view(-1, 128*61)
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        return xt

# LMs_Graph based model
class LMsGraphModel(torch.nn.Module):
    def __init__(self,config):
        super(LMsGraphModel, self).__init__()
        dropout = config['dropout']
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # SMILES graph branch
        if config['graphNet'] == 'GCN':
            self.graph = GCNNet(config['graph_output_dim'],config['graph_features_dim'],dropout)
        elif config['graphNet'] == 'GIN':
            self.graph = GINConvNet(config['graph_output_dim'],config['graph_features_dim'],dropout)
        elif config['graphNet'] == 'GAT_GCN':
            self.graph = GAT_GCN(config['graph_output_dim'],config['graph_features_dim'],dropout)
        elif config['graphNet'] == 'GAT':
            self.graph = GATNet(config['graph_output_dim'],config['graph_features_dim'],dropout)
        else:
            print("Unknow model name")
        
        # Seq branch
        self.seqnet = SeqNet(config['seq_embed_dim'],config['n_filters'],config['seq_output_dim'],dropout)

        # combined layers
        self.fc1 = nn.Linear(config['graph_output_dim']+config['seq_output_dim'], 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, graph,seq_embed,seq_mask=None):
        graph_output = self.graph(graph)
        seq_output = self.seqnet(seq_embed,seq_mask)
        # concat
        xc = torch.cat((graph_output, seq_output), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        out = self.out(xc)
        return out

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    import pickle
    from dataset import LMsGraphDataset
    from config import LMsGraphModelConfig
    
    with open('data/seq2path_prot_bert.pickle', 'rb') as handle:
        seq2path = pickle.load(handle)
    with open('data/smile_graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

    train_dataset = LMsGraphDataset('data/davis_train.csv',smile2graph,seq2path)
    loader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = LMsGraphModelConfig()
    print(config['graphNet'])
    model = LMsGraphModel(config).to(device)
    # graph_ = GAT_GCN().to(device)

    for i,data in enumerate(loader):
        graph,seq_embed,seq_mask = data
        graph = graph.to(device)
        seq_embed = seq_embed.to(device)
        seq_mask = seq_mask.to(device)
        out = model(graph,seq_embed)
        # out = graph_(graph)
        print(out.shape)
        break

    

   