from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import pickle
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

class LMsGraphDataset(Dataset):
    def __init__(self,dataset_file,smile2graph,seq2emb):
        self.smile2graph = smile2graph
        self.seq2emb = seq2emb
        self.dataset = pd.read_csv(dataset_file)
        self.max_seq_len = 1024

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,index):
        smile = self.dataset.iloc[index]['compound_iso_smiles']
        seq = self.dataset.iloc[index]['target_sequence']
        label = self.dataset.iloc[index]['affinity']
        label = float(label)

        c_size, features, edge_index = self.smile2graph[smile]
        GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([label]))
        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        
        seq_msg = torch.load(self.seq2emb[seq])
        seq_embed = seq_msg[0]
        seq_mask = seq_msg[1]

        return GCNData,seq_embed,seq_mask

if __name__ == '__main__':
    with open('data/seq2path_prot_xlnet.pickle', 'rb') as handle:
        seq2path = pickle.load(handle)
    with open('data/smile_graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

    train_dataset = LMsGraphDataset('data/davis_train.csv',smile2graph,seq2path)
    loader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=True)
    
    for graph,seq_embed,seq_mask in loader:
        print(graph)
        print(seq_embed.shape)
        break
    # graph,seq_embed,seq_mask = train_dataset.__getitem__(2)
    # print(graph.x.shape)
    # print(seq)    