def LMsGraphModelConfig():
    config = {}
    config['graphNet'] = 'GCN' # option: GCN、GAT、GIN 、GAT_GCN
    config['LMsNet'] = 't5'
    config['dropout'] = 0.2
    config['graph_output_dim'] = 128
    config['graph_features_dim'] = 78
    config['seq_embed_dim'] =  4096  # if albert seq_embed_dim = 4096, else 1024
    config['n_filters'] = 256
    config['seq_output_dim'] = 1024
    return config

