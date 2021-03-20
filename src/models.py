import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
from src.tcn import TemporalConvNet

class PadTrain(object): 
    def __init__(self, max_len, b_len):
        self.max_len = max_len 
        self.b_len = b_len 
        b_order = torch.tensor(list(range(b_len)), requires_grad=False)
        while True:
            self.new_order = torch.randperm(b_len)
            if not (self.new_order==b_order).any():
                break
        self.ones = torch.ones(b_len, dtype=torch.float, requires_grad=False)
        self.zeros = torch.zeros(b_len, dtype=torch.float, requires_grad=False)

    def __call__(self, batches):     
        out = {}
        for j,nm in enumerate(['x1','x2']):
            res = torch.zeros(self.b_len, self.max_len, 512, dtype=torch.float)
            for i in range(self.b_len):
                tl = batches[i][j].shape[0]
                res[i,-tl:] = batches[i][j]
            out[nm] = res 
        out['x1'] = torch.cat((out['x1'],out['x1']))
        out['x2'] = torch.cat( (out['x2'], out['x2'][self.new_order])   )
        out['y'] = torch.cat( (self.ones, self.zeros) )
        return out

class PadTest(object): 
    def __init__(self, max_len):
        self.max_len = max_len   
    def __call__(self, batches):     
        b_len = len(batches)       
        res = torch.zeros(b_len, max_len, 512, dtype=torch.float)
        for i in range(b_len):
            tl = batches[i].shape[0]
            res[i,-tl:] = batches[i]
        return res   

class DataGenerator(Dataset):
    def __init__(self, ind, data):
        self.ind = ind
        self.data = data

    def __len__(self):
        return len(self.ind)
    def __getitem__(self, idx):
        indices = self.ind.iloc[idx,:]
        
        v= self.data.loc[indices['seq_2'],:]
        v2= self.data.loc[indices['seq'],:]     
        return (torch.FloatTensor(v.values),torch.FloatTensor(v2.values))

class ArticlesToVec(nn.Module):
    def __init__(self, arch):
        super(ArticlesToVec, self).__init__()
        if arch['seq_type'] == 'tcn':
            self.seq_model = TemporalConvNet(arch['n_hidden'], [arch['nlat']]*arch['n_layers'], 
                                                 kernel_size=arch['kernel_size'], dropout=arch['dropout'])
        elif arch['seq_type'] == 'gru':
            self.seq_model = nn.GRU(arch['n_hidden'], arch['nlat'], num_layers=arch['n_layers'],batch_first =True)

        elif arch['seq_type'] == 'lstm':
            self.seq_model = nn.LSTM(arch['n_hidden'], arch['nlat'], num_layers=arch['n_layers'],batch_first =True)
        
        
        self.dropout = nn.Dropout(arch['dropout'])
        self.decoder = weight_norm(nn.Linear(arch['nlat'], arch['n_hidden']))
        self.convert_input = nn.Sequential(weight_norm(nn.Linear(arch['inp_size'], arch['n_hidden'])),nn.ReLU())

    def forward(self, x):
        x = self.convert_input(x)      
        out, _  = self.seq_model(x)
        return self.decoder(out[:, -1,:])

def calc_vect(items, model, reduce_mode='max'):
    tensors = torch.stack(model(items)) 
    if reduce_mode == 'max':
        return tensors.max(dim=0)
    elif reduce_mode == 'mean':
        return tensors.mean(dim=0)
    elif reduce_mode == 'no':
        return tensors
    return None

def reduce_tensors(data, reduce_mode='max'):
    df_fin = {}
    for i,it in data.items():
        it['vect_badges'] = calc_vect(it['badges'],model=model,reduce_mode=reduce_mode) 
        it['vect_abstract'] = calc_vect(it['sentences'],model=model,reduce_mode=reduce_mode)
        it['vect_name'] = calc_vect(it['name'],model=model,reduce_mode='no')
        res_tensor = torch.stack((it['vect_badges'],it['vect_abstract'],it['vect_name'])) 
        if reduce_mode == 'max':
            res_tensor =  res_tensor.max(dim=0)
        elif reduce_mode == 'mean':
            res_tensor = res_tensor.mean(dim=0)
        df_fin[it['id']] = res_tensor.cpu().numpy().tolist()     
        return df_fin

    
