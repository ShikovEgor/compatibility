import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from src.models import PadTrain, PadTest, DataGenerator, ArticlesToVec
from pathlib import Path


class CompatModel(object): 
    def __init__(self, device='cpu'):
        self.B_SIZE = 16
        self.MAX_LEN = 5
        self.log_interval = 100 
        
        self.device = device
        
        self._get_train_data()
        
        self.file_avatar_vectors = 'data/avatar.json'
        try:
            self.avatar_vectors = pd.read_json(self.file_avatar_vectors, 
                                               orient = 'index',convert_axes=False)
        except:
            self.avatar_vectors = None
            print('avatar_vectors do not exist')
            
        self.w_f_name = 'weights/comp.pt'
        self.load_model(self.w_f_name, self.device)
        self._vect2label = weight_norm(nn.Linear(1,1, bias=False))
        self.coll_test = PadTest(max_len = self.MAX_LEN)

    def _create_model(self, arch = None):
        if arch is None:
            arch = {'inp_size':512, 'n_layers':2, 'n_hidden':16, 'nlat':16, 
            'kernel_size':2, 'dropout':0.2, 'seq_type':'tcn'}     
        self._model = ArticlesToVec(arch)
        
    def save_model(self, filename):
#        torch.save(self._model.state_dict(), filename)
        torch.save(self._model, filename)

    def load_model(self, filename, device):
        if not Path(filename).is_file():
            print('no weights found')
            return None
        self._model = torch.load(filename, map_location=device)    
#        self._model.load_state_dict(torch.load(filename, map_location=device))

    
    def _iterate(self, data):
        x1, x2, y_true = data['x1'], data['x2'] , data['y']       
        x1 = x1.to(device = self.device)
        x2 = x2.to(device = self.device)
        y_true = y_true.to(device = self.device)
        v1, v2 = self._model(x1), self._model(x2) 
        y_pred = ((v1-v2)**2).sum(dim=1).unsqueeze(1)
        y_pred = self._vect2label(y_pred).squeeze()         
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        return loss, y_true, y_pred
    
    def _train_epoch(self, epoch, dloader, log_interval):  
        train_loss = 0
        for i, data in enumerate(dloader):
            self._optimizer.zero_grad()
            loss, _ , _ = self._iterate(data)       
            loss.backward()
            self._optimizer.step()
            train_loss += loss.item()
            if i and (i % log_interval == 0):          
                print('loss  ',train_loss/i)                 
                
    def train(self, max_epochs = 10, reset_params = False, save_params = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if reset_params:
            self._create_model()
        self._model = self._model.to(device = self.device)
        self._vect2label = self._vect2label.to(device = self.device)
        model_params = []
        model_params += self._model.parameters()
        model_params += self._vect2label.parameters()
        self._optimizer = torch.optim.Adam(model_params, lr=0.001)
        
        coll_fn = PadTrain(self.MAX_LEN, self.B_SIZE)
        training_set = DataGenerator(self.df,self.vects)
        g_train = data.DataLoader(training_set, batch_size = self.B_SIZE,
                          shuffle=True, num_workers=9, collate_fn=coll_fn, drop_last=True)

        valid_set = DataGenerator(self.df,self.vects)
        g_valid = data.DataLoader(valid_set, batch_size = self.B_SIZE,
                          shuffle=False, num_workers=9, collate_fn=coll_fn, drop_last=True)

        for epoch in range(1, max_epochs):
            self._train_epoch(epoch, g_train, 100)
            self._validate(g_valid)
            
        self.save_model(self.w_f_name)

    
    def _validate(self, dloader):
        self._model.eval()
        valid_loss = 0 
        ar_true, ar_pred = [], []
        with torch.no_grad():
            for i, data in enumerate(dloader):
                loss, y_t, y_p = self._iterate(data)      
                valid_loss += loss.item()
                ar_true.append(y_t)
                ar_pred.append(y_p)
            print('valid_loss = ', valid_loss/i)
        y_true = torch.cat(ar_true)    
        y_pred = torch.sigmoid(torch.cat(ar_pred))  
        self.yy_t = y_true
        self.yy_p = y_pred
        ROC = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())    
        print('ROC = ', ROC)
        return ROC
    
    def _get_train_data(self):
        try:
            self.df = pd.read_json('data/sequences.json',orient = 'records')
        except:
            print('train sequences file does not exist')
        try:
            self.vects = pd.read_json('data/vectors_mean.json',orient = 'index')
        except:
            print('articles vectors file do not exist')
        
    def _get_seq_for_v(self):
        try:
            df_test = pd.read_json('data/seq_curr.json',orient = 'index',convert_axes=False)
        except:
            print('test sequences file does not exist')
        return df_test
    
    def _test(self, device = 'cpu'):
        df = self._get_seq_for_v()
        self._model = self._model.to(device = device)
        self._model.eval()    
        ar = []
        with torch.no_grad():
            for index, row in df.iterrows():
                ar.append(self._compute_vector(row['seq']))
                
        res = torch.cat(ar).cpu().numpy()
        df_res = pd.DataFrame(res, columns = list(range(res.shape[1])), index = df.index)
        self.avatar_vectors = df_res
        df_res.to_json(self.file_avatar_vectors, orient='index')
         
    def _compute_vector(self, seq):
        v = torch.FloatTensor( self.vects.loc[seq,:].values )
        v = self.coll_test(v.unsqueeze(0)).to(device = device)
        vect = self._model(v)
        return vect
    
    def _add_update_user(self, avatar_seq, add):
        if add:
            if avatar_seq['id'] in self.avatar_vectors.index:
                return
        df_test = self._get_seq_for_v()
        #write to sequence storage
        df_test.loc[avatar_seq['id']] = pd.Series({'seq': avatar_seq['seq']})
        
        #write to vectors storage
        self.avatar_vectors.loc[avatar_seq['id']] = self._compute_vector(avatar_seq['seq']).cpu().numpy()

    def add_users(self, data):
        """ 
        Calculates vectors for new users and writes to DB.
        """
        for d in data:
            self._add_user(d, add=True)

    def update_users(self, data):
        """ 
        Updates vectors for new users and writes to DB.
        """
        for d in data:
            self._add_user(d, add=False)
        
    def all_scores(self, curr_id):
        """ 
        Computes compatibility scores of user curr_id with all other users.
        Returns dict sorted by score: {userid_1:score1, userid_2:score2, ...}.
        """
        if self.avatar_vectors is None:
            print('no vectors data')
            return None
        if curr_id not in self.avatar_vectors.index:
            print('id not found')
            return None    
        
        #zamenit na dict
        self.avatar_vectors['scores'] = cosine_similarity(self.avatar_vectors.values, 
                                     self.avatar_vectors.loc[curr_id,:].values.reshape(1,-1)) *100
        return self.avatar_vectors['scores'].sort_values(ascending = False).to_dict()
    
    def single_score(self, curr_id, other_id):
        """ 
        Computes compatibility score of user curr_id with user other_id
        """
        if self.avatar_vectors is None:
            print('no vectors data')
            return None
        if curr_id not in self.avatar_vectors.index:
            print('id not found')
            return None  
        if other_id not in self.avatar_vectors.index:
            return None
        score = cosine_similarity(self.avatar_vectors.loc[other_id,:].values.reshape(1,-1), 
                                self.avatar_vectors.loc[curr_id,:].values.reshape(1,-1))  
        score = np.ravel(score)[0]
        return {other_id: score}
    