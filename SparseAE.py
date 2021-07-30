# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:04:04 2021

@author: huangyuyao
"""



import torch.nn as nn
import torch.nn.functional
import torch.optim as optim



import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import time
import matplotlib.pyplot as plt



import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy
from glob import glob
from scipy.io import mmread
from sklearn.preprocessing import LabelEncoder

from torchvision import transforms, datasets
from torch import nn, optim
from torch.nn import init
from tqdm import trange
import pickle
import sys

class SingleCellDataset(Dataset):


    def __init__(self, path, 
                 low = 0,
                 high = 0.9,
                 min_peaks = 0,
                 transpose = False,
                 transforms=[]):
        
        self.data, self.peaks, self.barcode = load_data(path, transpose)
        
        if min_peaks > 0:
            self.filter_cell(min_peaks)
        
        self.filter_peak(low, high)
        
        for transform in transforms:
            self.data = transform(self.data)
        
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index];
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return data
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nPeak number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')
        
    def filter_peak(self, low=0, high=0.9):
        """
        Removes rare peaks and common peaks 
            low: low ratio threshold to remove the rare peaks
            high: high ratio threshold to remove the common peaks
        """
        total_cells = self.data.shape[0]
        count = np.array((self.data >0).sum(0)).squeeze()
#         indices = np.where(count > 0.01*X*total_cells)[0] 
        indices = np.where((count > low*total_cells) & (count < high*total_cells))[0] 
        self.data = self.data[:, indices]
        self.peaks = self.peaks[indices]
        print('filterpeak------')
        
    def filter_cell(self, min_peaks=0):
        """
        Remove low quality cells by threshold of min_peaks
            min_peaks: if >= 1 means the min_peaks number else is the ratio
        """
        if min_peaks < 1:
            min_peaks = len(self.peaks)*min_peaks
        indices = np.where(np.sum(self.data>0, 1)>=min_peaks)[0]
        self.data = self.data[indices]
        self.barcode = self.barcode[indices]
        p = type(self.barcode)
        print('filtercell------')
        print(p)
    def write_data(self,path):
        print('tmp dataset saving')
        data_ = self.data
        data1 = data_.todense()
        data =data1.T
        #print(type(data))
        recon_x = pd.DataFrame(data, index=self.peaks, columns=self.barcode)
        recon_x.to_csv(os.path.join(path, 'tmp_data.txt'), sep='\t') 
        #with open(path+'\\forebrian.pkl', 'wb') as f:
        #    pickle.dump(data, f)
            
#        pd.Series(self.peaks).to_csv(path+'\\'+'peak.txt',sep='\t',index =False, header=None)
#        pd.Series(self.barcode).to_csv(path+'\\'+'barcode.txt',sep='\t',index =False, header=None)




def load_data(path, transpose=False):
    print("Loading  data ...")
    t0 = time.time()
    if os.path.isdir(path):
        count, peaks, barcode = read_mtx(path)
    elif os.path.isfile(path):
        count, peaks, barcode = read_csv(path)
    else:
        raise ValueError("File {} not exists".format(path))
        
    if transpose: 
        count = count.transpose()
    print('Original data contains {} cells x {} peaks'.format(*count.shape))
    assert (len(barcode), len(peaks)) == count.shape
    print("Finished loading takes {:.2f} min".format((time.time()-t0)/60))
    return count, peaks, barcode


def read_mtx(path):
    for filename in glob(path+'/*'):
        basename = os.path.basename(filename)
        if (('count' in basename) or ('matrix' in basename)) and ('mtx' in basename):
            count = mmread(filename).T.tocsr().astype('float32')
        elif 'barcode' in basename:
            barcode = pd.read_csv(filename, sep='\t', header=None)[0].values
        elif 'gene' in basename or 'peak' in basename:
            feature = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values

    return count, feature, barcode
    

def read_csv(path):
    if ('.txt' in path) or ('tsv' in path):
        sep = '\t'
    elif '.csv' in path:
        sep = ','
    else:
        raise ValueError("File {} not in format txt or csv".format(path))
    data = pd.read_csv(path, sep=sep, index_col=0).T.astype('float32')
    print('Read csv ...')
    genes = data.columns.values
    barcode = data.index.values
    counts = scipy.sparse.csr_matrix(data.values)
    print('ok')
    return counts, genes, barcode





# model 



class Encoder(nn.Module):
    def __init__(self,dims):
        super(Encoder, self).__init__()
        
        [x_dim, h_dim] = dims
 
        self.hidden = nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=h_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.hidden(x)
        return x 

class Decoder(nn.Module):
    def __init__(self, dims, output_activation=None):
        
        super(Decoder, self).__init__()

        [h_dim, x_dim] = dims
        
        self.hidden = nn.Sequential(
            nn.Linear(in_features=h_dim, out_features= x_dim),
            nn.Sigmoid()
        )
       
        #h_dimT = sorted(h_dim)

    def forward(self, x):
        x = self.hidden(x)
        return x

        
class AE(nn.Module):
    def __init__(self,dims): 
        super(AE, self).__init__()

        [x_dim, z_dim, encode_dim, decode_dim] = dims


        self.encoder = Encoder([x_dim, encode_dim])
        self.decoder = Decoder([encode_dim, x_dim])

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):        
        feature = self.encoder(x)
        recon_x = self.decoder(feature)
       

        
        return recon_x


    def KL_devergence(self,p, q):

        batch_size=32
        q = torch.nn.functional.softmax(q, dim=0)
        q = torch.sum(q, dim=0)/batch_size  
        s1 = torch.sum(p*torch.log(p/q))
        s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))
        return s1+s2
        
    
    def loss_func(self,x):
        feature = self.encoder(x)
        recon_x = self.decoder(feature)
        criteon = nn.MSELoss()
      
        hidden =32         
        loss = criteon(recon_x,x) 
        expect_tho =0.03
        tho_tensor = torch.FloatTensor([expect_tho for _ in range(hidden)])
        _beta = 3
        _kl = self.KL_devergence(tho_tensor, feature)
        loss += _beta * _kl

        return loss 
   


    def fit(self,dataloader,dims,outdir,lr = 0.001,epochs = 10000 ,max_iter = 10000):
        optimizer = optim.Adam(model.parameters(), lr=lr)        


    
        iteration =0 
        t0 = time.time()
        Loss = []
        early_stopping = EarlyStopping()
        with trange(max_iter, disable=False) as pbar: 
            while True: 
                epoch_loss = 0 
                for i,x in enumerate(dataloader):
                    epoch_lr = adjust_learning_rate(lr, optimizer, iteration)

                    
                    optimizer.zero_grad()                          
                    
                    loss = self.loss_func(x) 
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    
                    pbar.set_postfix_str('loss={:.3f}'.format(loss))
                    pbar.update(1)
                    #print(loss)
                               
                    iteration+=1
                    Loss.append(loss)
                    if iteration >= max_iter:
                        break
        
              
                else:

                    early_stopping(epoch_loss, self)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} iteration'.format(iteration))
                        break
                    continue
                break

     


    def encodeBatch(self, dataloader,out='z',transforms=None):
        output = []
        for i, inputs in enumerate(dataloader):
            x = inputs
            x = x.view(x.size(0), -1).float()
            
            feature = self.encoder(x)

            if out == 'z':
                output.append(feature.detach().cpu())  
                
            elif out == 'x':
                recon_x = self.decoder(feature)
                output.append(recon_x.detach().cpu().data)

        output = torch.cat(output).numpy()
        if out == 'x':
            for transform in transforms:
                output = transform(output)
        return output                





class sAE(AE):
    def __init__(self, dims, n_centroids):
        super(nmfAE, self).__init__(dims)
        self.n_centroids = n_centroids



def adjust_learning_rate(init_lr, optimizer, iteration):
    lr = max(init_lr * (0.9 ** (iteration//10)), 0.00002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr   



class EarlyStopping:
    def __init__(self, patience=100, verbose=False, outdir='./'):

        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt')

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print( path+'path create')
        return True
    else:
        print ('already exist')
        return False
  




# plot
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns


def plot_embedding(X, labels, classes=None, method='tSNE', cmap='tab20', figsize=(4, 4), markersize=4, marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=True, **legend_params):
    if marker is not None:
        X = np.concatenate([X, marker], axis=0)
    N = len(labels)
    if X.shape[1] != 2:
        if method == 'tSNE':
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, random_state=124).fit_transform(X)
        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=30, min_dist=0.1).fit_transform(X)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=124).fit_transform(X)
        
    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)

    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.color_palette(cmap, n_colors=len(classes))
        
    for i, c in enumerate(classes):
        plt.scatter(X[:N][labels==c, 0], X[:N][labels==c, 1], s=markersize, color=colors[i], label=c)
    if marker is not None:
        plt.scatter(X[N:, 0], X[N:, 1], s=10*markersize, color='black', marker='*')
#     plt.axis("off")
    
    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 10,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method+' dim 1', fontsize=12)
        plt.ylabel(method+' dim 2', fontsize=12)

    if save:
        plt.savefig(save, format='jpg', bbox_inches='tight')
    else:
        plt.show()
        
    if save_emb:
        np.savetxt(save_emb, X)
    if return_emb:
        return X










    
 
normalizer = MaxAbsScaler()
#class 

dataset = SingleCellDataset('%s'%(sys.argv[2]), low=0.01, high=0.9, min_peaks=100,
                             transforms=[normalizer.fit_transform])

trainloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
testloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

cell_num = dataset.shape[0] 
input_dim = dataset.shape[1] #x_dim


n_centroids = 8 
name = 'Forebrain'
z_dim = 8  
h_dim = int('%s'%(sys.argv[4]))
decode_dim = []
lr = 0.001
epochs = 9999
max_iter = int('%s'%(sys.argv[1]))

# make path
mkpath='%s'%(sys.argv[3])
mkdir(mkpath)
outdir = mkpath 
#
#model fit 
dims = [input_dim, z_dim, h_dim, decode_dim]
model = sAE(dims, n_centroids= n_centroids)
print('\n ###   Training……  ###') 
model.fit(trainloader,dims,lr=lr,epochs=epochs,max_iter=max_iter,outdir = outdir)

### output ###
print('outdir: {}'.format(outdir))

feature = model.encodeBatch(testloader,out='z')
pd.DataFrame(feature, index=dataset.barcode).to_csv(os.path.join(outdir, 'feature.txt'), sep='\t', header=False)
#

print("Plotting embedding")
reference = '%s'%(sys.argv[5])

emb = 'UMAP'
#emb = 'tSNE'

ref = pd.read_csv(reference, sep='\t', header=None, index_col=0)[1]
labels = ref.reindex(dataset.barcode, fill_value='unknown')



X= plot_embedding(feature, labels, method=emb, 
               save=os.path.join(outdir, 'emb_{}sae.jpg'.format(emb)), save_emb=os.path.join(outdir, 'emb_{}.txt'.format(emb)),return_emb = True)


