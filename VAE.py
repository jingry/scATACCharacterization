# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:00:20 2021

@author: huangyuyao
"""


import time
import os
import numpy as np
import pandas as pd
import scipy
from glob import glob
from scipy.io import mmread
from sklearn.preprocessing import LabelEncoder
import sys
import torch
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau

import time
import math
import numpy as np
from tqdm import trange
from itertools import repeat
from sklearn.mixture import GaussianMixture


class SingleCellDataset(Dataset):
    """
    Single-cell dataset
    """

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
    genes = data.columns.values
    barcode = data.index.values
    return scipy.sparse.csr_matrix(data.values), genes, barcode






class VAE(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, binary=True):


        super(VAE, self).__init__()
        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.binary = binary
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_activation = None

        self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout, output_activation=decode_activation)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):

        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        likelihood, kld = elbo(recon_x, x, (mu, logvar), binary=self.binary)

        return (-likelihood, kld)


    def predict(self, dataloader, device='cpu', method='kmeans'):



        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
        feature = self.encodeBatch(dataloader, device)
        kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
        pred = kmeans.fit_predict(feature)


        return pred

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, dataloader,
            lr=0.002, 
            weight_decay=5e-4,
            device='cpu',
            beta = 1,
            n = 200,
            max_iter=30000,
            verbose=True,
            name='',                                                                            
            patience=100,
            outdir='./'
       ):

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) 
        Beta = DeterministicWarmup(n=n, t_max=beta)
        
        iteration = 0
        early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        with trange(max_iter, disable=False) as pbar:
            while True: 
                epoch_loss = 0
                for i, x in enumerate(dataloader):
                    epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
                    t0 = time.time()
                    x = x.float().to(device)
                    optimizer.zero_grad()
                    
                    recon_loss, kl_loss = self.loss_function(x)
                    loss = (recon_loss + next(Beta) * kl_loss)/len(x);
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix_str('loss={:.3f} recon_loss={:.3f} kl_loss={:.3f}'.format(
                            loss, recon_loss/len(x), kl_loss/len(x)))
                    pbar.update(1)
                    
                    iteration+=1
                    if iteration >= max_iter:
                        break
                else:
                    early_stopping(epoch_loss, self)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} iteration'.format(iteration))
                        break
                    continue
                break

    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []

        for i, inputs in enumerate(dataloader):
            x = inputs
            x = x.view(x.size(0), -1).float().to(device)
            z,mu,logvar = self.encoder(x)
            #z, mu, logvar = tmp 
            
            
            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach())

        output = torch.cat(output).numpy()
        if out == 'x':
            for transform in transforms:
                output = transform(output)
        return output



class VVAE(VAE):
    def __init__(self, dims, n_centroids):
        super(SCALE, self).__init__(dims)
        self.n_centroids = n_centroids
        z_dim = dims[1]

        # init c_params
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids)) # sigma^2

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
        likelihood, kld = elbo_SCALE(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)

        return -likelihood, kld

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init SCALE model with GMM model parameters
        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        z = self.encodeBatch(dataloader, device)
        gmm.fit(z)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))


def adjust_learning_rate(init_lr, optimizer, iteration):
    lr = max(init_lr * (0.9 ** (iteration//10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr	


import os
class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir='./'):
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
#        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

import math
import numpy as np


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
    """

    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class Encoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0):

        super(Encoder, self).__init__()


        self.hidden = build_mlp([x_dim]+h_dim, bn=bn, dropout=dropout)
        self.sample = GaussianSample(([x_dim]+h_dim)[-1], z_dim)


    def forward(self, x):
        x = self.hidden(x);

        return self.sample(x)

class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, output_activation=nn.Sigmoid()):

        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)


        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden(x)
        if self.output_activation is not None:
            return self.output_activation(self.reconstruction(x))
        else:
            return self.reconstruction(x)

class DeterministicWarmup(object):

    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

    def next(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t



class Stochastic(nn.Module):

    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var


import torch
import torch.nn.functional as F

import math

def kl_divergence(mu, logvar):

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


def elbo(recon_x, x, z_params, binary=True):

    mu, logvar = z_params
    kld = kl_divergence(mu, logvar)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x)
    else:
        likelihood = -F.mse_loss(recon_x, x)
    return torch.sum(likelihood), torch.sum(kld)
    # return likelihood, kld


def elbo_SCALE(recon_x, x, gamma, c_params, z_params, binary=True):

    mu_c, var_c, pi = c_params; #print(mu_c.size(), var_c.size(), pi.size())
    n_centroids = pi.size(1)
    mu, logvar = z_params
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    # log p(x|z)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x) #;print(logvar_expand.size()) #, torch.exp(logvar_expand)/var_c)
    else:
        likelihood = -F.mse_loss(recon_x, x)

    # log p(z|c)
    logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c) + \
                                           torch.exp(logvar_expand)/var_c + \
                                           (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
    # log p(c)
    logpc = torch.sum(gamma*torch.log(pi), 1)

    # log q(z|x) or q entropy    
    qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(gamma*torch.log(gamma), 1)

    kld = -logpzc - logpc + qentropy + logqcx

    return torch.sum(likelihood), torch.sum(kld)




import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
# import os

# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['figure.dpi'] = 300

def sort_by_classes(X, y, classes):
    if classes is None:
        classes = np.unique(y)
    index = []
    for c in classes:
        ind = np.where(y==c)[0]
        index.append(ind)
    index = np.concatenate(index)
    X = X.iloc[:, index]
    y = y[index]
    return X, y, classes, index


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
        plt.savefig(save, format='jpg', bbox_inches='tight',dpi=150)
    else:
        plt.show()
        
    if save_emb:
        np.savetxt(save_emb, X)
    if return_emb:
        return X


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


   

import time
import torch

import numpy as np
import pandas as pd
import os
import argparse


from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader
 
 
normalizer = MaxAbsScaler()
#class 

dataset = SingleCellDataset('%s'%(sys.argv[2]), low=0.01, high=0.9, min_peaks=100,
                             transforms=[normalizer.fit_transform])

trainloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
testloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

cell_num = dataset.shape[0] 
input_dim = dataset.shape[1] #x_dim

# parameter 
n_centroids = 8
name = 'Forebrain'
z_dim = int('%s'%(sys.argv[4]))
h_dim = [1024, 128]
decode_dim = []
lr = 0.002
epochs = 9999
max_iter = int('%s'%(sys.argv[1]))

mkpath='%s'%(sys.argv[3])
mkdir(mkpath)
outdir = mkpath 
t = time.time()
#model fit 
dims = [input_dim, z_dim, h_dim, decode_dim]
model = VVAE(dims, n_centroids= n_centroids)
print('\n ###   Training……  ###') 
model.fit(trainloader,lr=lr,max_iter=max_iter,outdir = outdir)
#torch.save(model.to('cpu').state_dict(), os.path.join(outdir, 'model_tmp.pt')) 

### output ###
print('outdir: {}'.format(outdir))

#

feature = model.encodeBatch(testloader,out='z')
pd.DataFrame(feature, index=dataset.barcode).to_csv(os.path.join(outdir, 'feature.txt'), sep='\t', header=False)

t1  = time.time()

print("Plotting embedding")
reference = '%s'%(sys.argv[5])

emb = 'UMAP'
#emb = 'tSNE'
ref = pd.read_csv(reference, sep='\t', header=None, index_col=0)[1]
labels = ref.reindex(dataset.barcode, fill_value='unknown')

X= plot_embedding(feature, labels, method=emb, 
               save=os.path.join(outdir, 'emb_{}.jpg'.format(emb)), save_emb=os.path.join(outdir, 'emb_{}.txt'.format(emb)),return_emb = True)



