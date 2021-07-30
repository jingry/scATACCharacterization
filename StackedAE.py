# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:12:32 2021

@author: huangyuyao
"""



import torch
from torch import nn, optim, functional, utils
import torchvision
from torchvision import datasets, utils

import time, os
from torch.utils.data import Dataset
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy
from glob import glob
from scipy.io import mmread
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../')
from torch.nn import init
import pickle
from torch.nn import BCELoss




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

        total_cells = self.data.shape[0]
        count = np.array((self.data >0).sum(0)).squeeze()
#         indices = np.where(count > 0.01*X*total_cells)[0] 
        indices = np.where((count > low*total_cells) & (count < high*total_cells))[0] 
        self.data = self.data[:, indices]
        self.peaks = self.peaks[indices]
        print('filterpeak------')
        
    def filter_cell(self, min_peaks=0):

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


class AutoEncoderLayer(nn.Module):


    def __init__(self, input_dim=None, output_dim=None, SelfTraining=False):
        super(AutoEncoderLayer, self).__init__()

        self.in_features = input_dim
        self.out_features = output_dim
        self.is_training_self = SelfTraining  
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=True),

            nn.ReLU()
        )
        self.decoder = nn.Sequential( 
            nn.Linear(self.out_features, self.in_features, bias=True),
            nn.Sigmoid()

        )

    def forward(self, x):
        out = self.encoder(x)
        if self.is_training_self:
            return self.decoder(out)
        else:
            return out

    def lock_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    @property
    def is_training_layer(self):
        return self.is_training_self

    @is_training_layer.setter
    def is_training_layer(self, other: bool):
        self.is_training_self = other


 


class StackedAutoEncoder(nn.Module):

    def __init__(self, layers_list=None):
        super(StackedAutoEncoder, self).__init__()
        self.layers_list = layers_list
        self.initialize()
        self.encoder_1 = self.layers_list[0]
        self.encoder_2 = self.layers_list[1]
        self.encoder_3 = self.layers_list[2]
        self.encoder_4 = self.layers_list[3]
        self.encoder_5 = self.layers_list[4]
        self.encoder_6 = self.layers_list[5]
        self.encoder_7 = self.layers_list[6]
        self.encoder_8 = self.layers_list[7]
        
    def initialize(self):
        for layer in self.layers_list:
            # assert isinstance(layer, AutoEncoderLayer)
            layer.is_training_layer = False


    def forward(self, x):
        out = x

        out = self.encoder_1(out)
        out = self.encoder_2(out)
        out = self.encoder_3(out)
        out = self.encoder_4(out)
        out2= out
        self.out= out2
        out = self.encoder_5(out)
        out = self.encoder_6(out)
        out = self.encoder_7(out)
        out = self.encoder_8(out)
        
        return out2,out

    
    


def train_layers(trainloader,layer,layers_list, epoch, validate=True):

    train_loader = trainloader


    optimizer = optim.SGD(layers_list[layer].parameters(), lr=0.01)

    criterion = torch.nn.MSELoss()


    for epoch_index in range(epoch):
        sum_loss = 0.


        if layer != 0:
            for index in range(layer):
                layers_list[index].lock_grad()
                layers_list[index].is_training_layer = False  

        for batch_index, train_data in enumerate(train_loader):

            if torch.cuda.is_available():
                train_data = train_data.cuda()  
            out = train_data.view(train_data.size(0), -1)


            if layer != 0:
                for l in range(layer):
                    out = layers_list[l](out)


            pred = layers_list[layer](out)

            optimizer.zero_grad()
            loss = criterion(pred, out)
            sum_loss += loss
            loss.backward()
            optimizer.step()
            if (batch_index + 1) % 10 == 0:
                print("Train Layer: {}, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                    layer, (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), loss
                ))

        if validate:
            pass

    
def train_whole(train_loader,test_loader,model=None, epoch=50, validate=True):
    #output=[]
    print(">> start training whole model")
    if torch.cuda.is_available():
        model.cuda()

    for param in model.parameters():
        param.require_grad = True


    optimizer = optim.SGD(model.parameters(), lr=0.01)

    criterion = torch.nn.MSELoss()


    test_data = next(iter(test_loader))

    for epoch_index in range(epoch):
        sum_loss = 0.
        output=[]
        for batch_index, train_data in enumerate(train_loader):
            if torch.cuda.is_available():
                train_data = train_data.cuda()
            x = train_data.view(train_data.size(0), -1)

            out = model(x)[1]
            out_ = model(x)[0] # 0是中间层

            output.append(out_)

            optimizer.zero_grad()
            loss = criterion(out, x)
            sum_loss += loss
            loss.backward()
            optimizer.step()

            if (batch_index + 1) % 10 == 0:
                print("Train Whole, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                    (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), loss
                ))

        
        if validate:
            x = test_data.view(test_data.size(0), -1)

            out = model(x)[1]

            
            
            loss = criterion(out, x)
            print("Test Epoch: {}/{}, Iter: {}/{}, test Loss: {}".format(
                epoch_index + 1, epoch, (epoch_index + 1), len(test_loader), loss
            ))



    output = torch.cat(output).numpy()
    print("<< end training whole model")
    return output




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
        plt.savefig(save, format='jpg', bbox_inches='tight',dpi = 300)
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
  

   
num_tranin_layer_epochs = int('%s'%(sys.argv[1]))
num_tranin_whole_epochs = int('%s'%(sys.argv[2]))
shuffle = False 



normalizer = MaxAbsScaler()
#class 

dataset = SingleCellDataset('%s'%(sys.argv[3]), low=0.01, high=0.9, min_peaks=100,
                             transforms=[normalizer.fit_transform])

trainloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
testloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)



nun_layers = 9
latent = int('%s'%(sys.argv[5]))
encoder_1 = AutoEncoderLayer(11285,8192, SelfTraining=True)
encoder_2 = AutoEncoderLayer(8192, 2048, SelfTraining=True)
encoder_3 = AutoEncoderLayer(2048, 256, SelfTraining=True)
encoder_4 = AutoEncoderLayer(256, latent, SelfTraining=True)
encoder_5 = AutoEncoderLayer(latent, 256, SelfTraining=True)
encoder_6 = AutoEncoderLayer(256, 2048, SelfTraining=True)
encoder_7 = AutoEncoderLayer(2048, 8192, SelfTraining=True)
encoder_8 = AutoEncoderLayer(8192, 11285, SelfTraining=True)
layers_list = [encoder_1, encoder_2, encoder_3, encoder_4,encoder_5,encoder_6,encoder_7,encoder_8]



# 按照顺序对每一层进行预训练
for level in range(nun_layers - 1):
    train_layers(trainloader,layers_list=layers_list, layer=level, epoch=num_tranin_layer_epochs, validate=True)

# 统一训练
SAE_model = StackedAutoEncoder(layers_list=layers_list)
output = train_whole(trainloader,testloader,model=SAE_model, epoch=num_tranin_whole_epochs, validate=True)
#feature = output

mkpath='%s'%(sys.argv[4])
mkdir(mkpath)
outdir = mkpath 


output1=[]
tmp1=[]

for i, data in enumerate(testloader):
    x = data.view(data.size(0), -1)
    out = SAE_model(x)[0]
    tmp = SAE_model.out
    tmp1.append(tmp)
    output1.append(out)
output1 = torch.cat(output1).numpy()
tmp1= torch.cat(tmp1).numpy() 
#
#
feature = pd.DataFrame(output1, index= dataset.barcode)
feature.to_csv(os.path.join(outdir, 'feature.txt'), sep='\t', header=False)




print("Plotting embedding")

reference = '%s'%(sys.argv[6])
#reference = None 
emb = 'UMAP'
#emb = 'tSNE'
ref = pd.read_csv(reference, sep='\t', header=None, index_col=0)[1]
labels = ref.reindex(dataset.barcode, fill_value='unknown')

X= plot_embedding(output1, labels, method=emb, 
               save=os.path.join(outdir, 'emb_{}.jpg'.format(emb)), save_emb=os.path.join(outdir, 'emb_{}.txt'.format(emb)),return_emb = True)
