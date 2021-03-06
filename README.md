# scATACCharacterization  

<h1>A multiple comprehensive analysis of scATAC-seq based on auto-encoder and matrix decomposition  </h1>

  
  
  
  
<h2>Author  </h2>

  

2019322030012@stu.scu.edu.cn  

jingry@scu.edu.cn  

  

  

<h2>overview  </h2>

  

We performed a multiple comparison for characterizing analyzing scATAC-seq based on four kinds of auto-encoder neural networks, and two kinds of matrix factorization methods. The autoencoder neural networks are implemented in [Pytorch](https://pytorch.org/) framework. The NMF and Lsnmf is implemented in jupyter.  

  

  

  

<h2>Datasets  </h2>
  

The dataset is from the work of “SCALE method for single-cell ATAC-seq analysis via latent feature extraction”, and the related web site is: https://github.com/jsxlei/SCALE   

  

  

<h2>Usage:  </h2>

When training autoencoders，the user can enter the following commands  

  

<h3>1. General autoencoder   </h3>

  

python AE.py \<EpochNum\> \<DataPath\> \<Outdir\> \<LatentFeatureNum\> \<LabelPath\>  

  

EpochNum: Integer, the epoches for training.  

DataPath: String of path, the input file of the dataset.  

Outdir: String of path, the folder for saving the outputs.  

LatentFeatureNum: Integer, the size of the latent layer.  

LabelPath: String of path, the label file for UMAP plotting.  

  

  

<h3>2. Sparse auotoencoder  </h3>

  

python SparseAE.py \<EpochNum\> \<DataPath\> \<Outdir\> \<LatentFeatureNum\> \<LabelPath\>  

  

EpochNum: Integer, the epoches for training.  

DataPath: String of path, the input file of the dataset.  

Outdir: String of path, the folder for saving the outputs.  

LatentFeatureNum: Integer, the size of the latent layer.  

LabelPath: String of path, the label file for UMAP plotting.  

  

  

<h3>3. Stacked autoencoder   </h3>

  

python StackedAE.py \<preTrainEpochNum\> \<wholeTrainEpochNum\> \<DataPath\> \<Outdir\> \<LatentFeatureNum\> \<LabelPath\>  

  

preTrainEpochNum: Integer, the epoches for pre-training.  

wholeTrainEpochNum: Integer, the epoches for whole-training.  

DataPath: String of path, the input file of the dataset.  

Outdir: String of path, the folder for saving the outputs.  

LatentFeatureNum: Integer, the size of the latent layer.  

LabelPath: String of path, the label file for UMAP plotting.  

  

  

<h3>4. VAE  </h3>

  

python VAE.py \<EpochNum\> \<DataPath\> \<Outdir\> \<LatentFeatureNum\> \<LabelPath\>  

  

EpochNum: Integer, the epoches for training.  

DataPath: String of path, the input file of the dataset.  

Outdir: String of path, the folder for saving the outputs.  

LatentFeatureNum: Integer, the size of the latent layer.  

LabelPath: String of path, the label file for UMAP plotting.  

  

  

<h3>5.NMF & Lsnmf  </h3>

See code(NMF_Lsnmf.ipynb) for details  

