# scATACCharacterization<br>
A multiple comprehensive analysis of scATAC-seq based on auto-encoder and matrix decomposition<br>
###########################<br>
<br>
Author<br>
###########################<br>
2019322030012@scu.edu.cn<br>
jingry@scu.edu.cn<br>
<br>
<br>
overview<br>
###########################<br>
We performed a multiple comparison for characterizing analyzing scATAC-seq based on four kinds of auto-encoder neural networks, and two kinds of matrix factorization methods. The autoencoder neural networks are implemented in [Pytorch](https://pytorch.org/) framework. The NMF and Lsnmf is implemented in jupyter.<br>
<br>
<br>
<br>
Datasets<br>
###########################<br>
The dataset is from the work of “SCALE method for single-cell ATAC-seq analysis via latent feature extraction”, and the related web site is: https://github.com/jsxlei/SCALE <br>
<br>
<br>
Usage:<br>
When training autoencoders，the user can enter the following commands<br>
############################<br>
1. General autoencoder <br>
<br>
```python AE.py <EpochNum> <DataPath> <Outdir> <LatentFeatureNum> <LabelPath>```<br>
<br>
EpochNum: Integer, the epoches for training.<br>
DataPath: String of path, the input file of the dataset.<br>
Outdir: String of path, the folder for saving the outputs.<br>
LatentFeatureNum: Integer, the size of the latent layer.<br>
LabelPath: String of path, the label file for UMAP plotting.<br>
<br>
###########################<br>
2. Sparse auotoencoder<br>
<br>
```python SparseAE.py <EpochNum> <DataPath> <Outdir> <LatentFeatureNum> <LabelPath>```<br>
<br>
EpochNum: Integer, the epoches for training.<br>
DataPath: String of path, the input file of the dataset.<br>
Outdir: String of path, the folder for saving the outputs.<br>
LatentFeatureNum: Integer, the size of the latent layer.<br>
LabelPath: String of path, the label file for UMAP plotting.<br>
<br>
###########################<br>
3. Stacked autoencoder <br>
<br>
```python StackedAE.py <preTrainEpochNum> <wholeTrainEpochNum> <DataPath> <Outdir> <LatentFeatureNum> <LabelPath>```<br>
<br>
preTrainEpochNum: Integer, the epoches for pre-training.<br>
wholeTrainEpochNum: Integer, the epoches for whole-training.<br>
DataPath: String of path, the input file of the dataset.<br>
Outdir: String of path, the folder for saving the outputs.<br>
LatentFeatureNum: Integer, the size of the latent layer.<br>
LabelPath: String of path, the label file for UMAP plotting.<br>
<br>
###########################<br>
4. VAE<br>
<br>
```python VAE.py <EpochNum> <DataPath> <Outdir> <LatentFeatureNum> <LabelPath>```<br>
<br>
EpochNum: Integer, the epoches for training.<br>
DataPath: String of path, the input file of the dataset.<br>
Outdir: String of path, the folder for saving the outputs.<br>
LatentFeatureNum: Integer, the size of the latent layer.<br>
LabelPath: String of path, the label file for UMAP plotting.<br>
<br>
###########################<br>
5.NMF & Lsnmf<br>
See code(NMF_Lsnmf.ipynb) for details<br>
