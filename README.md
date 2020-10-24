# second-order-neural-dmv

Source code of:  Second-Order Unsupervised Dependency Parsing

## Usage

### set up environment

conda create -n dmv python=3.7

conda activate dmv

pip install requirement.py

### set up directory

Please make sure the data folder looks like below.

```
config/
   ├── sib.yaml
   ├── lndmv.yaml
   ├── joint_lndmv_sib.yaml

data/
   ├── wsj10tr    
   ├── wsj10te
   ├── wsj10d
   ├── wsj-inf_2-21_dep_filter_10_init

embedding/
   ├── fast_text_wsj_100_1_300.model.trainables.vectors_ngrams_lockf.npy
   ├── fast_text_wsj_100_1_300.model
   ├── fast_text_wsj_100_1_300.model.wv.vectors_ngrams.npy

log/
fastNLP/
parser/
```

### set up pretrained embedding and dataset

100d FastText word embedding trained on WSJ corpus.  ( window size=3, train for 300 epochs): used in Lexcialzied NDMV. 

wsj-inf_2-21_dep_filter_10_init:  the predicted parse tree from Naseem's parser. We use this file for initialization.

 You can download the embedding and dataset from Google Drive link:   [download](https://drive.google.com/drive/folders/1Dn89gL28Gb2j7Fv6UdEVb4YdtA-fl5aO?usp=sharing)

#### Run

python run.py --mode train --conf config/xxxxx.yaml

### 

## Model

We have released the version of sibling NDMV and joint training of lexicalized DMV and sibling NDMV, trained in gradient descent fasion.

EM algorithm training and grand-parent NDMV variant is not updated yet

## Contact
If you have any question, contact yangsl@shanghaitech.edu.cn