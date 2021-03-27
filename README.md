# second-order-neural-dmv

Source code of COLING2020:  Second-Order Unsupervised Neural Dependency Parsing




## Usage

### set up environment

conda create -n dmv python=3.7

conda activate dmv

pip install -r requirements.txt

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

## Citation
If this repository helps your research, please cite our paper:
```
@inproceedings{yang-etal-2020-second,
    title = "Second-Order Unsupervised Neural Dependency Parsing",
    author = "Yang, Songlin  and
      Jiang, Yong  and
      Han, Wenjuan  and
      Tu, Kewei",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.347",
    doi = "10.18653/v1/2020.coling-main.347",
    pages = "3911--3924",
    abstract = "Most of the unsupervised dependency parsers are based on first-order probabilistic generative models that only consider local parent-child information. Inspired by second-order supervised dependency parsing, we proposed a second-order extension of unsupervised neural dependency models that incorporate grandparent-child or sibling information. We also propose a novel design of the neural parameterization and optimization methods of the dependency models. In second-order models, the number of grammar rules grows cubically with the increase of vocabulary size, making it difficult to train lexicalized models that may contain thousands of words. To circumvent this problem while still benefiting from both second-order parsing and lexicalization, we use the agreement-based learning framework to jointly train a second-order unlexicalized model and a first-order lexicalized model. Experiments on multiple datasets show the effectiveness of our second-order models compared with recent state-of-the-art methods. Our joint model achieves a 10{\%} improvement over the previous state-of-the-art parser on the full WSJ test set.",
}
```
