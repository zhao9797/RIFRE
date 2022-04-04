# RIFRE
Pytorch implementation for codes in Representation Iterative Fusion Based on Heterogeneous Graph Neural Network for Joint Entity and Relation Extraction

## Model
![RIFRE framework](https://github.com/zhao9797/RIFRE/blob/main/model.png)

## requirements

* python 3.7
* torch  1.3
* tqdm
* transformers
* numpy

### Clone and load BERT pretrained models
```
git clone https://github.com/zhao9797/RIFRE.git
mkdir RIFRE/datasets/bert
cd RIFRE/datasets/bert
sudo apt-get install git-lfs

## provide path of pretrained models
git clone https://huggingface.co/bert-base-cased
git clone https://huggingface.co/bert-base-uncased

cd bert-base-cased
git lfs pull
cd ..
cd bert-base-uncased
git lfs pull

```

### Run the Code
```
python train.py
```

## Citation
```
@article{ZHAO2021106888,
title = {Representation iterative fusion based on heterogeneous graph neural network for joint entity and relation extraction},
journal = {Knowledge-Based Systems},
pages = {106888},
year = {2021},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2021.106888},
url = {https://www.sciencedirect.com/science/article/pii/S0950705121001519},
author = {Kang Zhao and Hua Xu and Yue Cheng and Xiaoteng Li and Kai Gao}
}
```
