# Parallel Universe Configuration (PuTransE)

Here, you can find the implementation of the (inductive) knowledge graph embedding technique Parallel Universe TransE (puTransE) proposed by _Yi Tay_, _Luu Anh Tuan_ and _Siu Cheung Hui_ in their paper _Non-Parametric Estimation of Multiple Embeddings for Link Prediction on Dynamic Knowledge Graphs_.

As illustrated in the animation below, PuTransE maps the entities and relations of a knowledge graph onto multiple vector spaces also referred to as universes. To learn these embedding spaces the transductive knowledge graph embedding technique TransE is applied. 
The local embedding spaces are then aggregated to predict the plausibility of an arbitrary triple in the represented knowledge graph at hand. An analogy can be drawn to random forests, where a prediction is delegated to multiple, randomly-created decision trees. 

<br>

<p align="center">
   <img src="animations/PuTransE_concept.gif" height="60%" width="60%">
</p>   



In this way, PuTransE enables inductive knowledge graph embedding (also referred to as online-learning) as illustrated in the animation below.
Starting from the initial state of a knowledge graph, PuTransE produces multiple randomly-configured embedding spaces. In case the graph is updated afterwards, based on the new graph's state new embedding spaces are learned which are then aggregated with previously learned ones in order to calculate triple predictions.

<br>

<p align="center">
   <img src="animations/PuTransE_inductive_strategy.gif" height="60%" width="60%">
</p>   


## Usage

Similar to the approach of the authors to embed TransE into the setting of parallel universes to enable inductive knowledge graph embedding, basically any other knowledge graph embedding technique can be used that is based on translations in the vector space. Accordingly, I implemented the configuration of parallel universes (see [`Parallel_Universe_Config.py`](https://github.com/rlafraie/OpenKE-PuTransE/blob/OpenKE-PyTorch/openke/config/Parallel_Universe_Config.py) and integrated it as a module within the OpenKE framework which provides a variety of readily-made embedding models. As a result, parallel universe are not only combinable with TransE but moreover with multiple other embedding techniques supplied by OpenKE. 

These include:
- TransH
- TransD 
- TransR
- DistMult


The files [`static_experiment_PuTransE_on_WN18.py`](https://github.com/rlafraie/OpenKE-PuTransE/blob/OpenKE-PyTorch/experiments/static_experiment_PuTransE_on_WN18.py), [`static_experiment_PuTransH_on_WN18.py`](https://github.com/rlafraie/OpenKE-PuTransE/blob/OpenKE-PyTorch/experiments/static_experiment_PuTransH_on_WN18.py), [`static_experiment_PuTransD_on_WN18.py`](https://github.com/rlafraie/OpenKE-PuTransE/blob/OpenKE-PyTorch/experiments/static_experiment_PuTransD_on_WN18.py) show example usages of the parallel universe configuration. Here, I encapsulated the OpenKE instances of TransE, TransH and TransD to form the inductive relatives of PuTransE, PuTransH and PuTransD and evaluated them on the WN18 benchmark. 

Further, the file [`incremental_experiment_PuTransE_on_WikidataEvolve.py`](https://github.com/rlafraie/OpenKE-PuTransE/blob/OpenKE-PyTorch/experiments/incremental_experiment_PuTransE_on_WikidataEvolve.py) manifests an experiment conducted during my master's thesis where I applied and evaluated PuTransE by performing online embedding learning on the WikidataEvolve benchmark.

<br>
<h1 align="center"> README.md from forked repository </h1>

<br> 

<br>

# OpenKE-PyTorch

An Open-source Framework for Knowledge Embedding implemented with PyTorch.

More information is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)

If you use the code, please cite the following [paper](http://aclweb.org/anthology/D18-2024):

```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```

This package is mainly contributed (in chronological order) by [Xu Han](https://github.com/THUCSTHanxu13), [Yankai Lin](https://github.com/Mrlyk423), [Ruobing Xie](http://nlp.csai.tsinghua.edu.cn/~xrb/), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/), [Xin Lv](https://github.com/davidlvxin), [Shulin Cao](https://github.com/ShulinCao), [Weize Chen](https://github.com/chenweize1998), [Jingqin Yang](https://github.com/yjqqqaq).

## Overview

This is an Efficient implementation based on PyTorch for knowledge representation learning (KRL). We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by PyTorch with Python interfaces so that there is a convenient platform to run models on GPUs. OpenKE composes 4 repositories:

OpenKE-PyTorch: the project based on PyTorch, which provides the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0"> OpenKE-Tensorflow1.0</a>: OpenKE implemented with TensorFlow, also providing the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD. 

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE. 


*** **UPDATE** ***

We are now developing a new version of OpenKE-PyTorch. The project has been completely reconstructed and is faster, more extendable and the codes are easier to read and use now. If you need get to the old version, please refer to branch [OpenKE-PyTorch(old)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)).

*** **New Features** ***

- RotatE
- More enhancing strategies (e.g., adversarial training)
- More scripts of the typical models for the benchmark datasets.
- More extendable interfaces


## Models

OpenKE (Tensorflow): 

*	RESCAL, HolE
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD

OpenKE (PyTorch): 

*	RESCAL
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD
*  SimplE
*	RotatE

We welcome any issues and requests for model implementation and bug fix.

## Experimental Settings

For each test triplet, the head is removed and replaced by each of the entities from the entity set in turn. The scores of those corrupted triplets are first computed by the models and then sorted by the order. Then, we get the rank of the correct entity. This whole procedure is also repeated by removing those tail entities. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10, Hits@3, Hits@1). The mean rank (MRR) and mean reciprocal rank (MRR) of the test triplets under this setting are also reported.

Because some corrupted triplets may be in the training set and validation set. In this case, those corrupted triplets may be ranked above the test triplet, but this should not be counted as an error because both triplets are true. Hence, we remove those corrupted triplets appearing in the training, validation or test set, which ensures the corrupted triplets are not in the dataset. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10 (filter), Hits@3(filter), Hits@1(filter)) under this setting. The mean rank (MRR (filter)) and mean reciprocal rank (MRR (filter)) of the test triplets under this setting are also reported.

More details of the above-mentioned settings can be found from the papers [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

For those large-scale entity sets, to corrupt all entities with the whole entity set is time-costing. Hence, we also provide the experimental setting named "[type constraint](https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf)" to corrupt entities with some limited entity sets determining by their relations.

## Experiments

We have provided the hyper-parameters of some models to achieve the state-of-the-art performace (Hits@10 (filter)) on FB15K237 and WN18RR. These scripts can be founded in the folder "./examples/". Up to now, these models include TransE, TransH, TransR, TransD, DistMult, ComplEx. The results of these models are as follows,

|Model			|	WN18RR	|	FB15K237	| WN18RR (Paper\*)| FB15K237  (Paper\*)|
|:-:		|:-:	|:-:  |:-:  |:-:  |
|TransE	|0.512	|0.476|0.501|0.486|
|TransH	|0.507	|0.490|-|-|
|TransR	|0.519	|0.511|-|-|
|TransD	|0.508	|0.487|-|-|
|DistMult	|0.479	|0.419|0.49|0.419|
|ComplEx	|0.485	|0.426|0.51|0.428|
|ConvE		|0.506	|0.485|0.52|0.501|
|RotatE	|0.549	|0.479|-|0.480|
|RotatE (+adv)	|0.565	|0.522|0.571|0.533|


<strong> We are still trying more hyper-parameters and more training strategies (e.g., adversarial training and label smoothing regularization) for these models. </strong> Hence, this table is still in change. We welcome everyone to help us update this table and hyper-parameters.


## Installation

1. Install PyTorch

2. Clone the OpenKE-PyTorch branch:

	$ git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE
	
	$ cd OpenKE
	
	$ cd openke

3. Compile C++ files
	
	$ bash make.sh
	
4. Quick Start

	$ cd ../
	
	$ cp examples/train_transe_FB15K237.py ./
	
	$ python train_transe_FB15K237.py

## Data

* For training, datasets contain three files:

  train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** .
  **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**

  entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

* For testing, datasets contain additional two files (totally five files):

  test2id.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(e1, e2, rel)*** .

  valid2id.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(e1, e2, rel)*** .

  type_constrain.txt: type constraining file, the first line is the number of relations. Then the following lines are type constraints for each relation. For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733. The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088. You can get this file through **n-n.py** in folder benchmarks/FB15K
  
## To do

The document of the new version of OpenKE-PyTorch will come soon.
