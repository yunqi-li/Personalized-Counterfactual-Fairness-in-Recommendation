# Personalized-Counterfactual-Fairness-in-Recommendation
This repository includes the implementation for paper Personalized Counterfactual Fairness in Recommendation (a.k.a. Towards Personalized Fairness based on Causal Notion):

*Yunqi Li, Hanxiong Chen, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang. 2021. [Personalized Counterfactual Fairness in Recommendation](https://arxiv.org/pdf/2105.09829.pdf). 
In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information RetrievalJuly 2021 (SIGIR’21)*

## Refernece

For inquiries contact Yunqi Li (yunqi.li@rutgers.edu) or Hanxiong Chen (hanxiong.chen@rutgers.edu)

```
@inproceedings{li2021towards,
author = {Li, Yunqi and Chen, Hanxiong and Xu, Shuyuan and Ge, Yingqiang and Zhang, Yongfeng},
title = {Towards Personalized Fairness Based on Causal Notion},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3462966},
doi = {10.1145/3404835.3462966},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1054–1063},
numpages = {10},
keywords = {counterfactual fairness, personalized fairness, adversary learning, recommender system},
location = {Virtual Event, Canada},
series = {SIGIR '21}
}
```

## Environments

Python 3.6.6

Packages: See in [requirements.txt](https://github.com/rutgerswiselab/NCR/blob/master/requirements.txt)

```
numpy==1.18.1
torch==1.0.1
pandas==0.24.2
scipy==1.3.0
tqdm==4.32.1
scikit_learn==0.23.1
```

## Datasets

- The processed datasets are in  [`./dataset/`](https://github.com/rutgerswiselab/NCR/tree/master/dataset)

- **ML-100k**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/100k/). 

- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 
    

## Example to run the codes
-   To guarantee the program can execute properly, please keep the directory structure as given in this repository.
-   Some running commands can be found in [`./command/command.py`](https://github.com/rutgerswiselab/NCR/blob/master/command/command.py)
-   For example:

```
# Neural Collaborative Reasong on ML-100k dataset
> cd NCR/src/
> python main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 2022 --gpu 0
```
