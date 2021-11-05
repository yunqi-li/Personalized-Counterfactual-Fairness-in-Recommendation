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

Python 3.8.5

```
numpy==1.20.1
torch==1.8
pandas==1.2.4
scipy==1.6.2
tqdm==4.32.1
scikit_learn==0.23.1
```

## Datasets

- The processed datasets are in  [`./dataset/ml1M`](https://github.com/yunqi-li/Personalized-Counterfactual-Fairness-in-Recommendation/tree/main/dataset/ml1M)
- After the first execution, two pickle files "ml1M.validation.pkl" and "ml1M.test.pkl" will be generated to save time for future runnings. The generated files will locate at the same directory as the other data files.

## Example to run the codes
-   To guarantee the program can execute properly, please keep the directory structure as given in this repository.
-   Some running commands can be found in [`./command/cmd.txt`](https://github.com/yunqi-li/Personalized-Counterfactual-Fairness-in-Recommendation/blob/main/command/cmd.txt)
-   For example:

```
> cd ./src/
> python main.py --model_name BiasedMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/biasedMF_ml1m_no_filter_neg_sample=100/biasedMF_ml1m_l2=1e-4_dim=64_no_filter_neg_sample=100.pt" --runner RecRunner --d_step 10 --vt_num_neg 100 --vt_batch_size 1024 --no_filter --eval_dict
```
