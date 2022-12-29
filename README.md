# HGV4Risk: Hierarchical Global View-guided Sequence Representation Learning for Risk Prediction

## Overview
This repository is the implementation of HGV ([arXiv](https://arxiv.org/abs/2211.07956))
> Youru Li, Zhenfeng Zhu, Xiaobo Guo, Shaoshuai Li, Yuchen Yang, Yao Zhao: HGV4Risk: Hierarchical Global Views-guided Sequential Representation Learning for Risk Prediction, arXiv preprint arXiv:2211.07956, 2022.

![](https://github.com/LiYouru0228/HGV/blob/main/HGV.png?raw=true)
This is a graphical illustration of hierarchical global views-guided sequential representation learning for risk prediction. 


## Preliminaries

### How to download the benchmark dataset:
Follow the rule of MIMIC-III data administrator, we have no right to release the dataset directly, so you need to acquire the data by yourself from https://mimic.physionet.org/ with the guidance at https://mimic.mit.edu/docs/gettingstarted/. 

### How to build the benchmark task:
When you download the CSVs data successfully, you can build the in-hospital mortality benchmark task by derectly runing the following commands given in https://github.com/YerevaNN/mimic3-benchmarks/:
```
$ python -m mimic3benchmark.scripts.extract_subjects {YOUR PATH TO SAVE THE DOWNLOADED CSVs} data/root/
$ python -m mimic3benchmark.scripts.validate_events data/root/
$ python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
$ python -m mimic3benchmark.scripts.split_train_and_test data/root/
$ python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
```
After the above commands are done, there will be a directory data/in-hospital-mortality for the benchmark task and two sub-directories: train and test are created in this directory as well. Moreover, the split_index file for train/val/test are also created. Noted, you need to put these into the directory of "./data/" created by this repository. 

### Required packages:
The code has been tested running under Python 3.8.3, and some main following packages installed and their version are:
- PyTorch == 1.0.1
- numpy == 1.18.5
- scipy == 1.5.4
- scikit-learn == 0.19.1

## Running the code
Firstly, you can run "load_data.py" to finish the data preprocessing and this command can save the preprocessed data into some pickel files. Therefore, you only need to run it the first time.

```
$ python load_data.py
```
Then, you can start to train the model and evaluate the performance by run:
```
$ python train.py

```
## Citation 
If you want to use our codes in your research, please cite:
```
@article{DBLP:journals/corr/abs-2211-07956,
  author    = {Youru Li and
               Zhenfeng Zhu and
               Xiaobo Guo and
               Shaoshuai Li and
               Yuchen Yang and
               Yao Zhao},
  title     = {HGV4Risk: Hierarchical Global View-guided Sequence Representation
               Learning for Risk Prediction},
  journal   = {CoRR},
  volume    = {abs/2211.07956},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2211.07956},
  doi       = {10.48550/arXiv.2211.07956},
  eprinttype = {arXiv},
  eprint    = {2211.07956},
  timestamp = {Wed, 23 Nov 2022 18:03:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2211-07956.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgments
Thanks to these open source benchmark projects https://github.com/YerevaNN/mimic3-benchmarks/, https://github.com/choczhang/ConCare and https://github.com/choczhang/GRASP whose code has good reusability and easy to followed and the basic pipeline of this project can quickly complete with their help.
