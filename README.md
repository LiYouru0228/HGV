# HGV4Risk: Hierarchical Global View-guided Sequence Representation Learning for Risk Prediction

## Overview
This repository is the implementation of HGV4Risk ([TKDD](https://dl.acm.org/doi/full/10.1145/3605895)|[arXiv](https://arxiv.org/abs/2211.07956))
> Youru Li, Zhenfeng Zhu, Xiaobo Guo, Shaoshuai Li, Yuchen Yang, Yao Zhao: HGV4Risk: Hierarchical Global Views-guided Sequence Representation Learning for Risk Prediction. ACM Transactions on Knowledge Discovery from Data, 18(1), 1-21.


![](https://github.com/LiYouru0228/HGV/blob/main/HGV4Risk.png?raw=true)
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
@article{li2023hgv4risk,
  title={HGV4Risk: Hierarchical Global View-guided Sequence Representation Learning for Risk Prediction},
  author={Li, Youru and Zhu, Zhenfeng and Guo, Xiaobo and Li, Shaoshuai and Yang, Yuchen and Zhao, Yao},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={18},
  number={1},
  pages={1--21},
  year={2023},
  publisher={ACM New York, NY}
}
```

## Acknowledgments
Thanks to these open source benchmark projects https://github.com/YerevaNN/mimic3-benchmarks/, https://github.com/choczhang/ConCare and https://github.com/choczhang/GRASP whose code has good reusability and easy to followed and the basic pipeline of this project can quickly complete with their help.
