import numpy as np
from scipy import spatial
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import copy
from tqdm import tqdm
from utils import utils
from utils.readers import InHospitalMortalityReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils
import torch
from torch.utils import data

data_path = './data/'
file_name = './data/best_model2saved.pth'
small_part = False
arg_timestep = 1.0

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='what size of a batch')

train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                         listfile=os.path.join(data_path, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                       listfile=os.path.join(data_path, 'val_listfile.csv'),
                                       period_length=48.0)

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'test'),
                                            listfile=os.path.join(data_path, 'test_listfile.csv'),
                                            period_length=48.0)

discretizer = Discretizer(timestep=arg_timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

class Dataset(data.Dataset):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.name[index]

    def __len__(self):
        return len(self.x)

def data_load(args):
    batch_size = args.batch_size
    
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  
    normalizer_state = 'ihm_normalizer'
    normalizer_state = os.path.join(os.path.dirname(data_path), normalizer_state)
    normalizer.load_params(normalizer_state)

    n_trained_chunks = 0
    train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part, return_names=True)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, small_part, return_names=True)
    test_raw = utils.load_data(test_reader, discretizer, normalizer, small_part, return_names=True)

    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,valid_loader,test_loader,train_raw,val_raw,test_raw

def load_demo():
    demographic_data = []
    diagnosis_data = []
    idx_list = []

    demo_path = data_path + 'demographic/'
    for cur_name in os.listdir(demo_path):
        cur_id, cur_episode = cur_name.split('_', 1)
        cur_episode = cur_episode[:-4]
        cur_file = demo_path + cur_name

        with open(cur_file, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            if header[0] != "Icustay":
                continue
            cur_data = tsfile.readline().strip().split(',')
            
        if len(cur_data) == 1:
            cur_demo = np.zeros(12)
            cur_diag = np.zeros(128)
        else:
            if cur_data[3] == '':
                cur_data[3] = 60.0
            if cur_data[4] == '':
                cur_data[4] = 160
            if cur_data[5] == '':
                cur_data[5] = 70

            cur_demo = np.zeros(12)
            cur_demo[int(cur_data[1])] = 1
            cur_demo[5 + int(cur_data[2])] = 1
            cur_demo[9:] = cur_data[3:6]
            cur_diag = np.array(cur_data[8:], dtype=np.int)

        demographic_data.append(cur_demo)
        diagnosis_data.append(cur_diag)
        idx_list.append(cur_id+'_'+cur_episode)

    for each_idx in range(9,12):
        cur_val = []
        for i in range(len(demographic_data)):
            cur_val.append(demographic_data[i][each_idx])
        cur_val = np.array(cur_val)
        _mean = np.mean(cur_val)
        _std = np.std(cur_val)
        _std = _std if _std > 1e-7 else 1e-7
        for i in range(len(demographic_data)):
            demographic_data[i][each_idx] = (demographic_data[i][each_idx] - _mean) / _std

    return [demographic_data,idx_list]


def build_graph(train_raw, val_raw, test_raw):
    file_vec,file2sim = {},{}
    file_names = train_raw['names'] + val_raw['names'] + test_raw['names']

    for file in tqdm(file_names):
        if file in train_raw['names']:
            file_vec[file] = ([train_raw['data'][0][train_raw['names'].index(file)][i:i+1,:] for i in range(48)])
        elif file in val_raw['names']:
            file_vec[file] = ([val_raw['data'][0][val_raw['names'].index(file)][i:i+1,:] for i in range(48)])
        else:
            file_vec[file] = ([test_raw['data'][0][test_raw['names'].index(file)][i:i+1,:] for i in range(48)])

    for file in tqdm(file_vec):
        sim = np.random.random(size=(48,48))
        for i in range(len(file_vec[file])):
            for j in range(len(file_vec[file])):
                if i == j:
                    sim[i][j] = 0
                else:
                    sim[i][j] = (1 + (1 - spatial.distance.cosine(file_vec[file][i],file_vec[file][j]))) / 2
        file2sim[file[:-15]] = sim
    return file2sim

def main():
    args = parser.parse_args()
    print('Dataset is building...')
    train_loader, valid_loader, test_loader, train_raw, val_raw, test_raw = data_load(args)
    print('Dataset is saving...')
    
    train_raw_file = open("train_loader.pkl", "wb")
    pickle.dump(train_loader, train_raw_file)
    train_raw_file.close()
    
    val_raw_file = open("valid_loader.pkl", "wb")
    pickle.dump(valid_loader, val_raw_file)
    val_raw_file.close()

    test_raw_file = open("test_loader.pkl", "wb")
    pickle.dump(test_loader, test_raw_file)
    test_raw_file.close()

    print('Loding demographic data...')
    demo = load_demo()
    print ('Demographic data is saving...')
    demo_file = open("demo.pkl", "wb")
    pickle.dump(demo, demo_file)
    demo_file.close()

    print('Graph is building...')
    file2sim  = build_graph(train_raw,val_raw,test_raw)
    print('Graph is saving...')
    file2sim_file = open("file2sim_graph.pkl", "wb")
    pickle.dump(file2sim, file2sim_file)
    file2sim_file.close()
    print('=====DONE=====')


if __name__ == '__main__':
    main()