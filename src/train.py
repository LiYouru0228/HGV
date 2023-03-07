from model import HGV
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
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from utils import metrics
from load_data import Dataset


def get_loss(y_pred, y_true):
    loss = torch.nn.BCELoss()
    return loss(y_pred, y_true)


def Bootstrap(y_true,y_pred,test_ret):
    N = len(y_true)
    N_idx = np.arange(N)
    K = 1000

    auroc = []
    auprc = []
    minpse = []
    for i in range(K):
        boot_idx = np.random.choice(N_idx, N, replace=True)
        boot_true = np.array(y_true)[boot_idx]
        boot_pred = y_pred[boot_idx, :]
        test_ret = metrics.print_metrics_binary(boot_true, boot_pred, verbose=0)
        auroc.append(test_ret['auroc'])
        auprc.append(test_ret['auprc'])
        minpse.append(test_ret['minpse'])

    print('auroc %.4f(%.4f)'%(np.mean(auroc), np.std(auroc)))
    print('auprc %.4f(%.4f)'%(np.mean(auprc), np.std(auprc)))
    print('minpse %.4f(%.4f)'%(np.mean(minpse), np.std(minpse)))


file_name = './data/best_model2saved.pth'
train_loader = pickle.load(open("train_loader.pkl", "rb"))
valid_loader = pickle.load(open("valid_loader.pkl", "rb"))
test_loader = pickle.load(open("test_loader.pkl", "rb"))
file2sim = pickle.load(open("file2sim_graph.pkl", "rb"))
demographic_data,idx_list = pickle.load(open("demo.pkl", "rb"))[0],pickle.load(open("demo.pkl", "rb"))[1]


device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED) 
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED) 
torch.cuda.manual_seed(RANDOM_SEED) 
torch.backends.cudnn.deterministic=True 

def train(args):
    model = HGV(input_dim = args.input_dim, hidden_dim = args.hidden_dim, d_model = args.d_model, MHD_num_head = args.MHD_num_head, d_ff = args.d_ff, output_dim = 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_roc = 0
    max_prc = 0
    train_loss = []
    train_model_loss = []
    train_decov_loss = []
    valid_loss = []
    valid_model_loss = []
    valid_decov_loss = []
    history = []
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    for each_epoch in range(args.max_epoch):
        #-------------------- train -----------------------#
        batch_loss = []
        model_batch_loss = []
        decov_batch_loss = []

        model.train()

        for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):   
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_demo, graph = [],[]
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)
                graph.append(torch.tensor(file2sim[cur_idx], dtype=torch.float32))

            batch_demo = torch.stack(batch_demo).to(device)
            graph = torch.stack(graph).to(device)
            output, decov_loss = model(batch_x, batch_demo, graph, each_epoch, step)


            model_loss = get_loss(output, batch_y.unsqueeze(-1))
            loss = model_loss + 1000* decov_loss

            batch_loss.append(loss.cpu().detach().numpy())
            model_batch_loss.append(model_loss.cpu().detach().numpy())
            decov_batch_loss.append(decov_loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            if step % 30 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f'%(each_epoch, step, np.mean(np.array(batch_loss))))
                print('Model Loss = %.4f, Decov Loss = %.4f'%(np.mean(np.array(model_batch_loss)), np.mean(np.array(decov_batch_loss))))
        train_loss.append(np.mean(np.array(batch_loss)))
        train_model_loss.append(np.mean(np.array(model_batch_loss)))
        train_decov_loss.append(np.mean(np.array(decov_batch_loss)))

        #-------------------- val -----------------------#
        batch_loss = []
        model_batch_loss = []
        decov_batch_loss = []

        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_demo, graph = [],[]
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_demo.append(cur_demo)
                    graph.append(torch.tensor(file2sim[cur_idx], dtype=torch.float32))

                batch_demo = torch.stack(batch_demo).to(device)
                graph = torch.stack(graph).to(device)
                output,decov_loss = model(batch_x, batch_demo, graph, each_epoch, step)

                model_loss = get_loss(output, batch_y.unsqueeze(-1))

                loss = model_loss + 10* decov_loss
                batch_loss.append(loss.cpu().detach().numpy())
                model_batch_loss.append(model_loss.cpu().detach().numpy())
                decov_batch_loss.append(decov_loss.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        valid_loss.append(np.mean(np.array(batch_loss)))
        valid_model_loss.append(np.mean(np.array(model_batch_loss)))
        valid_decov_loss.append(np.mean(np.array(decov_batch_loss)))

        print("\n==>Predicting on validation")
        print('Valid Loss = %.4f'%(valid_loss[-1]))
        print('valid_model Loss = %.4f'%(valid_model_loss[-1]))
        print('valid_decov Loss = %.4f'%(valid_decov_loss[-1]))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        ret = metrics.print_metrics_binary(y_true, y_pred)
        history.append(ret)
        print()

        #-------------------- test -----------------------#
        batch_loss = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_demo, graph = [],[]
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_demo.append(cur_demo)
                    graph.append(torch.tensor(file2sim[cur_idx], dtype=torch.float32))

                batch_demo = torch.stack(batch_demo).to(device)
                graph = torch.stack(graph).to(device)
                output = model(batch_x, batch_demo, graph, each_epoch, step)[0]

                loss = get_loss(output, batch_y.unsqueeze(-1))
                batch_loss.append(loss.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        print("\n==>Predicting on test")
        print('Test Loss = %.4f'%(np.mean(np.array(batch_loss))))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        test_res = metrics.print_metrics_binary(y_true, y_pred)
        print ('experimental report:')
        Bootstrap(y_true,y_pred,test_res)
        
    print('=====DONE=====')


parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=int, default=76, help='size of input')
parser.add_argument('--hidden_dim', type=int, default=64, help='embeddings size')
parser.add_argument('--d_model', type=int, default=64, help='d_model')
parser.add_argument('--MHD_num_head', type=int, default=4, help='head of mutli-head attention networks')
parser.add_argument('--d_ff', type=int, default=256, help='d_ff')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=100, help='the number of epochs')


def main():
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
