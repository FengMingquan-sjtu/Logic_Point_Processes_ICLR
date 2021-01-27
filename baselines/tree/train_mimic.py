"""this file trains model using mimic dataset """
import sys
sys.path.extend(["./","../","../../"])
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle

from logic import Logic
from dataloader import get_dataset
from utils.args import get_args
from dataset_utils import convert_temporal_to_static_data
from train import GRUTree,visualize

def preprocess(tr_args):
    """
    input: args

    output: 3 np arrays:
        X(2-dim): num_neighbour * (T*N)
        F(1-dim): N
        y(2-dim): 1 * (T*N)
        where X is neighbour, y is target, F is spliter of samples. T is discrete time, N is num of samples.
        As example of F: fenceposts_Np1 = np.arange(0, (n_seqs + 1) * n_timesteps, n_timesteps)
    """
    if tr_args.task == "train":
        df = pd.read_csv(tr_args.train_csv_path)
    else:
        df = pd.read_csv(tr_args.test_csv_path)

    # get index_list (i.e. F)
    cur_id = None
    index_list = list()
    for index, row in df.iterrows():
        icustay_id = row['icustay_id']
        if cur_id != icustay_id:
            index_list.append(index)
            cur_id = icustay_id
    index_list.append(df.index[-1])
    F = np.array(index_list)

    #get X,y
    df = df.drop(['row_id','icustay_id','valuenum1','valuenum2'], axis=1)
    target = ['flag']
    y_df = df[target]
    y = y_df.to_numpy().reshape((1,-1))
    x_df = df.drop(target, axis=1)
    X = x_df.to_numpy().T

    return X,F,y



def train(tr_args):
    X,F,y = preprocess(tr_args)
    in_count = X.shape[0]
    out_count = y.shape[0]
    gru = GRUTree(in_count=in_count, state_count=20, hidden_sizes=[25], out_count=out_count, strength=1000.0) #strength means how much to weigh tree-regularization term.
    gru.train(X, F, y, iters_retrain=1, num_iters=1, batch_size=10, lr=1e-2, param_scale=0.1, log_every=10)
    visualize(gru.tree, './trained_models/pp_dataset_result.pdf')
    with open('./trained_models/trained_tr_model.pkl', 'wb') as fp:
        pickle.dump(gru, fp)
    print('saved visualize pdf and model to ./trained_models')
    return gru

def test(X,F,y):
    with open('./trained_models/trained_tr_model.pkl', 'rb') as fp:
        gru = pickle.load(fp)
    y_hat = gru.pred_fun(gru.weights, X, F)
    auc_test = roc_auc_score(y.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))


def get_tr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="train",
                        help="tesk is one of [train, test]",)
    parser.add_argument('--train_csv_path', type=str, default="/home/fengmingquan/data/sepsis_data_three_versions/sepsis_continuous/sepsis_continuous_train.csv",
                        help="input training csv file path",)
    parser.add_argument('--test_csv_path', type=str, default="/home/fengmingquan/data/sepsis_data_three_versions/sepsis_continuous/sepsis_continuous_test.csv",
                        help="input testing csv file path",)
    parser.add_argument('--model_save_path', type=str, default="./trained_tr_model.pkl",
                        help="trained model saving path",)
    tr_args = parser.parse_args()
    return tr_args

if __name__ == "__main__":
    #args = get_args()
    #train_dataset, test_dataset = get_dataset(args)
    #target_ID = args.target_predicate[0]
    #dataset = convert_temporal_to_static_data(train_dataset, args)
    #X,F,y = convert_dict_to_array(dataset,target_ID)
    #print(X,F,y)
    #GRU = train(X,F,y)
    #dataset_t = convert_temporal_to_static_data(test_dataset, args)
    #X_t, F_t, y_t = convert_dict_to_array(dataset_t, target_ID)
    #test(X,F,y,GRU)
    tr_args = get_tr_args()
    #X,F,y = preprocess(tr_args)
    if tr_args.task == "train":
        train(tr_args)
    
