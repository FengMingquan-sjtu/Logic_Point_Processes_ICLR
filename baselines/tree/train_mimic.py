"""this file trains model using mimic dataset """
import sys
sys.path.extend(["./","../","../../"])
import os
import argparse

import numpy as np
import pandas as pd
import sklearn
import pickle

from dataloader import get_dataset
from utils.args import get_args
from dataset_utils import convert_temporal_to_static_data
from train import GRUTree,visualize,GRU

def preprocess(tr_args):
    """
    input: args

    output: 3 np arrays:
        X(2-dim): Dx * (T*N)
        F(1-dim): N
        y(2-dim): Dy * (T*N)
        where X is neighbour, y is target, F is spliter of samples. T is discrete time, N is num of samples, Dx is num of body predicates, Dy is num of targets
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
    df = df.drop(['icustay_id'], axis=1)
    target = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
    y_df = df[target]
    y = y_df.to_numpy().T
    x_df = df.drop(target, axis=1)
    X = x_df.to_numpy().T

    return X,F,y



def train(tr_args):
    print("start train", flush=1)
    X,F,y = preprocess(tr_args)
    in_count = X.shape[0]
    out_count = y.shape[0]
    state_count = 5
    gru_args = (in_count, state_count, out_count)
    gru = GRUTree(in_count=in_count, state_count=state_count, hidden_sizes=[25], out_count=out_count, strength=80.0) #strength means how much to weigh tree-regularization term.
    gru.train(X, F, y, iters_retrain=25, gru_iters=300, mlp_iters=5000, batch_size=256, lr=1e-3, param_scale=0.1, log_every=100)
    
    visualize(gru.tree, './trained_models/vis.pdf')
    with open('./trained_models/model.pkl', 'wb') as fp:
        pickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights, 'gru_args':gru_args}, fp)
    print('saved visualize pdf and model to ./trained_models',flush=1)
    return gru

def test(tr_args):
    X,F,y = preprocess(tr_args)
    target = tr_args.target
    with open('./trained_models/model.pkl', 'rb') as fp:
        model = pickle.load(fp)
        weights = model['gru']
        gru_args = model['gru_args']

    gru = GRU(*gru_args)
    gru.weights = weights
    y_hat = gru.pred_fun(gru.weights, X, F)
    y_true = y.T 
    y_pred = y_hat.T
    #auc receives probability score of y_pred
    auc = sklearn.metrics.roc_auc_score(y_true,y_pred)

    #convert to binary y_pred
    rand = np.random.random(y_pred.shape)
    y_pred = (y_pred>rand).astype(int)
    acc = sklearn.metrics.accuracy_score(y_true,y_pred)
    f1 = sklearn.metrics.f1_score(y_true,y_pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="all").ravel()
    print("Target is ",target)
    print("auc =",auc)
    print("acc =",acc)
    print("f1=",f1)
    print("tn, fp, fn, tp =", confusion_matrix)
    print("-------")

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

def run(tr_args):
    if tr_args.task == "train":
        train(tr_args)
    elif tr_args.task == "test":
        test(tr_args)

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
    run(tr_args)
            
    
