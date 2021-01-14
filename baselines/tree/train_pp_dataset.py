"""this file trains model using logicPP dataset """
import sys
sys.path.extend(["./","../","../../"])
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from logic import Logic
from dataloader import get_dataset
from utils.args import get_args
from dataset_utils import convert_temporal_to_static_data
from train import GRUTree,visualize

def convert_dict_to_array(dataset, target_ID):
    """
    input: stattic dataset in dict form

    output: 3 np.array:
        X(2-dim): num_neighbour * (T*N)
        F(1-dim): N
        y(2-dim): 1 * (T*N)
        where X is neighbour, y is target, F is spliter of samples. T is discrete time, N is num of samples.
    """
    #print(dataset)
    predicate_ID_list = sorted(list(list(dataset.values())[0].keys()))
    F = [0,]
    new_data = {predicate_ID:[] for predicate_ID in predicate_ID_list}
    for sample_ID,data in dataset.items():
        min_len = min([len(t) for t in data.values()])
        F.append(F[-1] + min_len)
        for predicate_ID,data_ in data.items():
            new_data[predicate_ID].extend(data_[:min_len])#align each predicates
    #print(new_data)
    X,y = list(),list()
    for predicate_ID in predicate_ID_list:
        if predicate_ID == target_ID:
            y.append(new_data[predicate_ID])
        else:
            X.append(new_data[predicate_ID])
    X = np.array(X)
    F = np.array(F)
    y = np.array(y)
    return X,F,y

def train(X,F,y):
    in_count = X.shape[0]
    out_count = y.shape[0]
    gru = GRUTree(in_count=in_count, state_count=20, hidden_sizes=[25], out_count=out_count, strength=1000.0) #strength means how much to weigh tree-regularization term.
    gru.train(X, F, y, iters_retrain=1, num_iters=1, batch_size=10, lr=1e-2, param_scale=0.1, log_every=10)
    visualize(gru.tree, './trained_models/pp_dataset_result.pdf')
    print('saved final decision tree to ./trained_models')
    return gru

def test(X,F,y, gru):
    y_hat = gru.pred_fun(gru.weights, X, F)
    auc_test = roc_auc_score(y.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))

if __name__ == "__main__":
    args = get_args()
    train_dataset, test_dataset = get_dataset(args)
    target_ID = args.target_predicate[0]
    dataset = convert_temporal_to_static_data(train_dataset, args)
    X,F,y = convert_dict_to_array(dataset,target_ID)
    #print(X,F,y)
    GRU = train(X,F,y)
    dataset_t = convert_temporal_to_static_data(test_dataset, args)
    X_t, F_t, y_t = convert_dict_to_array(dataset_t, target_ID)
    test(X,F,y,GRU)