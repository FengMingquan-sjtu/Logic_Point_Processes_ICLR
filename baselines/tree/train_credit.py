"""this file trains model using mimic dataset """
import sys
sys.path.extend(["./","../","../../"])
import os
import argparse

import numpy as np
import pandas as pd
import sklearn
import pickle
import pydotplus
from sklearn.tree import export_graphviz

from train import GRUTree,GRU

def preprocess(tr_args, target):
    """
    input: args, target

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
        custAttr1 = row['custAttr1']
        if cur_id != custAttr1:
            index_list.append(index)
            cur_id = custAttr1
    index_list.append(df.index[-1])
    F = np.array(index_list)

    #get X,y
    df = df.drop(['custAttr1','time'], axis=1)
    y_df = df[target]
    y = y_df.to_numpy().T
    x_df = df.drop(target, axis=1)
    predicate_name = x_df.columns
    #print(list(x_df.columns))
    X = x_df.to_numpy().T

    return X,F,y,predicate_name



def train(tr_args, target):
    print("start train", flush=1)
    X,F,y,predicate_name = preprocess(tr_args, target)
    in_count = X.shape[0]
    #out_count = y.shape[0]
    out_count = len(target)
    state_count = 5
    gru_args = (in_count, state_count, out_count)
    gru = GRUTree(in_count=in_count, state_count=state_count, hidden_sizes=[25], out_count=out_count, strength=80.0) #strength means how much to weigh tree-regularization term.
    print("total weight ", gru.gru.num_weights+gru.mlp.num_weights)

    gru.train(X, F, y, iters_retrain=25, gru_iters=300, mlp_iters=5000, batch_size=256, lr=1e-3, param_scale=0.1, log_every=100)
    #gru.train(X, F, y, iters_retrain=1, gru_iters=1, mlp_iters=1, batch_size=256, lr=1e-3, param_scale=0.1, log_every=100)
    
    if not os.path.isdir('./trained_models_credit'):
        os.mkdir('./trained_models_credit')
    class_names = ['on', 'off']
    visualize(gru.tree, './trained_models_credit/vis_{}.pdf'.format(target[0]), predicate_name, class_names)
    with open('./trained_models_credit/model.pkl', 'wb') as fp:
        pickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights, 'gru_args':gru_args, 'tree':gru.tree}, fp)
    print('saved visualize pdf and model to ./trained_models_credit',flush=1)
    return gru

def test(tr_args):
    target = ['8']
    X,F,y,_ = preprocess(tr_args, target)
    
    with open('./trained_models_credit/model.pkl', 'rb') as fp:
        model = pickle.load(fp)
        weights = model['gru']
        gru_args = model['gru_args']
    gru = GRU(*gru_args)
    gru.weights = weights
    y_hat = gru.pred_fun(gru.weights, X, F)
    print(y.shape)
    print(y_hat.shape)

    for i in range(len(target)):
        y_true = y[i].T.astype(int)
        y_pred = y_hat[i].T
        pos_rate = y_true.sum() / y_true.shape[0]
        #auc receives probability score of y_pred
        auc = sklearn.metrics.roc_auc_score(y_true,y_pred)

        #convert to binary y_pred
        rand = np.random.random(y_pred.shape)
        y_pred = (y_pred>rand).astype(int)
        acc = sklearn.metrics.accuracy_score(y_true,y_pred)
        f1 = sklearn.metrics.f1_score(y_true,y_pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="all").ravel()
        print("Target is ",target[i])
        print("GT pos rate = ", pos_rate)
        print("auc =",auc)
        print("acc =",acc)
        print("f1=",f1)
        print("tn, fp, fn, tp =", confusion_matrix)
        print("-------")

def get_tr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="train",
                        help="tesk is one of [train, test]",)
    parser.add_argument('--train_csv_path', type=str, default="/home/fengmingquan/data/train_test_data_credit/train_lstm_50000.csv",
                        help="input training csv file path",)
    parser.add_argument('--test_csv_path', type=str, default="/home/fengmingquan/data/train_test_data_credit/test_lstm_balance.csv",
                        help="input testing csv file path",)
    parser.add_argument('--model_save_path', type=str, default="./trained_tr_model_credit.pkl",
                        help="trained model saving path",)
    tr_args = parser.parse_args()
    return tr_args

def run(tr_args):
    if tr_args.task == "train":
        target = ['8']
        for t in target:
            train(tr_args,[t])
    elif tr_args.task == "test":
        test(tr_args)

def get_vis():
    with open('./trained_models/model.pkl', 'rb') as fp:
        model = pickle.load(fp)
        tree = model['tree']
    save_path = './trained_models_credit/vis_with_name.pdf'
    #print(type(tree))
    visualize(tree,save_path)

def visualize(tree, save_path, predicate_name, class_names):
    """Generate PDF of a decision tree.

    @param tree: DecisionTreeClassifier instance
    @param save_path: string 
                      where to save tree PDF
    """

    dot_data = export_graphviz(tree, out_file=None,
                               filled=True, rounded=True, feature_names= predicate_name, class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph = make_graph_minimal(graph)  # remove extra text

    if not save_path is None:
        graph.write_pdf(save_path)


def make_graph_minimal(graph):
    nodes = graph.get_nodes()
    for node in nodes:
        old_label = node.get_label()
        label = prune_label(old_label)
        if label is not None:
            node.set_label(label)
    return graph


def prune_label(label):
    if label is None:
        return None
    if len(label) == 0:
        return None
    label = label[1:-1]
    parts = [part for part in label.split('\\n')
             if 'gini =' not in part and 'samples =' not in part]
    return '"' + '\\n'.join(parts) + '"'

if __name__ == "__main__":

    tr_args = get_tr_args()
    #X,F,y = preprocess(tr_args)
    run(tr_args)
    #get_vis()
            
    
