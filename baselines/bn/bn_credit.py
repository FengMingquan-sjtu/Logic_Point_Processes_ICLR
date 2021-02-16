import sys
sys.path.extend(["./","../","../../"])
import os
import argparse

import torch
import numpy as np
import pandas as pd 
import bnlearn as bn
import pickle
import sklearn
import networkx as nx
import matplotlib.pyplot as plt
import bnlearn.helpers.network as network



def train_bn(bn_args):
    print("start train")
    df = preprocess(bn_args)
    df = df.astype(int)
    df = df.drop(['custAttr1','time'], axis=1)
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model = bn.parameter_learning.fit(model, df, methodtype="bayes", verbose=2)
    with open(bn_args.model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print("train finished")
    
def test_bn(bn_args):
    print("start test")
    df = preprocess(bn_args)
    df = df.astype(int)
    df = df.drop(['custAttr1','time'], axis=1)
    with open(bn_args.model_save_path, 'rb') as f:
        model = pickle.load(f)
    
    targets = ['8']
    result = dict()
    for t in targets:
        target = [t]
        y_df = df[target]
        x_df = df.drop(target, axis=1)

        acc = 0
        y_hat_list = list()
        y_true_list = list()

        for index, x_row in x_df.iterrows():
            factors = bn.inference.fit(model, variables=target, evidence=dict(x_row), verbose=0)
            ##raise ValueError
            y_hat = factors.values[1] #probability that flag=1
            y_hat_list.append(y_hat)
            y_true = y_df.loc[index, target].values
            y_true_list.append(y_true)
            if int(round(y_hat)) == y_true:
                acc +=1
        print("target is", target)
        print("acc=",acc/len(y_hat_list))
        result[t] = [y_hat_list,y_true_list]
    
    with open("test_result_credit.pkl", 'wb') as f:
        pickle.dump(result, f)
    print("test finished, result saved.")

def result_analyze():
    with open("test_result_credit.pkl", 'rb') as f:
        d = pickle.load(f)
    targets = ['8']
    for t in targets:
        y_hat_list,y_true_list = d[t]
        y_true = np.array(y_true_list).reshape((-1))
        y_pred = np.array(y_hat_list)
        auc = sklearn.metrics.roc_auc_score(y_true,y_pred)
        pos_rate = y_true.sum() / y_true.shape[0]

        #convert y_pred from probability to integer
        rand = np.random.random(y_pred.shape)
        y_pred = (y_pred>rand).astype(int)
        acc = sklearn.metrics.accuracy_score(y_true,y_pred)
        f1 = sklearn.metrics.f1_score(y_true,y_pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="all").ravel()
        print("Target is ",t)
        print("GT pos rate = ", pos_rate)
        print("auc=",auc)
        print("acc =",acc)
        print("f1=",f1)
        print("tn, fp, fn, tp =", confusion_matrix)
        print("-------")
    #y_hat = np.array(y_hat_list).round().astype(int)
    #print(d["mechanical"])
    #y_hat_list = [i>0.85 for i in y_hat_list]
    #y_hat = np.array(y_hat_list).astype(int)
    #y_true = np.array(y_true_list).reshape((-1))
    #print(y_hat)
    #acc_score = sklearn.metrics.accuracy_score(y_true,y_hat)
    #print(acc_score)
    #maybe add more sklearn.metrics

def result_analyze_only_jump():
    #only calculate acc at jump from init_state to the other state
    with open("test_result.pkl", 'rb') as f:
        d = pickle.load(f)
    targets = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
    for t in targets:
        y_hat_list,y_true_list = d[t]
        y_true = np.array(y_true_list).reshape((-1))
        y_pred = np.array(y_hat_list)
        init_state = y_true[0]
        mask = y_true[1:]-y_true[:-1]
        m1 = np.pad(mask,(0,1)) != init_state
        m2 = np.pad(mask,(1,0)) != init_state
        m = np.logical_or(m1,m2)
        y_true = y_true[m]
        y_pred = y_pred[m]
        auc = sklearn.metrics.roc_auc_score(y_true,y_pred)

        #convert y_pred from probability to integer
        rand = np.random.random(y_pred.shape)
        y_pred = (y_pred>rand).astype(int)
        acc = sklearn.metrics.accuracy_score(y_true,y_pred)
        f1 = sklearn.metrics.f1_score(y_true,y_pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="all").ravel()
        print("Target is ",t)
        print("auc=",auc)
        print("acc =",acc)
        print("f1=",f1)
        print("tn, fp, fn, tp =", confusion_matrix)
        print("-------")

def get_bn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="train",
                        help="tesk is one of [train, test]",)
    parser.add_argument('--train_csv_path', type=str, default="/home/fengmingquan/data/train_test_data_credit/train_lstm_50000.csv",
                        help="input training csv file path",)
                        #mini-data at /Users/fmq/Downloads/sepsis_data_three_versions/sepsis_01/mini.csv
    parser.add_argument('--test_csv_path', type=str, default="/home/fengmingquan/data/train_test_data_credit/test_lstm_balance.csv",
                        help="input testing csv file path",)
    parser.add_argument('--model_save_path', type=str, default="./trained_bn_model_credit.pkl",
                        help="trained model saving path",)
    bn_args = parser.parse_args()
    return bn_args

def preprocess(bn_args):
    if bn_args.task == "train":
        df = pd.read_csv(bn_args.train_csv_path)
    else:
        df = pd.read_csv(bn_args.test_csv_path)

    df = df.astype(int)
    return df 

def count_flag(df):
    targets = ['8']
    for t in targets:
        print(t)
        print(df[t].sum() / df[t].size)
    #(credit dataset test_lstm_balance) 0.3672566371681416
    #(credit dataset train_lstm_50000 ) 0.023262245213345696

def print_graph(bn_args):
    with open(bn_args.model_save_path, 'rb') as f:
        model = pickle.load(f)
    adj_df = model['adjmat']
    targets = ['8']
    mapping = {'0': 'zero_amount', '1': 'historic_zero_amount', '2': 'multiple_credit_card', '3': 'extreme_large_amount', '4': 'extreme_small_amount', '5': 'time_gap', '6': 'multi_zip', '7': 'history_fraud', '8': 'fraud'}
    result = dict()
    for t in targets:
        cause = adj_df[t][adj_df[t]].index.to_list()
        cause = [mapping[c] for c in cause]
        print(mapping[t]," <-- ",cause)
        result[mapping[t]] = cause
    
    return result


def plot_graph_simple(bn_result):
    
    graph = nx.Graph()    
    for child, parents in bn_result.items():
        for parent in parents:
            graph.add_edge(parent, child)
    pos = nx.bipartite_layout(graph, list(bn_result.keys()))

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=5, edge_color="b")
    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color="#210070", alpha=0.9)
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(graph, pos, font_size=17, bbox=label_options)
    #nx.draw_networkx(graph, arrowsize=20,ax=ax, pos=pos,alpha=0.8)
    #ax.set_axis_off()
    ax.margins(0.2, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.savefig("bn_credit.pdf")
    


if __name__ == "__main__":
    
    #args = get_args() #global args
    bn_args = get_bn_args() #bn(local) args
    
    #if bn_args.task == "train":
    #    train_bn(bn_args)
    #if bn_args.task == "test":
    #    test_bn(bn_args)
    
    #df = preprocess(bn_args)
    #count_flag(df)
    #result_analyze()
    #result_analyze_only_jump()
    result = print_graph(bn_args)
    #plot_graph(bn_args)
    plot_graph_simple(result)