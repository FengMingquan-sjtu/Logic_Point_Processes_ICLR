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

from logic import Logic
from dataloader import get_dataset
from utils.args import get_args
from dataset_utils import convert_temporal_to_static_data


def convert_dict_to_df(dataset):
    """
    input: stattic dataset in dict form

    output: (N*T) * num_pred dataframe
    """
    #print(dataset)
    new_data = {predicate_ID:[] for predicate_ID in dataset[0].keys()}
    for sample_ID,data in dataset.items():
        min_len = min([len(t) for t in data.values()])
        for predicate_ID,data_ in data.items():
            new_data[predicate_ID].extend(data_[:min_len])#align each predicates
    #print(new_data)
    df = pd.DataFrame(data=new_data)
    return df


def train_bn(dataset, args):
    static_dataset = convert_temporal_to_static_data(dataset, args)
    df = convert_dict_to_df(static_dataset)
    model = bn.structure_learning.fit(df)
    G = bn.plot(model)

def train_mimic_bn(bn_args):
    print("start train")
    df = preprocess(bn_args)
    df = df.astype(int)
    df = df.drop(['row_id','icustay_id','valuenum1','valuenum2'], axis=1)
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model = bn.parameter_learning.fit(model, df, methodtype="bayes", verbose=2)
    with open(bn_args.model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print("train finished")
    
def test_mimic_bn(bn_args):
    print("start test")
    df = preprocess(bn_args)
    df = df.astype(int)
    df = df.drop(['row_id','icustay_id','valuenum1','valuenum2'], axis=1)
    with open(bn_args.model_save_path, 'rb') as f:
        model = pickle.load(f)
    
    targets = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
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
    
    with open("test_result.pkl", 'wb') as f:
        pickle.dump(result, f)
    print("test finished, result saved.")

def result_analyze():
    with open("test_result.pkl", 'rb') as f:
        d = pickle.load(f)
    targets = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
    for t in targets:
        y_hat_list,y_true_list = d[t]
        y_true = np.array(y_true_list).reshape((-1))
        y_pred = np.array(y_hat_list)
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
    parser.add_argument('--train_csv_path', type=str, default="/Users/fmq/Downloads/sepsis_data_three_versions/sepsis_01/sepsis_01_test.csv",
                        help="input training csv file path",)
                        #mini-data at /Users/fmq/Downloads/sepsis_data_three_versions/sepsis_01/mini.csv
    parser.add_argument('--test_csv_path', type=str, default="/Users/fmq/Downloads/sepsis_data_three_versions/sepsis_01/sepsis_01_test.csv",
                        help="input testing csv file path",)
    parser.add_argument('--model_save_path', type=str, default="./trained_bn_model.pkl",
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
    targets = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
    for t in targets:
        print(t)
        print(df[t].sum() / df[t].size)

def print_graph(bn_args):
    with open(bn_args.model_save_path, 'rb') as f:
        model = pickle.load(f)
    adj_df = model['adjmat']
    targets = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
    for t in targets:
        cause = adj_df[t][adj_df[t]].index.to_list()
        print(t," <-- ",cause)

def plot_graph(bn_args):
    with open(bn_args.model_save_path, 'rb') as f:
        model = pickle.load(f)
    adj_df = model['adjmat']
    plot(adj_df)


def plot(model, pos=None, scale=1, figsize=(15, 8), verbose=3):
    """Plot the learned stucture.
    Parameters
    ----------
    model : dict
        Learned model from the .fit() function..
    pos : graph, optional
        Coordinates of the network. If there are provided, the same structure will be used to plot the network.. The default is None.
    scale : int, optional
        Scaling parameter for the network. A larger number will linearily increase the network.. The default is 1.
    figsize : tuple, optional
        Figure size. The default is (15,8).
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE
    Returns
    -------
    dict containing pos and G
        pos : list
            Positions of the nodes.
        G : Graph
            Graph model
    """
    out = {}
    G = nx.DiGraph()  # Directed graph
    layout='fruchterman_reingold'

    # Extract model if in dict
    if 'dict' in str(type(model)):
        model = model.get('model', None)

    # Bayesian model
    if 'BayesianModel' in str(type(model)) or 'pgmpy' in str(type(model)):
        if verbose>=3: print('[bnlearn] >Plot based on BayesianModel')
        # positions for all nodes
        pos = network.graphlayout(model, pos=pos, scale=scale, layout=layout, verbose=verbose)
        # Add directed edge with weigth
        # edges=model.edges()
        edges=[*model.edges()]
        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight=1, color='k')
    elif 'networkx' in str(type(model)):
        if verbose>=3: print('[bnlearn] >Plot based on networkx model')
        G = model
        pos = network.graphlayout(G, pos=pos, scale=scale, layout=layout, verbose=verbose)
    else:
        if verbose>=3: print('[bnlearn] >Plot based on adjacency matrix')
        G = network.adjmat2graph(model)
        # Get positions
        #pos = network.graphlayout(G, pos=pos, scale=scale, layout=layout, verbose=verbose)
        pos = nx.spiral_layout(model, scale=scale)
    # Bootup figure
    plt.figure(figsize=figsize)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.85)
    # edges
    colors = [G[u][v].get('color', 'k') for u, v in G.edges()]
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, arrowstyle='->', edge_color=colors, width=weights)
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # Get labels of weights
    # labels = nx.get_edge_attributes(G,'weight')
    # Plot weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    # Making figure nice
    ax = plt.gca()
    ax.set_axis_off()
    #plt.show()
    plt.savefig("fig.png")

    # Store
    out['pos']=pos
    out['G']=G
    return(out)

def plot_graph_simple():
    bn_result = {"Mortality" :['urine_output', 'adm_order', 'paO2'],
            "Ventilation":['weight', 'height', 'SGOT', 'Magnesium'],
            "Median-Vaso":['urine_output', 'Ht', 'sofa', 'max_dose_vaso'],
            "Max-Vaso":['urine_output', 'adm_order', 'Arterial_lactate', 'INR', 'sofa']}
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
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.savefig("bn.pdf")
    


if __name__ == "__main__":
    
    #args = get_args() #global args
    bn_args = get_bn_args() #bn(local) args
    #train_dataset, test_dataset = get_dataset(args)
    #print(train_dataset)
    #_test_convert_dataset()
    #train_bn(train_dataset, args)
    #if bn_args.task == "train":
    #    train_mimic_bn(bn_args)
    #if bn_args.task == "test":
    #    test_mimic_bn(bn_args)
    
    #df = preprocess(bn_args)
    #count_flag(df)
    #result_analyze()
    #result_analyze_only_jump()
    #print_graph(bn_args)
    #plot_graph(bn_args)
    plot_graph_simple()