"""this file trains model using mimic dataset """
import sys
#sys.path.extend(["./","../","../../"])
import os
import argparse

import numpy as np

import sklearn
import pickle
#import pydotplus
#from sklearn.tree import export_graphviz

from train import GRUTree,GRU


def convert_from_dict(in_file, target, is_train, interval):
    dataset = np.load('../../Shuang_sythetic data codes/data/'+in_file, allow_pickle='TRUE').item()
    

    np_list = list()
    for d in dataset.values():
        T_max = max([d[p]["time"][-1] if d[p]["time"] else 0 for p in d.keys()])
        n_interval = int(T_max // interval)
        n_pred = len(d.keys())
        #print(n_interval)
        converted_d = np.zeros((n_pred, n_interval),dtype=int)
        
        for pid,p in d.items():
            for i in range(n_interval):
                for idx,t in enumerate(p["time"]):
                    if p["state"][idx] == 1 and i*interval <=t <=(i+1)*interval:
                        converted_d[pid,i] = 1
                        break
                    if t > (i+1)*interval:
                        break
        np_list.append(converted_d)
    
    if is_train:
        np_list = np_list[:int(len(np_list)*0.8)]
    else:
        np_list = np_list[int(len(np_list)*0.8):]

    
    F_list = list()
    for idx in range(len(np_list)+1):
        f = sum([d.shape[1] for d in np_list[:idx]])
        F_list.append(f)
    F = np.array(F_list).astype(int)
    D = np.concatenate(np_list, axis=1)
    target_list = [target]
    body_list = [i for i in range(D.shape[0]) if i != target]
    X = D[body_list]
    Y = D[target_list].reshape((1,-1))
    return F,X,Y



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
        icustay_id = row['icustay_id']
        if cur_id != icustay_id:
            index_list.append(index)
            cur_id = icustay_id
    index_list.append(df.index[-1])
    F = np.array(index_list)

    #get X,y
    df = df.drop(['icustay_id'], axis=1)
    #target = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
    y_df = df[target]
    y = y_df.to_numpy().T
    x_df = df.drop(target, axis=1)
    predicate_name = x_df.columns
    #print(list(x_df.columns))
    X = x_df.to_numpy().T

    return X,F,y,predicate_name

def run_2(in_file, model_file, target, interval):
    F,X,Y = convert_from_dict(in_file, target, is_train=True, interval=interval)
    print("data process finished",flush=1)
    in_count = X.shape[0]
    out_count = Y.shape[0]
    state_count = 4 #GRU hidden states
    hidden_sizes = 4 #MLP mid-layers
    gru_args = (in_count, state_count, out_count)
    gru = GRUTree(in_count=in_count, state_count=state_count, hidden_sizes=[hidden_sizes,], out_count=out_count, strength=80.0) #strength means how much to weigh tree-regularization term.
    print("total weight ", gru.gru.num_weights+gru.mlp.num_weights)
    #gru.train(X, F, Y, iters_retrain=25, gru_iters=300, mlp_iters=5000, batch_size=256, lr=1e-3, param_scale=0.1, log_every=100)
    gru.train(X, F, Y, iters_retrain=1, gru_iters=10, mlp_iters=100, batch_size=64, lr=1e-2, param_scale=0.1, log_every=100)
    with open('./trained_models/{}'.format(model_file), 'wb') as fp:
        pickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights, 'gru_args':gru_args, 'tree':gru.tree}, fp)


def test(in_file, model_file, target, interval):
    
    F,X,y = convert_from_dict(in_file, target, is_train=False, interval=interval)
    with open('./trained_models/{}'.format(model_file), 'rb') as fp:
        model = pickle.load(fp)
        weights = model['gru']
        gru_args = model['gru_args']
    gru = GRU(*gru_args)
    gru.weights = weights
    y_hat = gru.pred_fun(gru.weights, X, F)
    print(y.shape)
    print(y_hat.shape)

    
    y_true = y[0]        
    y_pred = y_hat[0]
    
    ae_list = list()
    ac_list = list()
    for f_idx in range(0, len(F)-1):
        sample_y_true = y_true[F[f_idx]: F[f_idx+1]]
        sample_y_pred = y_pred[F[f_idx]: F[f_idx+1]]
        #print("sample_y_pred=",sample_y_pred)
        gt_time = [i*interval for i,s in enumerate(sample_y_true) if s==1]
        pred_time = [i*interval for i,s in enumerate(sample_y_pred) if s>=0.4]
        #print("gt-time:", gt_time)
        #print("pred-time:", pred_time)
        length = len(gt_time)
        if len(pred_time) > length:
            pred_time = pred_time[:length]
        elif len(pred_time) < length:
            #gt_time = pred_time[:len(pred_time)]
            if len(pred_time) == 0:
                pred_time.append(0)
            pred_time.extend([pred_time[-1],] * (length-len(pred_time)))
        length = len(gt_time)
        if length == 0:
            continue
            
        threshold = 1
        if length > 1:
            ae = np.abs(np.diff(np.array(gt_time)) - np.diff(np.array(pred_time)))
            ac = threshold > np.abs(np.diff(np.array(gt_time)) - np.diff(np.array(pred_time)))
        else:
            ae = np.abs(np.array(gt_time) - np.array(pred_time))
            ac = threshold > np.abs(np.array(gt_time) - np.array(pred_time))
        ae_list.append(ae)
        ac_list.append(ac)
    mae = np.concatenate(ae_list).mean()
    acc = np.concatenate(ac_list).mean()
    print("target=", target)
    print("MAE = ", mae)
    print("ACC =", acc)
    
    



        

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
        target = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
        for t in target:
            train(tr_args,[t])
    elif tr_args.task == "test":
        test(tr_args)

def get_vis():
    with open('./trained_models/model.pkl', 'rb') as fp:
        model = pickle.load(fp)
        tree = model['tree']
    save_path = './trained_models/vis_with_name.pdf'
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

    #tr_args = get_tr_args()
    #X,F,y = preprocess(tr_args)
    #run(tr_args)
    #get_vis()

    #run_2(in_file="crime_all_day_scaled.npy", model_file="crime-10.pkl", target=10, interval=0.5)
    #test(in_file="crime_all_day_scaled.npy", model_file="crime-10.pkl", target=10, interval=0.5)
    #run_2(in_file="crime_all_day_scaled.npy", model_file="crime-11.pkl", target=11, interval=0.5)
    #test(in_file="crime_all_day_scaled.npy", model_file="crime-11.pkl", target=11, interval=0.5)
    #run_2(in_file="crime_all_day_scaled.npy", model_file="crime-12.pkl", target=12, interval=0.5)
    #test(in_file="crime_all_day_scaled.npy", model_file="crime-12.pkl", target=12, interval=0.5)
    run_2(in_file="crime_all_day_scaled.npy", model_file="crime-13.pkl", target=13, interval=0.5)
    test(in_file="crime_all_day_scaled.npy", model_file="crime-13.pkl", target=13, interval=0.5)
    
