import os.path as osp
import argparse

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt

from pkg.utils.misc import AverageMeter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# data format:
# [(1094.5910501207484, 4), (1098.1079940088416, 7), (1100.4624087040852, 1), (1101.3163237708202, 1), (1105.8506944816406, 1), (1106.1821513888342, 0), (1106.1918308538147, 8)]
def get_data(dataset_name, start_idx, end_idx):
    dataset_path = '../../Shuang_sythetic data codes/data/{}.npy'.format(dataset_name)
    print("dataset_path is ",dataset_path)
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    dataset = [dataset[i] for i in range(start_idx, end_idx)]
    num_sample = len(dataset)
    print("sample num is ", num_sample)
    return dataset

def load_mimic(dataset_name, start_idx, end_idx):
    data = get_data(dataset_name, start_idx, end_idx)
    event_seqs = list()
    n_types = 62
    pred_list = list(range(n_types))
    T_max = 500.0
    
    for sample_idx in range(end_idx-start_idx):
        sample = list()
        for pred in pred_list:
            pred_events = list()
            for idx,t in enumerate(data[sample_idx][pred]["time"]):
                
                if pred == 61:
                    if data[sample_idx][pred]["state"][idx]==0:
                        pred_events.append((t+1e-5, float(pred)))
                else:
                    if data[sample_idx][pred]["state"][idx]==1:
                        pred_events.append((t+1e-5, float(pred)))
            if len(pred_events) ==0: #use dummy event to fill empty pred.
                pred_events.append((T_max, float(pred)))
            sample.extend(pred_events)
            
        sample.sort(key=lambda x:x[0]) #sort by time
        event_seqs.append(sample)
    event_seqs = np.array(event_seqs,dtype=float)
    return event_seqs, n_types

def preprocess(args):
    print("preprocess start")
    train_event_seqs, n_types = load_mimic(args.dataset, 0, 3000)
    test_event_seqs, n_types = load_mimic(args.dataset, 3000, 3700)
    np.savez_compressed(osp.join("./data", "{}.npz".format(args.dataset)),
        train_event_seqs=train_event_seqs,
        test_event_seqs=test_event_seqs,
        n_types=n_types)
    print("preprocess finished")


def test_load(args):
    data = np.load(osp.join("./data", "{}.npz".format(args.dataset)), allow_pickle=True)
    n_types = int(data["n_types"])
    train_event_seqs = data["train_event_seqs"]
    test_event_seqs =  data["test_event_seqs"]
    #print(n_types)
    print(len(test_event_seqs))
    print(len(train_event_seqs))

def load_mat(args):
    path = "./data/output/{}/split_id=0/{}/scores_mat.txt".format(args.dataset, args.model_name)
    mat = np.genfromtxt(path)
    #print(mat)
    #print(mat.shape)
    return mat

def load_mae(args):
    with open("./{}/result_{}.pkl".format(args.dataset, args.model_name),'rb') as f:
        event_seqs_pred, test_event_seqs = pickle.load(f)
    #print(event_seqs_pred)
    #print(test_event_seqs)
    print(args.dataset, args.model_name)
    calc_mean_absolute_error(test_event_seqs, event_seqs_pred)
    


def calc_mean_absolute_error(event_seqs_true, event_seqs_pred):
    """
    Args:
        event_seqs_true (List[List[Tuple]]):
        event_seqs_pred (List[List[Tuple]]):
    """
    target_dict = {'survival':61, 'real_time_urine_output_low':51}

    result_dict_mae = {t:AverageMeter() for t in target_dict.keys() }
    result_dict_acc = {t:AverageMeter() for t in target_dict.keys() }

    for seq_true, seq_pred in zip(event_seqs_true, event_seqs_pred):
        for t_name, t_idx in target_dict.items():
            gt_time = [event[0] for event in seq_true if event[1] == t_idx]
            length = len(gt_time)
            pred_time = [event[0] for event in seq_pred if event[1] == t_idx]
            if len(pred_time) > length:
                pred_time = pred_time[:length]
            elif len(pred_time) < length:
                #gt_time = pred_time[:len(pred_time)]
                if len(pred_time) == 0:
                    pred_time.append(0)
                pred_time.extend([pred_time[-1],] * (length-len(pred_time)))
            
            threshold = 15
            if length > 1:
                mae = np.abs(np.diff(np.array(gt_time)) - np.diff(np.array(pred_time))).mean()
                acc = (threshold > np.abs(np.diff(np.array(gt_time)) - np.diff(np.array(pred_time)))).mean()
            else:
                mae = np.abs(np.array(gt_time) - np.array(pred_time)).mean()
                acc = (threshold > np.abs(np.array(gt_time) - np.array(pred_time))).mean()
            
            length = len(gt_time)
            if length > 0:
                result_dict_mae[t_name].update(mae, length)
                result_dict_acc[t_name].update(acc, length)
        #raise ValueError
    
    target = list(target_dict.keys())
    mae = [result_dict_mae[t].avg for t in target]
    acc = [result_dict_acc[t].avg for t in target]
    print("MAE:", target)
    print(("&{:.3f} "*len(target)).format(*mae))
    print("ACC:", target)
    print(("&{:.3f} "*len(target)).format(*acc))






class Kernel:
    def __init__(self,norms):
        self.norms =  (norms-np.min(norms))/(np.max(norms)-np.min(norms))
        self.n_nodes = norms.shape[0]
    def get_kernel_norms(self):
        return self.norms

def plot_hawkes_kernel_norms(kernel_object, show=True, pcolor_kwargs=None,
                             node_names=None, rotate_x_labels=0.):
    """Generic function to plot Hawkes kernel norms.

    Parameters
    ----------
    kernel_object : `Object`
        An object that must have the following API :

        * `kernel_object.n_nodes` : a field that stores the number of nodes
          of the associated Hawkes process (thus the number of kernels is
          this number squared)
        * `kernel_object.get_kernel_norms()` : must return a 2d numpy
          array with the norm of each kernel

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    pcolor_kwargs : `dict`, default=`None`
        Extra pcolor kwargs such as cmap, vmin, vmax

    node_names : `list` of `str`, shape=(n_nodes, ), default=`None`
        node names that will be displayed on axis.
        If `None`, node index will be used.

    rotate_x_labels : `float`, default=`0.`
        Number of degrees to rotate the x-labels clockwise, to prevent 
        overlapping.

    Notes
    -----
    Kernels are displayed such that it shows norm of column influence's
    on row.
    """
    n_nodes = kernel_object.n_nodes

    if node_names is None:
        node_names = range(n_nodes)
    elif len(node_names) != n_nodes:
        ValueError('node_names must be a list of length {} but has length {}'
                   .format(n_nodes, len(node_names)))

    row_labels = ['$\\leftarrow {}$'.format(i) for i in node_names]
    column_labels = ['${} \\rightarrow $'.format(i) for i in node_names]

    norms = kernel_object.get_kernel_norms()
    fig, ax = plt.subplots()

    if rotate_x_labels != 0.:
        # we want clockwise rotation because x-axis is on top
        rotate_x_labels = -rotate_x_labels
        x_label_alignment = 'center'
    else:
        x_label_alignment = 'center'

    if pcolor_kwargs is None:
        pcolor_kwargs = {}

    if norms.min() >= 0:
        pcolor_kwargs.setdefault("cmap", plt.cm.Blues)
    else:
        # In this case we want a diverging colormap centered on 0
        pcolor_kwargs.setdefault("cmap", plt.cm.RdBu)
        max_abs_norm = np.max(np.abs(norms))
        pcolor_kwargs.setdefault("vmin", -max_abs_norm)
        pcolor_kwargs.setdefault("vmax", max_abs_norm)

    heatmap = ax.pcolor(norms, **pcolor_kwargs)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(norms.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(norms.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    fontsize = 5
    ax.set_xticklabels(row_labels, minor=False, fontsize=fontsize, 
                       rotation=rotate_x_labels, ha=x_label_alignment)
    ax.set_yticklabels(column_labels, minor=False, fontsize=fontsize)

    fig.subplots_adjust(bottom=0.1, right=0.9, left=0.2, top=0.8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.5)
    fig.colorbar(heatmap, cax=cax)

    if show:
        plt.show()

    return fig

def draw_mat(args):
    predicates = ['sysbp_low', 'spo2_sao2_low', 'cvp_low', 'svr_low', 'potassium_meql_low', 'sodium_low', 'chloride_low', 'bun_low', 'creatinine_low', 'crp_low', 'rbc_count_low', 'wbc_count_low', 'arterial_ph_low', 'arterial_be_low', 'arterial_lactate_low', 'hco3_low', 'svo2_scvo2_low', 'sysbp_normal', 'spo2_sao2_normal', 'cvp_normal', 'svr_normal', 'potassium_meql_normal', 'sodium_normal', 'chloride_normal', 'bun_normal', 'creatinine_normal', 'crp_normal', 'rbc_count_normal', 'wbc_count_normal', 'arterial_ph_normal', 'arterial_be_normal', 'arterial_lactate_normal', 'hco3_normal', 'svo2_scvo2_normal', 'sysbp_high', 'spo2_sao2_high', 'cvp_high', 'svr_high', 'potassium_meql_high', 'sodium_high', 'chloride_high', 'bun_high', 'creatinine_high', 'crp_high', 'rbc_count_high', 'wbc_count_high', 'arterial_ph_high', 'arterial_be_high', 'arterial_lactate_high', 'hco3_high', 'svo2_scvo2_high', 'real_time_urine_output_low', 'or_colloid', 'or_crystalloid', 'oral_water', 'norepinephrine_norad_levophed', 'epinephrine_adrenaline', 'dobutamine', 'dopamine', 'phenylephrine_neosynephrine', 'milrinone', 'survival']
    mat = load_mat(args.model_name)
    kernel_object = Kernel(mat)
    fig = plot_hawkes_kernel_norms(kernel_object, show=False, pcolor_kwargs=None, node_names=predicates, rotate_x_labels=-90.0)
    fig.savefig('./{}/{}.pdf'.format(args.dataset, args.model_name),dpi =800)

def get_args():
    """Get argument parser.
    Inputs: None
    Returns:
        args: argparse object that contains user-input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="mimic")
    parser.add_argument('--dataset', type=str, default="mimic_1")
    parser.add_argument('--mode', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "preprocess":
        preprocess(args)
    elif args.mode == "mae":
        load_mae(args)
    elif args.mode == "mat":
        load_mat(args)
    