import numpy as np
import pandas as pd
import os.path as osp
from pkg.utils.misc import AverageMeter
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_credit(file_path, n_types):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()  #remove duplicate rows
    #print(df)

    event_seqs = list()
    cust_id_array = df['custAttr1'].unique()
    print("cust num=",len(cust_id_array))

    

    for cust_id in cust_id_array:
        cust_df = df[df['custAttr1']==cust_id]
        cust_list = list()
        for predicateID in range(0,n_types):
            pred_df = cust_df[cust_df['itemid']==predicateID].sort_values(by=['time'])
            state = None
            init_state = None
            for _, row in pred_df.iterrows():
                if state is None:
                    state = row['value']
                    init_state = state
                elif state != row['value']: 
                    if state == init_state: #only record jumping from init_state to other.
                        cust_list.append((row['time'], predicateID)) 
                    state = row['value']
        if len(cust_list)==0:
            continue
        cust_list.sort(key=lambda x:x[0]) #sort by time
        event_seqs.append(cust_list)
    return np.array(event_seqs,dtype=object)

def preprocess():
    print("preprocess start")
    n_types = 9 # num of predicates, including both body and target
    data_path = "/home/fengmingquan/data/train_test_data_credit/"
    test_path = osp.join(data_path,"test_credit_balance_add26.csv")
    train_path = osp.join(data_path,"train_credit_50000_add26.csv")
    
    train_event_seqs = load_credit(train_path, n_types)
    test_event_seqs = load_credit(test_path, n_types)

    print(train_event_seqs)
    print(test_event_seqs)
    raise ValueError

    np.savez_compressed(osp.join(data_path, "credit_cause.npz"),
        train_event_seqs=train_event_seqs,
        test_event_seqs=test_event_seqs,
        n_types=n_types)
    print("preprocess finished")


def test_load():
    data_path = "/home/fengmingquan/data/sepsis_data_three_versions/sepsis_logic/"
    data = np.load(osp.join(data_path, "sepsis_logic_cause.npz"), allow_pickle=True)
    n_types = int(data["n_types"])
    train_event_seqs = data["train_event_seqs"]
    test_event_seqs =  data["test_event_seqs"]
    #print(n_types)
    print(len(test_event_seqs))
    print(len(train_event_seqs))

def load_mat(model_name):
    path = "/home/fengmingquan/data/cause/output/mimic/split_id=0/{}/scores_mat.txt".format(model_name)
    mat = np.genfromtxt(path)
    #print(mat)
    #print(mat.shape)
    return mat

def load_mae(model_name):
    with open("result_{}.pkl".format(model_name),'rb') as f:
        event_seqs_pred, test_event_seqs = pickle.load(f)
    #print(event_seqs_pred)
    #print(test_event_seqs)
    print(model_name)
    calc_mean_absolute_error(test_event_seqs, event_seqs_pred)
    calc_acc(test_event_seqs, event_seqs_pred)

def calc_mean_absolute_error(event_seqs_true, event_seqs_pred):
    """
    Args:
        event_seqs_true (List[List[Tuple]]):
        event_seqs_pred (List[List[Tuple]]):
    """
    target_dict = {'flag': 43, 'mechanical': 1, 'median_dose_vaso': 46, 'max_dose_vaso': 47}
    result_dict = {t:AverageMeter() for t in target_dict.keys() }

    for seq_true, seq_pred in zip(event_seqs_true, event_seqs_pred):
        for t, t_idx in target_dict.items():
            l = [abs(event_true[0]-event_pred[0]) for event_true,event_pred in zip(seq_true,seq_pred) if event_true[1] ==  t_idx]
            if l:
                result_dict[t].update(np.mean(l), len(l))
    
    target = ['flag', 'mechanical', 'median_dose_vaso', 'max_dose_vaso']
    mae = [result_dict[t].avg for t in target]
    print("MAE:", target)
    print("&{:.3f} &{:.3f} &{:.3f} &{:.3f}".format(*mae))

def calc_acc(event_seqs_true, event_seqs_pred):
    """
    Args:
        event_seqs_true (List[List[Tuple]]):
        event_seqs_pred (List[List[Tuple]]):
    """
    target_dict = {'flag': 43, 'mechanical': 1, 'median_dose_vaso': 46, 'max_dose_vaso': 47}
    result_dict = {t:AverageMeter() for t in target_dict.keys() }
    threshold = 1
    for seq_true, seq_pred in zip(event_seqs_true, event_seqs_pred):
        for t, t_idx in target_dict.items():
            l = [abs(event_true[0]-event_pred[0])<1 for event_true,event_pred in zip(seq_true,seq_pred) if event_true[1] ==  t_idx]
            if l:
                result_dict[t].update(np.mean(l), len(l))
    
    target = ['flag', 'mechanical', 'median_dose_vaso', 'max_dose_vaso']
    acc = [result_dict[t].avg for t in target]
    print("ACC:", target)
    print("&{:.3f} &{:.3f} &{:.3f} &{:.3f}".format(*acc))

def predicate_index():
    predicates = ['out_put', 'mechanical', '220277',  'adm_order', 'gender',  'weight', 'height', 'Arterial_BE', 'CO2_mEqL', 'Ionised_Ca', 'Glucose', 'Hb', 'Arterial_lactate', 'paCO2', 'ArterialpH', 'paO2', 'SGPT', 'Albumin', 'SGOT', 'HCO3', 'Direct_bili', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Magnesium', 'Potassium_mEqL', 'Total_protein', 'Sodium', 'Troponin', 'BUN', 'Ht', 'INR', 'Platelets_count', 'PT', 'PTT', 'RBC_count', 'WBC_count','adm_order', 'gender','Total_bili','sofa', 'age','flag','valuenum1','valuenum2','median_dose_vaso','max_dose_vaso']
    #len(predicates)=48
    target = ['flag','mechanical','median_dose_vaso','max_dose_vaso']
    pred_dict = {t : predicates.index(t) for t in target}
    print(pred_dict)

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

def draw_mat(model_name):
    predicates = ['urine\_output', r'\bf{Ventilation}', 'FiO2\_100',  'adm\_order', 'gender',  'weight', 'height', 'Arterial\_BE', 'CO2\_mEqL', 'Ionised\_Ca', 'Glucose', 'Hb', 'Arterial\_lactate', 'paCO2', 'ArterialpH', 'paO2', 'SGPT', 'Albumin', 'SGOT', 'HCO3', 'Direct\_bili', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Magnesium', 'Potassium\_mEqL', 'Total\_protein', 'Sodium', 'Troponin', 'BUN', 'Ht', 'INR', 'Platelets\_count', 'PT', 'PTT', 'RBC\_count', 'WBC\_count','adm\_order', 'gender','Total\_bili','sofa', 'age',r'\bf{Mortality}','valuenum1','valuenum2',r'\bf{Median-Vaso}',r'\bf{Max-Vaso}']
    mat = load_mat(model_name)
    kernel_object = Kernel(mat)
    fig = plot_hawkes_kernel_norms(kernel_object, show=False, pcolor_kwargs=None, node_names=predicates, rotate_x_labels=-90.0)
    fig.savefig('{}.pdf'.format(model_name),dpi =800)


if __name__ == "__main__":
    #load_mimic()
    #load_mhp()
    preprocess()
    #test_load()
    #load_mat()
    #load_mae("ERPP")
    #load_mae("RPPN")
    #load_mae("HExp")
    #predicate_index()
    #draw_mat("HExp")
    #draw_mat("RPPN")
    #draw_mat("ERPP")