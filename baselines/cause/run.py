import numpy as np
import pandas as pd
import os.path as osp

def load_mhp():
    input_path = "/home/fengmingquan/data/cause/input/mhp-1K-10"
    data = np.load(osp.join(input_path, "data.npz"), allow_pickle=True)
    n_types = int(data["n_types"])
    #print(n_types)
    event_seqs = data["event_seqs"]
    print(type(event_seqs))
    split_id = 0
    train_event_seqs = event_seqs[data["train_test_splits"][split_id][0]]
    test_event_seqs = event_seqs[data["train_test_splits"][split_id][1]]
    #print(len(test_event_seqs))
    #print(test_event_seqs[0])
    # [(1094.5910501207484, 4), (1098.1079940088416, 7), (1100.4624087040852, 1), (1101.3163237708202, 1), (1105.8506944816406, 1), (1106.1821513888342, 0), (1106.1918308538147, 8)]

def load_mimic(file_path, n_types):
    df = pd.read_csv(file_path)
    #print(df)

    event_seqs = list()
    patient_id_array = df['patient_id'].unique()

    for patient_id in patient_id_array:
        patient_df = df[df['patient_id']==patient_id]
        patient_list = list()
        for predicateID in range(1,n_types+1):
            pred_df = patient_df[patient_df['predicateID']==predicateID].sort_values(by=['time'])
            state = None
            for _, row in pred_df.iterrows():
                if state is None:
                    state = row['value']
                elif state != row['value']:
                    patient_list.append((row['time'], predicateID-1)) 
                    state = row['value']
        patient_list.sort(key=lambda x:x[0]) #sort by time
        event_seqs.append(patient_list)
    return np.array(event_seqs,dtype=object)

def preprocess():
    n_types = 48 # num of predicates, including both body and target
    data_path = "/home/fengmingquan/data/sepsis_data_three_versions/sepsis_logic/"
    test_path = osp.join(data_path,"sepsis_logic_test.csv")
    train_path = osp.join(data_path,"sepsis_logic_train.csv")
    
    train_event_seqs = load_mimic(train_path, n_types)
    test_event_seqs = load_mimic(test_path, n_types)
    np.savez_compressed(osp.join(data_path, "sepsis_logic.npz"),
        train_event_seqs=train_event_seqs,
        test_event_seqs=test_event_seqs,
        n_types=n_types)


def test_load():
    data_path = "/home/fengmingquan/data/sepsis_data_three_versions/sepsis_logic/"
    data = np.load(osp.join(data_path, "sepsis_logic.npz"), allow_pickle=True)
    n_types = int(data["n_types"])
    train_event_seqs = data["train_event_seqs"]
    test_event_seqs =  data["test_event_seqs"]
    #print(n_types)
    #print(train_event_seqs)
    #print(test_event_seqs)

def load_mat():
    path = "/home/fengmingquan/data/cause/output/mimic/split_id=0/HExp/scores_mat.txt"
    mat = np.genfromtxt(path)
    print(mat)
    print(mat.shape)

if __name__ == "__main__":
    #load_mimic()
    #load_mhp()
    preprocess()
    #test_load()
    #load_mat()