import os.path as osp

import numpy as np
import pandas as pd
import pickle
from logic_learning import Logic_Learning_Model

def load_mimic(file_path, n_types):
    df = pd.read_csv(file_path)
    event_seqs = dict()
    patient_id_array = df['patient_id'].unique()

    for patient_id in patient_id_array:
        patient_df = df[df['patient_id']==patient_id]
        patient_dict = dict()
        for predicateID in range(1,n_types+1):
            predicate_dict = {'time':[], 'state':[]}
            pred_df = patient_df[patient_df['predicateID']==predicateID]
            state = None
            for _, row in pred_df.iterrows():
                if state is None:
                    state = row['value']
                    predicate_dict['time'].append(0.0)
                    predicate_dict['state'].append(int(state))
                elif state != row['value']: 
                    state = row['value']
                    predicate_dict['time'].append(row['time'])
                    predicate_dict['state'].append(int(state))
            patient_dict[predicateID] = predicate_dict
                    
        event_seqs[patient_id] = patient_dict

    return event_seqs

def preprocess():
    print("preprocess start")
    n_types = 48 # num of predicates, including both body and target
    data_path = "/home/fengmingquan/data/sepsis_data_three_versions/sepsis_logic/"
    test_path = osp.join(data_path,"sepsis_logic_test.csv")
    train_path = osp.join(data_path,"sepsis_logic_train.csv")
    
    train_event_seqs = load_mimic(train_path, n_types)
    test_event_seqs = load_mimic(test_path, n_types)
    #print(test_event_seqs[204223])
    with open(osp.join(data_path, "sepsis_logic_pp.pkl"), 'wb') as f:
        pickle.dump((train_event_seqs,test_event_seqs), f)

    print("preprocess finished")

def load_data():
    data_path = "/home/fengmingquan/data/sepsis_data_three_versions/sepsis_logic/"
    use_small = True 
    if use_small:
        file_name = "sepsis_logic_pp_small.pkl"
    else:
        file_name = "sepsis_logic_pp.pkl"
    with open(osp.join(data_path, file_name), 'rb') as f:
        train_event_seqs, test_event_seqs =  pickle.load(f)
    return train_event_seqs, test_event_seqs

def train():
    n_types = 48
    target_dict = {'flag': 44, 'mechanical': 2, 'median_dose_vaso': 47, 'max_dose_vaso': 48}
    predicate_set = list(range(1,1+n_types))
    predicate_notation = ['out_put', 'mechanical', '220277',  'adm_order', 'gender',  'weight', 'height', 'Arterial_BE', 'CO2_mEqL', 'Ionised_Ca', 'Glucose', 'Hb', 'Arterial_lactate', 'paCO2', 'ArterialpH', 'paO2', 'SGPT', 'Albumin', 'SGOT', 'HCO3', 'Direct_bili', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Magnesium', 'Potassium_mEqL', 'Total_protein', 'Sodium', 'Troponin', 'BUN', 'Ht', 'INR', 'Platelets_count', 'PT', 'PTT', 'RBC_count', 'WBC_count','adm_order', 'gender','Total_bili','sofa', 'age','flag','valuenum1','valuenum2','median_dose_vaso','max_dose_vaso']
    model = Logic_Learning_Model(head_predicate_idx = list(target_dict.values()))
    model.predicate_set = predicate_set
    model.predicate_notation = predicate_notation
    model.batch_size = 32
    T_max = 15
    train_data, test_data = load_data()
    #print(type(train_data)) -->dict
    for idx in target_dict.values():
        model.search_algorithm(head_predicate_idx=idx, dataset=train_data, T_max=T_max)
    model.print_rule()
    with open("mimic_model.pkl",'wb') as f:
        pickle.dump(model, f)
    

if __name__ == "__main__":
    head_predicate_idx = [3,4]
    #model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    num_samples = 5000
    T_max = 10
    #dataset = np.load('data.npy', allow_pickle='TRUE').item()

    #small_dataset = {i:dataset[i] for i in range(1)}
    #print(small_dataset)
    #preprocess()
    #load_data()
    train()
