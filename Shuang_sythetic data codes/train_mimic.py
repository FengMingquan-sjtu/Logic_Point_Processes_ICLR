import datetime
import os
import argparse
import pickle 

import numpy as np
import torch
import pandas

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer, get_data

def get_model(model_name, dataset_name):
    lab_preds = list(range(34)) #[0-33] 
    output_preds = [34]
    input_preds = [35,36,37]
    drug_preds = [38,39,40,41,42,43]
    survival = [44]
    
    if model_name == "mimic":
        head_predicate_idx = output_preds[0] #urine 
        model = Logic_Learning_Model(head_predicate_idx=[head_predicate_idx,])
        
        model.predicate_notation = ['sysbp_low', 'spo2_sao2_low', 'cvp_low', 'svr_low', 'potassium_meql_low', 'sodium_low', 'chloride_low', 'bun_low', 'creatinine_low', 'crp_low', 'rbc_count_low', 'wbc_count_low', 'arterial_ph_low', 'arterial_be_low', 'arterial_lactate_low', 'hco3_low', 'svo2_scvo2_low', 'sysbp_high', 'spo2_sao2_high', 'cvp_high', 'svr_high', 'potassium_meql_high', 'sodium_high', 'chloride_high', 'bun_high', 'creatinine_high', 'crp_high', 'rbc_count_high', 'wbc_count_high', 'arterial_ph_high', 'arterial_be_high', 'arterial_lactate_high', 'hco3_high', 'svo2_scvo2_high', 'real_time_urine_output_low', 'or_colloid', 'or_crystalloid', 'oral_water', 'norepinephrine_norad_levophed', 'epinephrine_adrenaline', 'dobutamine', 'dopamine', 'phenylephrine_neosynephrine', 'milrinone', 'survival']
        model.predicate_set= list(range(len(model.predicate_notation))) # the set of all meaningful predicates
        model.body_pred_set = lab_preds + input_preds + drug_preds #only learn lab-->urine
        model.instant_pred_set = input_preds + drug_preds #input/drugs are instant.
        model.max_rule_body_length = 2
        model.max_num_rule = 20
        model.weight_threshold = 0.2
        model.strict_weight_threshold= 0.5
        model.gain_threshold = 0.1
        model.low_grad_threshold = 0.05
        

        model.time_window = 24 * 5
        model.Time_tolerance = 10
        model.decay_rate = 0.01
        model.batch_size = 64
        model.integral_resolution = 10
    
    
    return model

def convert_time(date_time_pair):
    date, time = date_time_pair.split()
    year, month, day = date.split("-")
    hour, minute, second = time.split(":")
    float_hour = int(hour) + int(minute)/60 + int(second)/3600
    float_hour += int(day)*24 + int(month)*30*24 
    year = int(year)
    if year%4==0 and year%100 !=0:
        float_hour += (year-2100)*366*24
    else:
        float_hour += (year-2100)*365*24
    return float_hour

def process_raw_data(input_file, output_file):
    df = pandas.read_csv("./data/"+input_file)
    lab_preds = ["sysbp", "spo2_sao2", "cvp", "svr", "potassium_meql", "sodium", "chloride", "bun", "creatinine", "crp", "rbc_count",  "wbc_count", "arterial_ph", "arterial_be", "arterial_lactate", "hco3", "svo2_scvo2"]
    lab_preds = [p+"_low" for p in lab_preds] + [p+"_normal" for p in lab_preds] + [p+"_high" for p in lab_preds]
    output_preds = ["real_time_urine_output_low"] #only low?
    input_preds = ["or_colloid", "or_crystalloid", "oral_water"] #only one pred, no low,high,normal
    drug_preds = ["norepinephrine_norad_levophed", "epinephrine_adrenaline", "dobutamine", 'dopamine', 'phenylephrine_neosynephrine', 'milrinone']#only one pred, no low,high,normal
    pred_list = lab_preds + output_preds + input_preds + drug_preds
    treat_list = input_preds + drug_preds
    instant_list = treat_list #treatments are all instant.

    #{0: 'sysbp_low', 1: 'spo2_sao2_low', 2: 'cvp_low', 3: 'svr_low', 4: 'potassium_meql_low', 5: 'sodium_low', 6: 'chloride_low', 7: 'bun_low', 8: 'creatinine_low', 9: 'crp_low', 10: 'rbc_count_low', 11: 'wbc_count_low', 12: 'arterial_ph_low', 13: 'arterial_be_low', 14: 'arterial_lactate_low', 15: 'hco3_low', 16: 'svo2_scvo2_low', 17: 'sysbp_normal', 18: 'spo2_sao2_normal', 19: 'cvp_normal', 20: 'svr_normal', 21: 'potassium_meql_normal', 22: 'sodium_normal', 23: 'chloride_normal', 24: 'bun_normal', 25: 'creatinine_normal', 26: 'crp_normal', 27: 'rbc_count_normal', 28: 'wbc_count_normal', 29: 'arterial_ph_normal', 30: 'arterial_be_normal', 31: 'arterial_lactate_normal', 32: 'hco3_normal', 33: 'svo2_scvo2_normal', 34: 'sysbp_high', 35: 'spo2_sao2_high', 36: 'cvp_high', 37: 'svr_high', 38: 'potassium_meql_high', 39: 'sodium_high', 40: 'chloride_high', 41: 'bun_high', 42: 'creatinine_high', 43: 'crp_high', 44: 'rbc_count_high', 45: 'wbc_count_high', 46: 'arterial_ph_high', 47: 'arterial_be_high', 48: 'arterial_lactate_high', 49: 'hco3_high', 50: 'svo2_scvo2_high', 51: 'real_time_urine_output_low', 52: 'or_colloid', 53: 'or_crystalloid', 54: 'oral_water', 55: 'norepinephrine_norad_levophed', 56: 'epinephrine_adrenaline', 57: 'dobutamine', 58: 'dopamine', 59: 'phenylephrine_neosynephrine', 60: 'milrinone', 61: 'survival'}
    #[0-50]: lab_preds 
    #[51]: real_time_urine_output_low
    #[52, 53, 54]: input 
    #[55-60] :drug 
    #[61]: survival

    hour = df['charttime'].apply(convert_time)
    hour.rename("hour" ,inplace=True)

    columns = list()
    columns.append(df["hadm_id"])
    columns.append(hour)
    columns.append(df['item_state'])

    for p in pred_list:
        column = (df["item_name"] == p)
        column.rename(p ,inplace=True)
        columns.append(column)
        
    survival = (df["expire_flag"] == "S")
    survival.rename("survival" ,inplace=True)
    columns.append(survival)
    selected_data = pandas.concat(objs=columns, axis=1)
    dict_data = convert_from_df_to_dict(pred_list,instant_list, treat_list, selected_data)
    np.save("./data/"+output_file, dict_data)
    

def convert_from_df_to_dict(pred_list, instant_list, treat_list, selected_df):
    # convert to logic-learning data, 1 sample for 1 patient.
    # logic-learning input data format :
    # {0: {'time': [0, 3.420397849789993, 6.341931048876761, 7.02828641970859, 8.16064604149825, 9.504766128153767], 'state': [1, 0, 1, 0, 1, 0]}, 1: {'time': [0, 0.9831991978572895, 1.4199066113445857, 1.6292485101191285, 2.096475266132198, 4.005069218069917, 5.767948388984466, 5.77743637565431, 6.852427239152188, 7.8930935707342424, 8.589873100390394, 8.820903625226093, 9.048162949232953, 9.342080514689219], 'state': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}, 2: {'time': [0, 1.1058850237811462, 1.271081383531531, 2.7239366123865625, 2.855376376115736, 3.4611879524020916, 3.674093142880717, 4.3536109404218095, 4.531223527024521, 4.951502883753997, 5.096495412716558, 6.194746446461735, 9.743255577798488], 'state': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}, 3: {'time': [0, 9.374137055918649], 'state': [1, 0]}, 4: {'time': [0], 'state': [1]}, 5: {'time': [0, 0.01395797611577883, 1.4515718053899762, 1.5554166263608424, 3.2631901045050062, 3.377071446493159, 3.3997887424994264, 3.416948377319663, 6.879589474535199, 8.348522758390544, 9.384507895416254], 'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}}
    
    data = dict()
    name_dict = dict()
    pred_list_ = pred_list + ['survival']
    idx = 0
    for group in selected_df.groupby(by=["hadm_id"]):
        sample = dict() 
        patient_df = group[1]
        start_time = patient_df['hour'].tolist()[0]
        

        
        patient_df = patient_df[patient_df['hour'] > start_time]
        patient_df.loc[:, 'hour'] = patient_df['hour'] - start_time
        
        for pid, pred in enumerate(pred_list_):
            if pred == "survival":
                time = [0, ]
                state = [1, ]
                if len(patient_df['hour'])>0:
                    time.append(patient_df['hour'].tolist()[-1])
                    state.append(int(patient_df["survival"].tolist()[-1]))
            else:
                state_ = patient_df['item_state'][patient_df[pred]==True].tolist()
                time_ = patient_df['hour'][patient_df[pred]==True].tolist()
                if len(time_) == 0:
                    state = [0,]
                    time = [0,]
                else:
                    if pred in instant_list:
                        time = time_
                        state = [1] * len(time)
                    else:
                        state = [state_[0], ]
                        time = [time_[0], ]
                        for i in range(1, len(time_)):
                            if state_[i] != state_[i-1]:
                                time.append(time_[i])
                                state.append(state_[i])
            sample[pid] = {"time":time, "state":state}
        
        data[idx] = sample
        idx+=1
        #break
    name_dict = dict(enumerate(pred_list_))
    print(name_dict)
    return data




def fit(model_name, dataset_name, num_sample, worker_num=8, num_iter=5, use_cp=False, rule_set_str = None, algorithm="BFS"):
    time_ = str(datetime.datetime.now())
    print("Start time is", time_, flush=1)

    if not os.path.exists("./model"):
        os.makedirs("./model")
    
    #get model
    model = get_model(model_name, dataset_name)

    #set initial rules if required
    if rule_set_str:
        set_rule(model, rule_set_str)
        

    #get data
    dataset,num_sample =  get_data(dataset_name, num_sample)

    #set model hyper params
    model.batch_size_grad = num_sample  #use all samples for grad
    model.batch_size_cp = num_sample
    model.num_iter = num_iter
    model.use_cp = use_cp
    model.worker_num = worker_num
    

    if algorithm == "DFS":
        with Timer("DFS") as t:
            model.DFS(model.head_predicate_set[0], dataset, tag="DFS_{}_{}".format(dataset_name, time_))
    elif algorithm == "BFS":
        with Timer("BFS") as t:
            model.BFS(model.head_predicate_set[0], dataset, tag="BFS_{}_{}".format(dataset_name, time_))
    
    print("Finish time is", datetime.datetime.now())
 

def run_expriment_group(args):
    #downtown districts
    #DFS
    fit(model_name=args.model, dataset_name=args.dataset, num_sample=-1, worker_num=args.worker, num_iter=6, algorithm="DFS")
    #BFS
    fit(model_name=args.model, dataset_name=args.dataset, num_sample=-1, worker_num=args.worker, num_iter=6, algorithm="BFS")


def run_preprocess():
    process_raw_data(input_file="mimic_dataset_v3.csv", output_file="mimic_1.npy")
    process_raw_data(input_file="second_mimic_dataset_v3.csv", output_file="mimic_2.npy")

def get_args():
    """Get argument parser.
    Inputs: None
    Returns:
        args: argparse object that contains user-input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mimic")
    parser.add_argument('--dataset', type=str, default="mimic_1")
    parser.add_argument('--worker', type=int, default=12)
    parser.add_argument('--print_log', action="store_true")
    
    args = parser.parse_args()
    return args

def dataset_stat(dataset):
    dataset,num_sample =  get_data(dataset_name=dataset, num_sample=-1)
    #print("num sample=", num_sample)
    seq_len = list()
    seq_size= list()
    for sample in dataset.values():
        l = 0
        s = 0
        for data in sample.values():
            if data["time"]:
                l = max(l, data["time"][-1])
                s = max(s, len(data["time"]))
        seq_len.append(l)
        seq_size.append(s)
    seq_len = np.array(seq_len)
    seq_size = np.array(seq_size)
    print("seq length mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_len), np.std(seq_len)))
    print("seq size mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_size), np.std(seq_size)))


def test(dataset_name, model_file):
    dataset,num_sample =  get_data(dataset_name=dataset_name, num_sample=10)
    with open("./model/"+model_file, "rb") as f:
        model = pickle.load(f)
        model.instant_pred_set = list()
    model.generate_target(head_predicate_idx=34, dataset=dataset)


if __name__ == "__main__":
    #run_preprocess()
    #process_raw_data(input_file="mimic_dataset_v3.csv", output_file="mimic_0.npy")

    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78
    #args = get_args()
    #if not args.print_log:
    #    redirect_log_file()
    #run_expriment_group(args)
    #dataset_stat(dataset=args.dataset)
    #FS_mimic_1.pkl")
    #data, num_sample = get_data(dataset_name=args.dataset, num_sample=100)
    #for k,v in data.items():
    #    print(v)
