import pandas
import numpy as np

import datetime
import os

import numpy as np
import torch

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer, get_data, get_args

def get_model(model_name, dataset_name):
    if model_name == "mimic":
        model = Logic_Learning_Model(head_predicate_idx=[34])
        
        model.predicate_notation = ['sysbp_low', 'spo2_sao2_low', 'cvp_low', 'svr_low', 'potassium_meql_low', 'sodium_low', 'chloride_low', 'bun_low', 'creatinine_low', 'crp_low', 'rbc_count_low', 'wbc_count_low', 'arterial_ph_low', 'arterial_be_low', 'arterial_lactate_low', 'hco3_low', 'svo2_scvo2_low', 'sysbp_high', 'spo2_sao2_high', 'cvp_high', 'svr_high', 'potassium_meql_high', 'sodium_high', 'chloride_high', 'bun_high', 'creatinine_high', 'crp_high', 'rbc_count_high', 'wbc_count_high', 'arterial_ph_high', 'arterial_be_high', 'arterial_lactate_high', 'hco3_high', 'svo2_scvo2_high', 'real_time_urine_output_low', 'or_colloid', 'or_crystalloid', 'oral_water', 'norepinephrine_norad_levophed', 'epinephrine_adrenaline', 'dobutamine', 'dopamine', 'phenylephrine_neosynephrine', 'milrinone', 'survival']
        model.predicate_set= list(range(len(model.predicate_notation))) # the set of all meaningful predicates
        model.body_pred_set = list(range(33)) #only learn lab-->urine
        model.max_rule_body_length = 3
        model.max_num_rule = 20
        model.weight_threshold = 0.0001
        model.strict_weight_threshold= 0.0005
        model.gain_threshold = 0.0001
        model.low_grad_threshold = 0.0001
        model.batch_size_grad = 200

        model.time_window = 24 * 5
        model.Time_tolerance = 10
        model.decay_rate = 0.01
        model.batch_size = 64
        model.integral_resolution = 1
    
    
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
    lab_preds = [p+"_low" for p in lab_preds] + [p+"_high" for p in lab_preds]
    output_preds = ["real_time_urine_output_low"] #only low?
    input_preds = ["or_colloid", "or_crystalloid", "oral_water"] #only one pred, no low,high,normal
    drug_preds = ["norepinephrine_norad_levophed", "epinephrine_adrenaline", "dobutamine", 'dopamine', 'phenylephrine_neosynephrine', 'milrinone']#only one pred, no low,high,normal
    pred_list = lab_preds + output_preds + input_preds + drug_preds
    treat_list = input_preds + drug_preds

    #{0: 'sysbp_low', 1: 'spo2_sao2_low', 2: 'cvp_low', 3: 'svr_low', 4: 'potassium_meql_low', 5: 'sodium_low', 6: 'chloride_low', 7: 'bun_low', 8: 'creatinine_low', 9: 'crp_low', 10: 'rbc_count_low', 11: 'wbc_count_low', 12: 'arterial_ph_low', 13: 'arterial_be_low', 14: 'arterial_lactate_low', 15: 'hco3_low', 16: 'svo2_scvo2_low', 17: 'sysbp_high', 18: 'spo2_sao2_high', 19: 'cvp_high', 20: 'svr_high', 21: 'potassium_meql_high', 22: 'sodium_high', 23: 'chloride_high', 24: 'bun_high', 25: 'creatinine_high', 26: 'crp_high', 27: 'rbc_count_high', 28: 'wbc_count_high', 29: 'arterial_ph_high', 30: 'arterial_be_high', 31: 'arterial_lactate_high', 32: 'hco3_high', 33: 'svo2_scvo2_high', 34: 'real_time_urine_output_low', 35: 'or_colloid', 36: 'or_crystalloid', 37: 'oral_water', 38: 'norepinephrine_norad_levophed', 39: 'epinephrine_adrenaline', 40: 'dobutamine', 41: 'dopamine', 42: 'phenylephrine_neosynephrine', 43: 'milrinone', 44: 'survival'}
    #[0-33]: lab_preds 
    #[34]: real_time_urine_output_low
    #[35,36,37]: input 
    #[38-43] :drug 
    #[44]: survival

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
    dict_data = convert_from_df_to_dict(pred_list, treat_list, selected_data)
    np.save("./data/"+output_file, dict_data)
    

def convert_from_df_to_dict(pred_list, treat_list, selected_df):
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
        treat_time = 1e10
        for treat in treat_list:
            t = patient_df['hour'][patient_df[treat]==True].tolist()
            if len(t)>0:
                t = t[0]
            else:
                t = 1e10
            treat_time = min(treat_time, t)
        if treat_time == 1e10:
            #ignore patients without treatment
            continue
        #print("old start time=",start_time)
        start_time = max(start_time, treat_time-5)
        #print("new start time=",start_time)
        
        
        new_hour = patient_df['hour'] - start_time
        patient_df.loc[:, 'hour'] = new_hour
        
        for pid, pred in enumerate(pred_list_):
            if pred == "survival":
                time = [0, ]
                state = [1, ]
                time.append(patient_df['hour'].tolist()[-1])
                state.append(int(patient_df["survival"].tolist()[-1]))
            else:
                state_ = patient_df['item_state'][patient_df[pred]==True].tolist()
                time_ = patient_df['hour'][patient_df[pred]==True].tolist()
                if len(time_) == 0:
                    state = [0,]
                    time = [0,]
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
    print("Start time is", datetime.datetime.now(),flush=1)

    if not os.path.exists("./model"):
        os.makedirs("./model")
    
    #get model
    model = get_model(model_name, dataset_name)

    #set initial rules if required
    if rule_set_str:
        set_rule(model, rule_set_str)
        

    #get data
    dataset =  get_data(dataset_name, num_sample)

    #set model hyper params
    model.batch_size_grad = num_sample #use all sample for grad
    model.batch_size_cp = num_sample
    model.num_iter = num_iter
    model.use_cp = use_cp
    model.worker_num = worker_num
    
    
    
    


    if algorithm == "DFS":
        with Timer("DFS") as t:
            model.DFS(model.head_predicate_set[0], dataset, tag="DFS_"+dataset_name)
    elif algorithm == "BFS":
        with Timer("BFS") as t:
            model.BFS(model.head_predicate_set[0], dataset, tag="BFS_"+dataset_name)
    
    print("Finish time is", datetime.datetime.now())
 

def run_expriment_group(dataset_name):
    #downtown districts
    #DFS
    fit(model_name="mimic", dataset_name=dataset_name, num_sample=2000, worker_num=12, num_iter=12, algorithm="DFS")
    #BFS
    fit(model_name="mimic", dataset_name=dataset_name, num_sample=2000, worker_num=12, num_iter=12, algorithm="BFS")


def run_preprocess():
    process_raw_data(input_file="mimic_dataset_v3.csv", output_file="mimic_1.npy")
    process_raw_data(input_file="second_mimic_dataset_v3.csv", output_file="mimic_2.npy")

if __name__ == "__main__":
    #run_preprocess()
    #redirect_log_file()

    #torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78
    #args = get_args()
    run_expriment_group(dataset_name="mimic_1")