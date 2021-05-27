import datetime
import os
import argparse
import pickle 

import numpy as np
import torch
import pandas

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer, get_data

def get_model(model_name, dataset_name, head_predicate_idx):
    lab_preds = list(range(51)) #[0-50] 
    output_preds = [51]
    input_preds = [52,53,54]
    drug_preds = list(range(55,61))
    survival = [61]
    
    if model_name == "mimic":
        model = Logic_Learning_Model(head_predicate_idx=[head_predicate_idx,])
        
        model.predicate_notation = ['sysbp_low', 'spo2_sao2_low', 'cvp_low', 'svr_low', 'potassium_meql_low', 'sodium_low', 'chloride_low', 'bun_low', 'creatinine_low', 'crp_low', 'rbc_count_low', 'wbc_count_low', 'arterial_ph_low', 'arterial_be_low', 'arterial_lactate_low', 'hco3_low', 'svo2_scvo2_low', 'sysbp_normal', 'spo2_sao2_normal', 'cvp_normal', 'svr_normal', 'potassium_meql_normal', 'sodium_normal', 'chloride_normal', 'bun_normal', 'creatinine_normal', 'crp_normal', 'rbc_count_normal', 'wbc_count_normal', 'arterial_ph_normal', 'arterial_be_normal', 'arterial_lactate_normal', 'hco3_normal', 'svo2_scvo2_normal', 'sysbp_high', 'spo2_sao2_high', 'cvp_high', 'svr_high', 'potassium_meql_high', 'sodium_high', 'chloride_high', 'bun_high', 'creatinine_high', 'crp_high', 'rbc_count_high', 'wbc_count_high', 'arterial_ph_high', 'arterial_be_high', 'arterial_lactate_high', 'hco3_high', 'svo2_scvo2_high', 'real_time_urine_output_low', 'or_colloid', 'or_crystalloid', 'oral_water', 'norepinephrine_norad_levophed', 'epinephrine_adrenaline', 'dobutamine', 'dopamine', 'phenylephrine_neosynephrine', 'milrinone', 'survival']
        model.predicate_set= list(range(len(model.predicate_notation))) # the set of all meaningful predicates
        model.survival_pred_set = survival
        if head_predicate_idx == 51:
            model.body_pred_set = lab_preds + output_preds + input_preds + drug_preds #only learn lab-->urine
            #rule template
            model.body_pred_set_first_part = lab_preds + output_preds
            model.body_pred_set_second_part = input_preds + drug_preds
            #model.deleted_rules.add("sysbp_normal --> real_time_urine_output_low , sysbp_normal BEFORE real_time_urine_output_low")
            #model.deleted_rules.add("sysbp_normal --> real_time_urine_output_low , sysbp_normal EQUAL real_time_urine_output_low")
            #model.deleted_rules.add("sysbp_normal --> Not real_time_urine_output_low , sysbp_normal BEFORE Not real_time_urine_output_low")
            #model.deleted_rules.add("sysbp_normal --> Not real_time_urine_output_low , sysbp_normal EQUAL Not real_time_urine_output_low")

        elif head_predicate_idx == 61:
            model.body_pred_set = lab_preds + output_preds + input_preds + drug_preds 
            #rule template
            model.body_pred_set_first_part = lab_preds + output_preds
            model.body_pred_set_second_part = input_preds + drug_preds
            model.prior_preds.append(51) #add urine as prior.
        else:
            raise ValueError


        model.instant_pred_set = input_preds + drug_preds #input/drugs are instant.
        model.max_rule_body_length = 3
        model.max_num_rule = 20
        model.weight_threshold = 0.01
        model.strict_weight_threshold= 0.5
        model.gain_threshold = 0.005
        model.low_grad_threshold = 0.005
        model.learning_rate = 0.0001
        model.base_lr = 0.00005
        model.weight_lr = 0.02
        model.use_decay = True
        model.use_2_bases = True
        model.init_base = 0.1
        model.opt_worker_num = 16
        
        
        if dataset_name == "mimic_3_clip_scaled":
            scale = 10/55
            model.best_N = 2
            model.integral_resolution = 1 * scale
            model.time_window = 36 * scale
            model.Time_tolerance = 4 * scale
        elif dataset_name == "mimic_3_scaled":
            scale = 10/400
            model.best_N = 1
            model.integral_resolution = 10 * scale
            model.time_window = 200 * scale
            model.Time_tolerance = 24 * scale

        model.scale = scale
        model.batch_size = 64
        model.decay_rate = 0.05
    
    
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
        # [0-16] low
        # [17-33] normal
        # [33-50] high
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
                if len(patient_df["survival"])>1:
                    is_survival = int(patient_df["survival"].tolist()[-1])
                    if is_survival == 0:
                        time.append(patient_df['hour'].tolist()[-1])
                        state.append(is_survival)
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
        print(idx)
        #break
    name_dict = dict(enumerate(pred_list_))
    print(name_dict)
    return data




def fit(model_name, dataset_name, head_predicate_idx, num_sample, worker_num=8, num_iter=5, use_cp=False, rule_set_str = None, algorithm="BFS"):
    time_ = str(datetime.datetime.now())
    print("Start time is", time_, flush=1)

    if not os.path.exists("./model"):
        os.makedirs("./model")
    
    #get model
    model = get_model(model_name, dataset_name, head_predicate_idx)

    #set initial rules if required
    if rule_set_str:
        set_rule(model, rule_set_str)
        

    #get data
    dataset,num_sample =  get_data(dataset_name, num_sample)
    training_dataset = {i: dataset[i] for i in range(int(num_sample*0.8))}
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}

    #set model hyper params
    model.batch_size_grad = num_sample // 2  #use 1/2 samples for grad
    model.batch_size_cp = num_sample
    model.num_iter = num_iter
    model.use_cp = use_cp
    model.worker_num = worker_num
    

    if algorithm == "DFS":
        with Timer("DFS") as t:
            model.DFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag="DFS_{}_{}".format(dataset_name, time_))
    elif algorithm == "BFS":
        with Timer("BFS") as t:
            model.BFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag="BFS_{}_{}".format(dataset_name, time_))
    elif algorithm == "Hawkes":
        model.Hawkes(model.head_predicate_set[0], training_dataset, testing_dataset, tag="{}_{}_{}".format(algorithm, dataset_name, time_))
    elif algorithm == "Hawkes_Rev":
        print(training_dataset[0][51])
        for data in [training_dataset,testing_dataset]:
            for sid,d in data.items():
                data[sid][51]["state"] = list(1-np.array(data[sid][51]["state"])) 
        print(training_dataset[0][51])

        model.Hawkes(model.head_predicate_set[0], training_dataset, testing_dataset, tag="{}_{}_{}".format(algorithm, dataset_name, time_))
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
    parser.add_argument('--dataset', type=str, default="mimic_1_scaled")
    
    parser.add_argument('--head_predicate_idx', type=int, default=51, help="51-urine, 61-survival")
    parser.add_argument('--worker', type=int, default=12)
    parser.add_argument('--print_log', action="store_true")
    
    args = parser.parse_args()
    return args

def dataset_stat(dataset):
    dataset,num_sample =  get_data(dataset_name=dataset, num_sample=-1)
    seq_len = list()
    seq_size= list()
    seq_tot_size = list()
    target_l_list = list()
    target_s_list = list()
    for sample in dataset.values():
        l = 0
        s = 0
        tot_s = 0
        for pid, data in sample.items():
            if data["time"]:
                l = max(l, data["time"][-1])

                s = max(s, len(data["time"]))
                tot_s += len(data["time"])
                if pid == 51: #target: urine
                    target_l = data["time"][-1] - data["time"][0]
                    target_s = len(data["time"])
                    target_l_list.append(target_l)
                    target_s_list.append(target_s)


            else:
                print("empty sample")
        seq_len.append(l)
        seq_size.append(s)
        seq_tot_size.append(tot_s)
    seq_len = np.array(seq_len)
    seq_size = np.array(seq_size)
    seq_tot_size = np.array(seq_tot_size)
    target_l_arr = np.array(target_l_list)
    target_s_arr = np.array(target_s_list)

    print("seq length mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_len), np.std(seq_len)))
    print("seq size mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_size), np.std(seq_size)))
    print("target length mean = {:.4f}, std = {:.4f}.".format(np.mean(target_l_arr), np.std(target_l_arr)))
    print("target size mean = {:.4f}, std = {:.4f}.".format(np.mean(target_s_arr), np.std(target_s_arr)))
    print("seq total size mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_tot_size), np.std(seq_tot_size)))



def test(dataset_name, model_file):
    dataset,num_sample =  get_data(dataset_name=dataset_name, num_sample=10)
    with open("./model/"+model_file, "rb") as f:
        model = pickle.load(f)
        model.instant_pred_set = list()
    model.generate_target(head_predicate_idx=34, dataset=dataset)

def rescale_data(input_file, output_file, scale):
    data = np.load("./data/"+input_file, allow_pickle='TRUE').item()
    for sample in data.values():
        for d in sample.values():
            d["time"] = list(np.array(d["time"]) * scale)
    np.save("./data/"+output_file, data)

def get_state(cur_time, pid, history):
    instant_pred_set = [52, 53, 54, 55, 56, 57, 58, 59, 60]
    t = np.array(history[pid]['time'])
    s = np.array(history[pid]['state'])
    if pid in instant_pred_set:
        #instant pred state is always zero.
        return None
    else:
        if len(t) == 0:
            return None
        else:
            idx = np.sum(cur_time > t) - 1
            if idx < 0:
                return None
            else:
                return s[idx]


def clip_data(input_file, output_file, head_predicate_idx, horizon):
    def clip(start_time, end_time, sample):
        new_sample = dict()
        for pid,d in sample.items():
            t = np.array(d["time"])
            s = np.array(d["state"])
            init_state = get_state(start_time, pid, sample)
            idx = (t>=start_time) * (t<=end_time)
            t_ = list(t[idx] - start_time)
            s_ = list(s[idx])
            if (not (init_state is None)) and (init_state != 0):
                t_ = [0]+t_
                s_ = [init_state]+s_
            

            if len(t_) == 0:
                t_ = [0,]
                s_ = [0,]
            if pid == head_predicate_idx and s_[0] == 0:
                s_ = s_[1:]
                t_ = t_[1:]
            if (pid <= 16 or (33 <= pid and pid <= 50)) and s_[0] == 0:
                # abnormal preds, init values
                s_ = [0] + s_[1:]
                t_ = [0] + t_[1:]
            if (17 <= pid and pid <= 33) and s_[0] == 1:
                #normal preds, init values
                s_ = [1] + s_[1:]
                t_ = [0] + t_[1:]
                
            new_sample[pid] = {"time": t_, "state":s_}
        return new_sample
    data = np.load("./data/"+input_file, allow_pickle='TRUE').item()
    new_data = list()
    
    for sample in data.values():
        for idx, s in enumerate(sample[head_predicate_idx]["state"]):
            if s == 1:
                t = sample[head_predicate_idx]["time"][idx]
                new_data.append(clip(t-horizon, t+horizon, sample))
                
        
        
    new_data = dict(enumerate(new_data))   
    np.save("./data/"+output_file, new_data)
    

def retrain(model_file_name, delete_formula_idx_list, dataset_name, head_predicate_idx,  num_sample, worker_num, num_iter, algorithm):
    time_ = str(datetime.datetime.now())
    print("Start time is", time_, flush=1)
    with open("./model/"+model_file_name, "rb") as f:
        model = pickle.load(f)
    model.deleted_rules = set() # clear banned rules.
    model.weight_lr = 0.02
    model.opt_worker_num = 16
    model.base_lr = 0.00005
    model.delete_rules(head_predicate_idx, delete_formula_idx_list)

    #get data
    dataset,num_sample =  get_data(dataset_name, num_sample)
    training_dataset = {i: dataset[i] for i in range(int(num_sample*0.8))}
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}

    if algorithm == "DFS":
        with Timer("DFS-Retrain") as t:
            model.DFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag="DFS_Retrain_{}_{}".format(dataset_name, time_), init_params=False)
    elif algorithm == "BFS":
        with Timer("BFS-Retrain") as t:
            model.BFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag="BFS_Retrain_{}_{}".format(dataset_name, time_), init_params=False)

def test(model_file_name, dataset_name, head_predicate_idx,  num_sample):
    #load model
    with open("./model/"+model_file_name, "rb") as f:
        model = pickle.load(f)

    #get data
    dataset,num_sample =  get_data(dataset_name, num_sample)
    training_dataset = {i: dataset[i] for i in range(int(num_sample*0.8))}
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}

    model.generate_target(head_predicate_idx, testing_dataset, num_repeat=100)




if __name__ == "__main__":
    #run_preprocess()
    
    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78
    args = get_args()
    if not args.print_log:
        redirect_log_file()
    if args.head_predicate_idx == 51:
        mimic = "mimic_3_clip_scaled"
    elif args.head_predicate_idx == 61:
        mimic = "mimic_3_scaled"
    else:
        raise ValueError
    #fit(model_name="mimic", dataset_name=mimic, head_predicate_idx=args.head_predicate_idx,  num_sample=-1, worker_num=args.worker, num_iter=2, algorithm="DFS")
    #retrain(model_file_name="model-DFS_mimic_3_clip_scaled_2021-05-25 15:07:45.234259.pkl", delete_formula_idx_list=[1,2,5],dataset_name=mimic, head_predicate_idx=args.head_predicate_idx,  num_sample=-1, worker_num=args.worker, num_iter=2, algorithm="DFS")
    #retrain(model_file_name="model-DFS_mimic_1_scaled_2021-05-23 15:56:18.702553.pkl", delete_formula_idx_list=[],dataset_name=mimic, head_predicate_idx=args.head_predicate_idx,  num_sample=-1, worker_num=args.worker, num_iter=1, algorithm="DFS")
    #test(model_file_name='model-DFS_Retrain_mimic_3_clip_scaled_2021-05-26 20:58:01.109485.pkl', dataset_name=mimic, head_predicate_idx=args.head_predicate_idx,  num_sample=-1)
    #fit(model_name="mimic", dataset_name=mimic, head_predicate_idx=args.head_predicate_idx, num_sample=-1, worker_num=args.worker, num_iter=2, algorithm="Hawkes")
    #fit(model_name="mimic", dataset_name=mimic, head_predicate_idx=args.head_predicate_idx, num_sample=-1, worker_num=args.worker, num_iter=2, algorithm="Hawkes_Rev")
    
    
    #urine 1 ratio in mimic: 0.776597959608578
        

    #run_expriment_group(args)
    #rescale_data("mimic_2.npy", "mimic_2_scaled.npy", scale=10/500)
    #dataset_stat(dataset=args.dataset)
    
    #data, num_sample = get_data(dataset_name=mimic, num_sample=100)
    #l = list()
    #for k,v in data.items():
        #print(k,v)
    #    print(v[61])
        #if v[51]["state"][0] == 1 or len(v[51]["state"])>2:
        #print(v[51]["state"]) 
        #print(v[51]["time"])
    #print(np.array(l).mean())
        

    #process_raw_data(input_file="mimic_dataset_v3.csv", output_file="mimic_1.npy")
    #process_raw_data(input_file="second_mimic_dataset_v3.csv", output_file="mimic_2.npy")
    #process_raw_data(input_file="mimic_3.csv", output_file="mimic_3.npy")
    #dataset_stat("mimic_2")
    #dataset_stat("mimic_1")
    #dataset_stat("mimic_3")
    #rescale_data("mimic_3.npy", "mimic_3_scaled.npy", scale=10/400)
    #dataset_stat("mimic_3_scaled")
    #rescale_data("mimic_1.npy", "mimic_1_scaled.npy", scale=10/500)

    #dataset_stat("mimic_1_scaled")
    #clip_data("mimic_1.npy", "mimic_1_clip.npy", 51, 48)
    #dataset_stat("mimic_1_clip")

    #clip_data("mimic_3.npy", "mimic_3_clip.npy", 51, 36)
    #dataset_stat("mimic_3_clip")
    #rescale_data("mimic_3_clip.npy", "mimic_3_clip_scaled.npy", scale=10/55)
    #dataset_stat("mimic_3_clip_scaled")

