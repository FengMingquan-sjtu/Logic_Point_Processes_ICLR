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
    if model_name == "crime":
        model = Logic_Learning_Model(head_predicate_idx=[head_predicate_idx,])
        
        model.predicate_notation = ["SPRING", "SUMMER", "AUTUMN", "WINTER", "WEEKDAY", "WEEKEND", "MORNING", "AFTERNOON", "EVENING", "NIGHT",  'A','B', 'C', 'D']
        model.predicate_set= list(range(len(model.predicate_notation))) # the set of all meaningful predicates
        model.body_pred_set =  model.predicate_set
        #model.body_pred_set = [10, 11, 12, 13]
        #model.body_pred_set = [13]
        model.static_pred_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        model.instant_pred_set = [10, 11, 12, 13]
        
        model.max_rule_body_length = 2
        model.max_num_rule = 20
        
        model.gain_threshold = 0.01
        model.low_grad_threshold = 0.005
        model.learning_rate = 0.0001
        model.base_lr = 0.0001
        model.weight_lr = 0.01
        model.use_decay = True
        model.use_2_bases = False
        model.init_base = 0.5
        model.opt_worker_num = 16
        model.best_N = 1
        
    if head_predicate_idx in [10,13]:
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.2
    else:
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.2

    if dataset_name.endswith("day_scaled"):
        model.scale = 10/24
        model.time_window = 24 * model.scale
        model.Time_tolerance = 1 * model.scale
        model.decay_rate = 0.1
        model.batch_size = 64
        model.integral_resolution = 0.1 * model.scale
    
    elif dataset_name.endswith("day"):
        model.time_window = 20
        model.Time_tolerance = 1
        model.decay_rate = 0.1
        model.batch_size = 64
        model.integral_resolution = 0.1

    elif dataset_name.endswith("week"):
        model.time_window = 24 * 2
        model.Time_tolerance = 12
        model.decay_rate = 0.01
        model.integral_resolution = 0.5
    
    elif dataset_name.endswith("week_scaled"):
        model.scale = 10/(24*7)
        model.time_window = 24 * 7 *  model.scale
        model.Time_tolerance = 12 *  model.scale
        model.decay_rate = 0.01
        model.batch_size = 64
        model.integral_resolution = 0.5 *  model.scale
    
    
    
    #model.static_pred_coef = model.time_window/24
    
    return model

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
    dataset, num_sample =  get_data(dataset_name, num_sample)
    training_dataset = {i: dataset[i] for i in range(int(num_sample*0.8))}
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}


    #set model hyper params
    model.batch_size_grad = num_sample #use all samples for grad
    model.batch_size_cp = num_sample
    model.num_iter = num_iter
    model.use_cp = use_cp
    model.worker_num = worker_num
    model.batch_size //= worker_num # use small batch-size, due to #bug107
    

    with Timer(algorithm) as t:
        if algorithm == "DFS":
            model.DFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag="{}_{}_{}".format(algorithm, dataset_name, time_))
        elif algorithm == "BFS":
            model.BFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag="{}_{}_{}".format(algorithm, dataset_name, time_))
        elif algorithm == "Hawkes":
            
            model.static_pred_set = []
            model.instant_pred_set = list(range(14)) # treat all static as instant.
            model.Hawkes(model.head_predicate_set[0], training_dataset, testing_dataset, tag="{}_{}_{}".format(algorithm, dataset_name, time_))
        
    print("Finish time is", datetime.datetime.now())
 

def run_expriment_group(args):
    
    #DFS
    #fit(model_name="crime", dataset_name=args.dataset, num_sample=-1, worker_num=args.worker, num_iter=6, algorithm="DFS")
    #BFS
    fit(model_name="crime", dataset_name=args.dataset, num_sample=-1, worker_num=args.worker, num_iter=6, algorithm="BFS")


def process(crime_selected):
    spring_months = [3,4,5]
    is_spring = crime_selected['MONTH'].isin(spring_months)
    is_spring.rename("SPRING", inplace=True)

    summer_months = [6,7,8]
    is_summer = crime_selected['MONTH'].isin(summer_months)
    is_summer.rename("SUMMER", inplace=True)

    autumn_months = [9,10,11]
    is_autumn = crime_selected['MONTH'].isin(autumn_months)
    is_autumn.rename("AUTUMN", inplace=True)

    winter_months = [12,1,2]
    is_winter = crime_selected['MONTH'].isin(winter_months)
    is_winter.rename("WINTER", inplace=True)

    weekend_days = ["Saturday", "Sunday"]
    is_weekend = crime_selected['DAY_OF_WEEK'].isin(weekend_days)
    is_weekend.rename("WEEKEND", inplace=True)

    weekday_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    is_weekday = crime_selected['DAY_OF_WEEK'].isin(weekday_days)
    is_weekday.rename("WEEKDAY", inplace=True)

    morning_hours = [6,7,8,9,10,11]
    is_morning =  crime_selected['HOUR'].isin(morning_hours)
    is_morning.rename("MORNING", inplace=True)
    
    afternoon_hours=[12,13,14,15,16]
    is_afternoon =  crime_selected['HOUR'].isin(afternoon_hours)
    is_afternoon.rename("AFTERNOON", inplace=True)

    evening_hours = [17,18,19,20,21]
    is_evening =  crime_selected['HOUR'].isin(evening_hours)
    is_evening.rename("EVENING", inplace=True)

    night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
    is_night =  crime_selected['HOUR'].isin(night_hours)
    is_night.rename("NIGHT", inplace=True)

    date = crime_selected["OCCURRED_ON_DATE"].apply(lambda x: x.split()[0])
    date.rename("DATE",inplace=True)

    def convert_time(date_time_pair):
        time = date_time_pair.split()[1]
        hour, minute, second = time.split(":")
        float_hour = int(hour) + int(minute)/60 + int(second)/3600
        return float_hour
    hour = crime_selected["OCCURRED_ON_DATE"].apply(convert_time)
    hour.rename("TIME",inplace=True)

    crime_processed = pandas.concat(objs=(crime_selected["OFFENSE_DESCRIPTION"], date, hour, is_spring, is_summer, is_autumn, is_winter, is_weekday, is_weekend, is_morning, is_afternoon, is_evening, is_night), axis=1)
    return crime_processed

def convert_from_df_to_date(crime_processed):
    # convert to logic-learning data, 1-day horizon seqs.
    # logic-learning input data format :
    # {0: {'time': [0, 3.420397849789993, 6.341931048876761, 7.02828641970859, 8.16064604149825, 9.504766128153767], 'state': [1, 0, 1, 0, 1, 0]}, 1: {'time': [0, 0.9831991978572895, 1.4199066113445857, 1.6292485101191285, 2.096475266132198, 4.005069218069917, 5.767948388984466, 5.77743637565431, 6.852427239152188, 7.8930935707342424, 8.589873100390394, 8.820903625226093, 9.048162949232953, 9.342080514689219], 'state': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}, 2: {'time': [0, 1.1058850237811462, 1.271081383531531, 2.7239366123865625, 2.855376376115736, 3.4611879524020916, 3.674093142880717, 4.3536109404218095, 4.531223527024521, 4.951502883753997, 5.096495412716558, 6.194746446461735, 9.743255577798488], 'state': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}, 3: {'time': [0, 9.374137055918649], 'state': [1, 0]}, 4: {'time': [0], 'state': [1]}, 5: {'time': [0, 0.01395797611577883, 1.4515718053899762, 1.5554166263608424, 3.2631901045050062, 3.377071446493159, 3.3997887424994264, 3.416948377319663, 6.879589474535199, 8.348522758390544, 9.384507895416254], 'state': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}}
    crimes = ["VANDALISM", "LARCENY THEFT FROM MV - NON-ACCESSORY", "ASSAULT - SIMPLE", "LARCENY SHOPLIFTING"]
    names = ["A", "B", "C", "D"]
    static_preds = ["SPRING", "SUMMER", "AUTUMN", "WINTER", "WEEKDAY", "WEEKEND", "MORNING", "AFTERNOON", "EVENING", "NIGHT"]
    temporal_preds = []
    data = dict()
    name_dict = dict()
    for idx, group in enumerate(crime_processed.groupby(by=["DATE"])):
        sample = dict()
        df = group[1]
        pid = 0
        for static_pred in static_preds:
            if static_pred in ["SPRING", "SUMMER", "AUTUMN", "WINTER", "WEEKDAY", "WEEKEND"]:
                #these preds do not change during seq(day), so just use the first event's value.
                #their are activated at the start of day, thus time=[0,].
                state = df[static_pred].tolist()[0]
                state = [int(state),]
                time = [0,]

            elif static_pred == "MORNING":
                morning_hours = [6,7,8,9,10,11]
                state = [0,1,0]
                time = [0, morning_hours[0], morning_hours[-1]+0.99]

            elif static_pred == "AFTERNOON":
                afternoon_hours=[12,13,14,15,16]
                state = [0,1,0]
                time = [0, afternoon_hours[0], afternoon_hours[-1]+0.99]

            elif static_pred == "NIGHT":
                night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
                #"NIGHT" is activated at [00:00-05:59] and [22:00-23:59]
                state = [1,0,1]
                time = [0, night_hours[-1]+0.99, night_hours[0]] 
            elif static_pred == "EVENING":
                evening_hours = [17,18,19,20,21]
                #"EVENING" is activated at [16:00-21:59]
                state = [0,1,0]
                time = [0, evening_hours[0], evening_hours[-1]+0.99]


            sample[pid] = {"time": time, "state": state}
            name_dict[pid] = static_pred
            pid +=1

        for pred in temporal_preds + crimes:

            if pred in crimes:
                time_ = df["TIME"][df["OFFENSE_DESCRIPTION"] == pred].tolist()
            else:
                time_ = df["TIME"][df[pred]].tolist()
            
            sample[pid] = {"time":time_, "state":[1]*len(time_)}
            name_dict[pid] = pred
            pid +=1


        data[idx] = sample

    print(name_dict.values())
    return data

def refreq(data, freq):
    # e.g. convert from day to week, freq=7
    # e.g. convert from day to month, freq=30
    
    new_data = dict()
    preds = data[0].keys()
    static_preds = [0,1,2,3,4]
    for idx in range(0, len(data.keys()) - freq):
        sample = dict()
        start_day = idx 
        for pred in preds:
            time = [np.array(data[start_day+day][pred]["time"]) + day*24 for day in range(freq)]
            time = list(np.concatenate(time))
            state = list(np.concatenate([np.array(data[start_day+day][pred]["state"]) for day in range(freq)]))
            if pred in static_preds:
                state_ = [state[0], ]
                time_ = [time[0], ]
                for i in range(1, len(state)):
                    if state[i] != state[i-1]:
                        state_.append(state[i])
                        time_.append(time[i])
                time = time_
                state = state_
            sample[pred] = {"time":time, "state":state}
        new_data[idx] = sample
    return new_data

def process_raw_data(input_file, output_file):
    crime_all = pandas.read_csv("./data/"+input_file)
    crimes = ["VANDALISM", "LARCENY THEFT FROM MV - NON-ACCESSORY", "ASSAULT - SIMPLE", "LARCENY SHOPLIFTING"]
    index = crime_all.OFFENSE_DESCRIPTION.isin(crimes)
    crime_selected = crime_all[index]
    crime_selected = crime_selected.sort_values(by="OCCURRED_ON_DATE")
    crime_processed = process(crime_selected)
    date_data = convert_from_df_to_date(crime_processed)
    np.save("./data/"+output_file, date_data)

def refreq_data(input_file, output_file, freq):
    date_data = np.load("./data/"+input_file, allow_pickle='TRUE').item()
    data = refreq(date_data, freq)
    np.save("./data/"+output_file, data)
    
def rescale_data(input_file, output_file, scale):
    data = np.load("./data/"+input_file, allow_pickle='TRUE').item()
    for sample_ID, sample in data.items():
        for pid, d in sample.items():
            data[sample_ID][pid]["time"] = list(np.array(d["time"]) * scale)
    np.save("./data/"+output_file, data)


def dataset_stat(dataset, head_predicate_idx):
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
                if pid == head_predicate_idx: #target: urine
                    target_l = data["time"][-1] - data["time"][0]
                    target_s = len(data["time"])
                    target_l_list.append(target_l)
                    target_s_list.append(target_s)
        seq_len.append(l)
        seq_size.append(s)
        seq_tot_size.append(tot_s)
    target_l_arr = np.array(target_l_list)
    target_s_arr = np.array(target_s_list)
    seq_len = np.array(seq_len)
    seq_size = np.array(seq_size)
    seq_tot_size = np.array(seq_tot_size)
    print("head pred idx=",head_predicate_idx)
    print("seq length mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_len), np.std(seq_len)))
    print("seq size mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_size), np.std(seq_size)))
    print("seq total size mean = {:.4f}, std = {:.4f}.".format(np.mean(seq_tot_size), np.std(seq_tot_size)))
    print("target length mean = {:.4f}, std = {:.4f}.".format(np.mean(target_l_arr), np.std(target_l_arr)))
    print("target size mean = {:.4f}, std = {:.4f}.".format(np.mean(target_s_arr), np.std(target_s_arr)))

def get_args():
    """Get argument parser.
    Inputs: None
    Returns:
        args: argparse object that contains user-input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="crime")
    parser.add_argument('--dataset', type=str, default="crime_all_day")
    
    parser.add_argument('--head_predicate_idx', type=int, default=13, help="A-10,B-11,C-12,D-13")
    parser.add_argument('--worker', type=int, default=16)
    parser.add_argument('--print_log', action="store_true", help="to print out training log. Defaultly, log is saved in a file named with date-time, in ./log folder.")
    
    args = parser.parse_args()
    return args

def test(model_file_name, dataset_name,   num_sample):
    #load model
    with open("./model/"+model_file_name, "rb") as f:
        model = pickle.load(f)
    head_predicate_idx = model.head_predicate_set[0]
    #get data
    dataset,num_sample =  get_data(dataset_name, num_sample)
    training_dataset = {i: dataset[i] for i in range(int(num_sample*0.8))}
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}

    model.generate_target(head_predicate_idx, testing_dataset, num_repeat=100)


if __name__ == "__main__":
    

    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78

    args = get_args()
    if not args.print_log:
        redirect_log_file()
    
    
    #fit(model_name="crime", dataset_name=args.dataset, head_predicate_idx=args.head_predicate_idx, num_sample=-1, worker_num=args.worker, num_iter=6, algorithm="DFS")
    #fit(model_name="crime", dataset_name=args.dataset, head_predicate_idx=args.head_predicate_idx, num_sample=-1, worker_num=args.worker, num_iter=6, algorithm="BFS")
    #or head_predicate_idx in [10,11,12,13]:
    #    fit(model_name="crime", dataset_name=args.dataset, head_predicate_idx=head_predicate_idx, num_sample=-1, worker_num=args.worker, num_iter=6, algorithm="Hawkes")
    models = [
        "model-BFS_crime_all_day_scaled_2021-05-25 21:13:52.385025.pkl",
        "model-BFS_crime_all_day_scaled_2021-05-25 21:19:46.688345.pkl",
        "model-BFS_crime_all_day_scaled_2021-05-25 21:25:03.354947.pkl",
        "model-BFS_crime_all_day_scaled_2021-05-26 08:23:17.427424.pkl",
    ]
    for m in models:
        test(model_file_name=m, dataset_name=args.dataset,  num_sample=-1)
    
    #dataset_stat("crime_all_day_scaled", 10)
    #dataset_stat("crime_all_day_scaled", 11)
    #dataset_stat("crime_all_day_scaled", 12)
    #dataset_stat("crime_all_day_scaled", 13)
    
    
    
    #process_raw_data("crime_all.csv","crime_all_day.npy" )
    
    #refreq_data("crime_all_day.npy", "crime_all_week.npy", freq=7)
    #refreq_data("crime_all_day.npy", "crime_all_month.npy", freq=30)

    #rescale_data("crime_all_day.npy", "crime_all_day_scaled.npy", scale=10/24)
    #rescale_data("crime_all_week.npy", "crime_all_week_scaled.npy", scale=10/(24*7))

    #data = np.load("./data/crime_all_week_scaled.npy", allow_pickle='TRUE').item()
    #print(len(data.keys()))
    #for k,v in data.items():
    #    print(v)
    #    break
    #dataset_stat(dataset=args.dataset)

    #test(dataset_name="crime_all_week", model_file="model-BFS_crime_all_week_2021-05-18 09:17:22.136497.pkl")
    #test(dataset_name="crime_all_day", model_file="model-BFS_crime_all_day_2021-05-18 09:34:09.591180.pkl")
    
    
