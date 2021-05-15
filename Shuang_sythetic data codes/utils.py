import os 
import sys
import datetime
import time 
import argparse

import torch
#import cvxpy as cp
import numpy as np


def get_args():
    """Get argument parser.
    Inputs: None
    Returns:
        args: argparse object that contains user-input arguments.
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset', type=str)
    
    args = parser.parse_args()
    return args

def get_data(dataset_name, num_sample):
    dataset_path = './data/{}.npy'.format(dataset_name)
    print("dataset_path is ",dataset_path)
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    if len(dataset.keys())> num_sample and num_sample>0: 
        dataset = {i:dataset[i] for i in range(num_sample)}
    num_sample = len(dataset.keys())
    print("sample num is ", num_sample)
    return dataset, num_sample
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[%s] " % self.name, end="")
        dt = time.time() - self.tstart
        if dt < 60:
            print("Elapsed: {:.4f} sec.".format(dt))
        elif dt < 3600:
            print("Elapsed: {:.4f} min.".format(dt / 60))
        elif dt < 86400:
            print("Elapsed: {:.4f} hour.".format(dt / 3600))
        else:
            print("Elapsed: {:.4f} day.".format(dt / 86400))


def redirect_log_file():
    log_root = ["./log/out","./log/err"]   
    for root in log_root:     
        if not os.path.exists(root):
            os.makedirs(root)
    t = str(datetime.datetime.now())
    out_file = os.path.join(log_root[0], t)
    err_file = os.path.join(log_root[1], t)
    sys.stdout = open(out_file, 'w')
    sys.stderr = open(err_file, 'w')


def get_weight(rule_str):
    r = rule_str.split("=")[1]
    r = r.split(",")[0]
    w = float(r)
    return w

def get_rule_body(rule_str, template, predicate_notation):
    static_rule = rule_str.split(",")[0]
    body, head = static_rule.split("-->")
    body = body.split(":")[1]
    preds = body.split("^")
    predicate_notation = ['A','B', 'C', 'D','E']
    template['body_predicate_idx'] = list()
    template['body_predicate_sign'] = list()
    template['head_predicate_sign'] = list()
    template['head_predicate_sign'].append(int(not ("Not" in head)))
    for pred in preds:
        for pid, pred_notation in enumerate(predicate_notation):
            if pred_notation in pred:
                body_predicate_sign = int(not ("Not" in pred))
                template['body_predicate_idx'].append(pid)
                template['body_predicate_sign'].append(body_predicate_sign)
    return template

def get_rule_rel(rule_str, template, predicate_notation):
    temp_rule = rule_str.split(",")[1]
    rules = temp_rule.split("^")
    
    template['temporal_relation_idx'] = list()
    template['temporal_relation_type'] = list()

    for rule in rules:
        for rel in ["BEFORE","EQUAL"]:
            if rel in rule:
                pred1, pred2 = rule.split(rel)
                idx1, idx2 = 0,0
                for pid, pred_notation in enumerate(predicate_notation):
                    if pred_notation in pred1:
                        idx1 = pid
                    elif pred_notation in pred2:
                        idx2 = pid
                template['temporal_relation_idx'].append((idx1, idx2))
                template['temporal_relation_type'].append(rel)
                
    return template


def get_template(rule_set_str, predicate_notation):
    model_parameter = dict()
    logic_template = dict()
    rule_set = rule_set_str.split("\n")
    num_formula = 0
    head_predicate_idx = 0
    for rule_str in rule_set:
        if "Head" in rule_str:
            for pid, pred_notation in enumerate(predicate_notation):
                if pred_notation in rule_str:
                    head_predicate_idx = pid
                    break
            base = get_weight(rule_str)
            model_parameter[head_predicate_idx] = dict()
            model_parameter[head_predicate_idx]["base"] = torch.autograd.Variable((torch.ones(1) * base).double(), requires_grad=True)
            model_parameter[head_predicate_idx]["base_cp"] = cp.Variable(1)
            model_parameter[head_predicate_idx]["base_cp"].value = [base]
            logic_template[head_predicate_idx] = dict()
        elif "Rule" in rule_str:
            w = get_weight(rule_str)
            logic_template[head_predicate_idx][num_formula] = dict()
            get_rule_body(rule_str, logic_template[head_predicate_idx][num_formula], predicate_notation)
            get_rule_rel(rule_str, logic_template[head_predicate_idx][num_formula], predicate_notation)
            model_parameter[head_predicate_idx][num_formula]=dict()
            model_parameter[head_predicate_idx][num_formula]["weight"] = torch.autograd.Variable((torch.ones(1) * w).double(), requires_grad=True)
            model_parameter[head_predicate_idx][num_formula]["weight_cp"] = cp.Variable(1)
            model_parameter[head_predicate_idx][num_formula]["weight_cp"].value = [w]
            num_formula += 1
        else:
            continue
            
    return model_parameter, logic_template, head_predicate_idx, num_formula

if __name__ == "__main__":
    rule_set_str = """Head:E, base(torch)=0.1411, base(cp)=0.1411,
                        Rule0: A --> E , A BEFORE E, weight(torch)=0.7962, weight(cp)=0.7962.
                        Rule1: Not C --> E , Not C BEFORE E, weight(torch)=0.5391, weight(cp)=0.5391.
                        Rule2: D --> E , D EQUAL E, weight(torch)=0.4747, weight(cp)=0.4747.
                        Rule3: Not A --> E , Not A EQUAL E, weight(torch)=0.5983, weight(cp)=0.5983.
                        Rule4: Not C --> Not E , Not C EQUAL Not E, weight(torch)=0.3989, weight(cp)=0.3989.
                        Rule5: Not B ^ A --> E , Not B EQUAL E ^ A BEFORE E, weight(torch)=1.3190, weight(cp)=1.3190."""
    predicate_notation = ['A','B', 'C', 'D','E']
    get_template(rule_set_str, predicate_notation)