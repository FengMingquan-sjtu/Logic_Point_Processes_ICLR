import sys
sys.path.append("../")

import os

import torch
import cvxpy as cp
import numpy as np

from logic import Logic
from dataloader import get_dataset
from model.master_problem import Master_Problem
from model.sub_problem import Sub_Problem
class CG_Solver():
    def __init__(self,args):
        self.args = args
        # data cache
        if self.args.dataset_name == "synthetic" or self.args.dataset_name == "handcrafted":
            data_cache_file = os.path.join(self.args.data_cache_folder, "{}_{}.pt".format(self.args.dataset_name, self.args.synthetic_logic_name))
        elif self.args.dataset_name == "mimic":
            data_cache_file = os.path.join(self.args.data_cache_folder, self.args.dataset_name + ".pt")
        else:
            print("the dataset {} does not define cache file.".format(self.args.dataset_name))
            data_cache_file = None

        if os.path.isfile(data_cache_file) and not self.args.update_data_cache:
            # load cache
            print("==> Load data from cache",data_cache_file)
            train_dataset, test_dataset = torch.load(f = data_cache_file)
        else:
            # preprocess data
            print("==> Not use cache, preprocess data")
            train_dataset, test_dataset = get_dataset(self.args)
            #print(self.args.data_cache_folder)
            if self.args.data_cache_folder: #only save cache when folder is non-empty
                if not os.path.exists(self.args.data_cache_folder):
                    os.makedirs(self.args.data_cache_folder)
                print("==> save cache to",data_cache_file)
                torch.save(obj=(train_dataset, test_dataset), f=data_cache_file)

        self.train_sample_ID_set = list(train_dataset.keys())
        self.test_sample_ID_set = list(test_dataset.keys())
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset



    def run(self):
        mp = Master_Problem(self.args)
        sp = Sub_Problem(self.args)
        logic = sp.get_init_logic()
        while True:
            mp.set_logic(logic)
            w, b, lambda_ = mp.iter(self.train_dataset, self.train_sample_ID_set)
            sp.set_logic_and_param(logic, w, b, lambda_)
            new_rule_triplet = sp.iter(self.train_dataset, self.train_sample_ID_set)
            if new_rule_triplet:
                logic.add_rule(*new_rule_triplet)
            else:
                break


if __name__ == "__main__":
    from utils.args import get_args
    args = get_args()