import sys
sys.path.append("../")

import os

import torch
import cvxpy as cp
import numpy as np

from logic import Logic
from dataloader import get_dataset
from model.master_problem import Master_Problem

class Master_Solver():
    def __init__(self,args):
        self.args = args
        
        logic = Logic(self.args)

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

        self.model = Master_Problem(args)

    def train(self):
        log_likelihood = self.model.log_likelihood(self.train_dataset, self.train_sample_ID_set)
        objective = cp.Minimize( - log_likelihood)
        constraints = self.model.constraints()
        prob = cp.Problem(objective, constraints)
        opt = prob.solve()
        w = self.model._parameters["weight"]
        b = self.model._parameters["base"]
        print("opt value = ", opt)
        print("opt w = ", w.value)
        print("opt b = ", b.value)
        print("dual_value[0] = ", constraints[0].dual_value)
        print("dual_value[1] = ", constraints[1].dual_value)
        





if __name__ == "__main__":
    from utils.args import get_args
    args = get_args()
    t = Master_Solver(args)
    t.train()