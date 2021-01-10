import sys
sys.path.append("./")
sys.path.append("../")

import os

import torch
import cvxpy as cp
import numpy as np

from logic import Logic
from dataloader import get_dataset
from model.master_problem import Master_Problem
from model.sub_problem import Sub_Problem
class CG_Solver:
    def __init__(self,args):
        self.args = args
        train_dataset, test_dataset = get_dataset(self.args)

        self.train_sample_ID_set = list(train_dataset.keys())
        self.test_sample_ID_set = list(test_dataset.keys())
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset



    def run(self):
        mp = Master_Problem(self.args)
        sp = Sub_Problem(self.args)
        logic = sp.get_init_logic()
        for num_iter in range(self.args.num_iter):
            print("=== begin %dth iter ==="%num_iter)
            print("Current rules are:")
            logic.print_rules()
            mp.set_logic(logic)
            # notice: set_logic also clears cache

            if num_iter == 0:
                w = np.array([-9.02285332e-10])
                b = np.array([0., 0.37099999])
                lambda_ = 3.369082285172151e-08
            else:
                w, b, lambda_ = mp.iter(self.train_dataset, self.train_sample_ID_set)
            
            print("Master problem solved")
            print("w =", w)
            print("b =", b)
            print("lambda =", lambda_)


            sp.set_logic_and_param(logic, w, b, lambda_)
            new_rule_triplet = sp.iter(self.train_dataset, self.train_sample_ID_set)
            print("Sub problem solved")
            print("new_rule_triplet =", new_rule_triplet)

            if new_rule_triplet:
                print("Add new rule and go to next iter.")
                logic.add_rule(*new_rule_triplet)
            else:
                print("=== Finished ===")
                print("w = ", w)
                print("b = ", b)
                print("Logic rules: ")
                logic.print_rules()
                break


if __name__ == "__main__":
    from utils.args import get_args
    args = get_args()
    cg = CG_Solver(args)
    cg.run()
    