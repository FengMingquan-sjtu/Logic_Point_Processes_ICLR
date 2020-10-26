import sys
sys.path.extend(["../","./","../../"])
import numpy as np
from numpy.random import default_rng

from dataloader.synthetic import Synthetic
from model.point_process import Point_Process
from utils.args import get_args


class Test_synthetic:
    """ Testing group for ./dataloader/synthetic.py
    """
    def get_synthetic(self, target_predicate=[1], dataset_name="synthetic",logic_name="hawkes"):
        args = get_args()
        args.dataset_name = dataset_name
        args.target_predicate = target_predicate
        args.synthetic_logic_name = logic_name
        return Synthetic(args)

    def get_pp(self,target_predicate=[1], dataset_name="synthetic",logic_name="hawkes"):
        args = get_args()
        args.target_predicate = target_predicate
        args.dataset_name = dataset_name
        args.synthetic_logic_name = logic_name
        return Point_Process(args)
        
    def _run_synthetic_simple(self, logic_name):
        args = get_args()
        args.target_predicate = [1]
        args.dataset_name = "synthetic"
        args.synthetic_logic_name = logic_name
        args.synthetic_time_horizon = 50
        dataloader = Synthetic(args)
        seed = 1
        dataset = dataloader.generate_data(sample_ID_lb=0, sample_ID_ub=1, seed=seed)

        pp = Point_Process(args)
        pp.set_parameters(w=dataloader.model._parameters["weight"], b=dataloader.model._parameters["base"], requires_grad=False)
        rng = default_rng(seed)
        expected_dataset = {0:{}}
        expected_dataset[0][1] = {"time":[0], "state":[0]}
        expected_dataset[0][0] = expected_dataset[0][1]
        cur_time = 0
        while cur_time < args.synthetic_time_horizon:
            if logic_name == "hawkes":
                t_m = cur_time
            else:
                t_m = cur_time + args.synthetic_time_horizon
            intensity_m = pp.intensity(t=t_m, dataset=expected_dataset, sample_ID=0, target_predicate=1)
            dwell_time = rng.exponential(scale=1.0/intensity_m).item()
            cur_time += dwell_time
            if cur_time > args.synthetic_time_horizon:
                break
            # following line is dummy, only used to keep same rng with synthetic.py
            pred_idx = rng.choice(2, 1, p=[0,1])
            intensity_ = pp.intensity(t=cur_time, dataset=expected_dataset, sample_ID=0, target_predicate=1)
            accept_rate = intensity_ / intensity_m
            if rng.random() < accept_rate:
                expected_dataset[0][1]["time"].append(cur_time)
                cur_time += (args.time_tolerence * 1.1 )
                expected_dataset[0][1]["time"].append(cur_time)
                expected_dataset[0][1]["state"].extend([1,0])
        
        expected_data = expected_dataset[0][1]["time"]
        real_data = dataset[0][1]["time"]
        #print(real_data)
        #print(expected_data)
        for i in range(len(real_data)):
            assert abs(real_data[i] - expected_data[i]) < 1e-5
    
    def test_hawkes(self):
        self._run_synthetic_simple(logic_name="hawkes")
    def test_sc(self):
        self._run_synthetic_simple(logic_name="self_correcting")


if __name__ == "__main__":
    t = Test_synthetic()
    t.test_sc()
    #d = t.test_hawkes()
    #print(d)
            



        