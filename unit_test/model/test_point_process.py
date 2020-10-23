import sys
sys.path.extend(["../","./","../../"])

import numpy as np
import torch

from model.point_process import Point_Process 
from utils.args import get_args


class Test_Point_Process:
    """ Testing group for ./model/point_process.py
    """
    def get_pp(self,target_predicate=[1], dataset_name="synthetic",logic_name="hawkes"):
        args = get_args()
        args.target_predicate = target_predicate
        args.dataset_name = dataset_name
        args.synthetic_logic_name = logic_name
        return Point_Process(args)

    def test_check_state(self):
        pp = self.get_pp()
        seq = {"time":[0.0, 1.2, 3.4, 5.6, 7.8, 9.0, 10.1], "state":[0,1,0,1,0,1,0]}

        test_cases = list()
        # test case 1:
        cur_time_list = seq['time']
        expected_output_cur_state = seq['state']
        test_cases.append((cur_time_list, expected_output_cur_state))
        #test case 2:
        cur_time_list = [0.1, 1.3, 2.4, 4.6, 5.8, 7.0, 11.1]
        expected_output_cur_state = [0, 1, 1, 0, 1, 1, 0]
        test_cases.append((cur_time_list, expected_output_cur_state))

        for cur_time_list, expected_output_cur_state in test_cases:
            for i in range(len(cur_time_list)):
                output_cur_state = pp._check_state(seq=seq, cur_time=cur_time_list[i])
                assert expected_output_cur_state[i] == output_cur_state
    
    def test_get_filtered_transition_time(self):
        pp = self.get_pp()
        # test case 1
        # checking time_window and neighbor_combination
        data = {0:{"time":[0, 1.1, 1.2, 2.1, 2.2], "state":[0,1,0,1,0]}}
        time_window = 0.9
        t = 2.1
        neighbor_ind = [0]
        neighbor_combination = np.array([1]) 
        input_ = (data, time_window, t, neighbor_ind, neighbor_combination)
        transition_time_list, is_early_stop = pp._get_filtered_transition_time(*input_) 
        assert is_early_stop == False
        assert (transition_time_list[0] == np.array([2.1])).all()

        # test case 2
        # checking early stop
        data = {0:{"time":[0, 1.1, 1.2, 2.1, 2.2], "state":[0,1,0,1,0]}}
        time_window = 3
        t = 1.0
        neighbor_ind = [0]
        neighbor_combination = np.array([1]) 
        input_ = (data, time_window, t, neighbor_ind, neighbor_combination)
        transition_time_list, is_early_stop = pp._get_filtered_transition_time(*input_) 
        assert is_early_stop == True
        assert (transition_time_list[0] == np.array([])).all()

        # test case 3
        # checking neighbor_combination
        data = {0:{"time":[0, 1.1, 1.2, 2.1, 2.2], "state":[0,1,0,1,0]}}
        time_window = 0.9001
        t = 2.1
        neighbor_ind = [0]
        neighbor_combination = np.array([0]) 
        input_ = (data, time_window, t, neighbor_ind, neighbor_combination)
        transition_time_list, is_early_stop = pp._get_filtered_transition_time(*input_) 
        assert is_early_stop == False
        assert (transition_time_list[0] == np.array([1.2])).all()


    def test_get_history_cnt_hawkes(self):
        pp = self.get_pp() 
        BEFORE = pp.logic.logic.BEFORE
        DT = pp.args.time_tolerence
        D = pp.args.time_decay_rate

        #test case 1
        target_ind_in_predicate = 1
        transition_time_list = [np.array([0,1,2,3,4]) ] 
        t = 4 + DT
        time_template = np.array([0, BEFORE])
        input_ = (target_ind_in_predicate, time_template, transition_time_list, t)
        history_cnt = pp._get_history_cnt(*input_)

        t_a = transition_time_list[0]
        expected_history_cnt = np.sum(np.exp(-(t-t_a)*D))
        assert history_cnt == expected_history_cnt
        assert history_cnt == 3.487190701886659

        #test case 2
        target_ind_in_predicate = 2
        transition_time_list = [np.array([0,1,2,3,4]), np.array([0,1,2,3,4,5,6]) ] 
        t = 8
        time_template = np.array([0, BEFORE, BEFORE])
        input_ = (target_ind_in_predicate, time_template, transition_time_list, t)
        history_cnt = pp._get_history_cnt(*input_)
        t_a,t_b = transition_time_list
        t_a_r = t_a.reshape(1,5)
        t_b_r = t_b.reshape(7,1)
        hist = np.sum(t_b_r-t_a_r > DT, axis=0)
        decay = np.exp(-(t-t_a)*D)
        expected_history_cnt = np.sum(hist * decay)
        assert history_cnt == expected_history_cnt
        assert history_cnt == 5.651437027073542
    
    def test_get_history_cnt_self_correcting(self):
        pp = self.get_pp(logic_name = "self_correcting")
        BEFORE = pp.logic.logic.BEFORE
        DT = pp.args.time_tolerence
        D = pp.args.time_decay_rate

        #test case 1
        target_ind_in_predicate = 1
        transition_time_list = [np.array([0,1,2,3,4]) ] 
        t = 4 + DT
        time_template = np.array([0, BEFORE])
        input_ = (target_ind_in_predicate, time_template, transition_time_list, t)
        history_cnt = pp._get_history_cnt(*input_)

        t_a = transition_time_list[0]
        expected_history_cnt = np.sum(np.exp(-(t-t_a)*D))
        assert history_cnt == expected_history_cnt
        assert history_cnt == 3.487190701886659

    def test_get_feature_hawkes(self):
        pp = self.get_pp() 
        DT = pp.args.time_tolerence
        D = pp.args.time_decay_rate
        #test case 1: hawkes
        target_predicate = 1
        sample_ID = 1
        pred0 = {"time":[0, 1, 1, 2, 2], "state":[0,1,0,1,0]}
        pred1 = pred0.copy()
        dataset = {sample_ID:{0:pred0, 1:pred1}}
        t = 2 + DT
        input_ = (t,dataset,sample_ID,target_predicate)
        feature_list = pp.get_feature(*input_)

        t_a = np.array([1,2])
        expected_feature_list = torch.tensor(np.sum(np.exp(-(t-t_a)*D)) )
        assert (feature_list[0]==expected_feature_list).all() 
        assert (feature_list[0].item() == 1.8187303893318676)

    def test_get_feature_self_correcting(self):
        #test case 1: self_correcting
        pp = self.get_pp(logic_name = "self_correcting")
        DT = pp.args.time_tolerence
        D = pp.args.time_decay_rate 
        target_predicate = 1
        sample_ID = 1
        pred0 = {"time":[0, 1, 1, 2, 2], "state":[0,1,0,1,0]}
        pred1 = pred0.copy()
        dataset = {sample_ID:{0:pred0, 1:pred1}}
        t = 2 + DT
        input_ = (t,dataset,sample_ID,target_predicate)
        feature_list = pp.get_feature(*input_)

        t_a = np.array([1,2])
        expected_feature_list = torch.tensor(-1 * np.sum(np.exp(-(t-t_a)*D)))
        assert (feature_list[0]==expected_feature_list).all() 
        assert (feature_list[0].item() == -1.8187303893318676)
    
    def test_intensity_hawkes(self):
        pp = self.get_pp() 
        DT = pp.args.time_tolerence
        D = pp.args.time_decay_rate
        W = 0.1
        B = 0.2
        pp.set_parameters(w=W, b=B)
        #test case 1: hawkes
        target_predicate = 1
        sample_ID = 1
        pred0 = {"time":[0, 1, 1, 2, 2], "state":[0,1,0,1,0]}
        pred1 = pred0.copy()
        dataset = {sample_ID:{0:pred0, 1:pred1}}
        t = 2 + DT
        input_ = (t,dataset,sample_ID,target_predicate)
        intensity = pp.intensity(*input_)

        t_a = np.array([1,2])
        expected_intensity = W * np.sum(np.exp(-(t-t_a)*D)) + B 
        assert abs(intensity[0].item() - expected_intensity) <= 1e-6
        assert abs(intensity[0].item() - 0.3818730389331868) <= 1e-6        

    def test_intensity_self_correcting(self):
        pp = self.get_pp(logic_name = "self_correcting") 
        DT = pp.args.time_tolerence
        D = pp.args.time_decay_rate
        W = 0.1
        B = 0.2
        pp.set_parameters(w=W, b=B)
        #test case 1: self_correcting
        target_predicate = 1
        sample_ID = 1
        pred0 = {"time":[0, 1, 1, 2, 2], "state":[0,1,0,1,0]}
        pred1 = pred0.copy()
        dataset = {sample_ID:{0:pred0, 1:pred1}}
        t = 2 + DT
        input_ = (t,dataset,sample_ID,target_predicate)
        intensity = pp.intensity(*input_)

        t_a = np.array([1,2])
        expected_intensity = - W * np.sum(np.exp(-(t-t_a)*D)) + B 
        assert abs(intensity[0].item() - expected_intensity) <= 1e-6
        assert abs(intensity[0].item() - 0.018126961066813246) <= 1e-6   


 

if __name__ =="__main__":
    tpp = Test_Point_Process()
    tpp.test_get_feature()

        
        

