import sys
sys.path.extend(["../","./","../../"])
import numpy as np

from model.point_process import Point_Process 
from utils.args import get_args


class Test_Point_Process:
    """ Testing gourp for ./model/point_process.py
    """
    def get_pp(self, dataset_name="synthetic",logic_name="hawkes"):
        args = get_args()
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
        time_window = 2
        t = 3.5
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
        
        

