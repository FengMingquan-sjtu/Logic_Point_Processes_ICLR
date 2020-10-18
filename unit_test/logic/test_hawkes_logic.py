import sys
sys.path.extend(["../","./","../../"])
import numpy as np

from logic.hawkes_logic import Hawkes_logic


class Test_Hawkes_Logic:
    """ Testing gourp for ./logic/hawkes_logic.py
    """
    def get_logic(self):
        return Hawkes_logic()

    def test_R_matrix(self):
        expected_R_matrix = np.zeros((2, 1), dtype=int)
        expected_R_matrix[0,0] = 1
        expected_R_matrix[1,0] = 1
        
        logic = self.get_logic()
        assert (logic.R_matrix == expected_R_matrix).all()
    
    def test_logic_rule_list(self):
        expected_logic_rule_list = [np.array([0,1])]

        logic = self.get_logic()
        for i in range(len(expected_logic_rule_list)):
            assert (logic.logic_rule_list[i] == expected_logic_rule_list[i]).all()
    
    def test_time_template_list(self):
        logic = self.get_logic()
        assert (logic.BEFORE == 100)
        expected_time_template_list = [np.array([0, 100])]
        for i in range(len(expected_time_template_list)):
            assert (logic.time_template_list[i] == expected_time_template_list[i]).all()
    
    def test_formula_ind_to_rule_ind(self):
        expected = [0]
        logic = self.get_logic()
        assert logic.formula_ind_to_rule_ind == expected
    
    def test_predicate_mapping(self):
        expected = {1:[0]}
        logic = self.get_logic()
        assert logic.predicate_mapping == expected

