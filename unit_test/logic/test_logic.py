import sys
sys.path.extend(["../","./","../../"])
import numpy as np

from logic import Logic
from utils.args import get_args


class Test_Logic:
    """ Testing group for ./logic/__init__.py
    """
    def get_logic(self, dataset_name="synthetic",logic_name="hawkes"):
        args = get_args()
        args.dataset_name = dataset_name
        args.synthetic_logic_name = logic_name
        return Logic(args)

    def test_template_hawkes(self):
        logic = self.get_logic()
        template = logic.get_template(target_predicate_ind=1)
        assert template[0]['neighbor_ind'] == [0]
        assert template[0]['neighbor_combination'].tolist() == [1]
        assert template[0]['formula_effect'] == [1,-1]
        BEFORE = logic.logic.BEFORE
        assert template[0]['time_template'].tolist() == [0, BEFORE]
    
    def test_add_rule(self):
        logic = self.get_logic()
        logic.delete_rule(0)
        rule = np.array([1,0])
        time_template = np.array([100, 0]) 
        R_arrray = np.array([1,1])
        logic.add_rule(rule, time_template, R_arrray)
        template = logic.get_template(target_predicate_ind=1)
        assert template[0]['neighbor_combination'].tolist() == [0]
        assert template[0]['formula_effect'] == [-1,1]
    
    def test_delete_rule(self):
        logic = self.get_logic()
        template_original = logic.get_template(target_predicate_ind=1)
        rule, time_template, R_arrray = logic.delete_rule(0)
        template = logic.get_template(target_predicate_ind=1)
        assert template == {}
        logic.add_rule(rule, time_template, R_arrray)
        template_after = logic.get_template(target_predicate_ind=1)
        assert template_original == template_after

if __name__ == "__main__":
    t = Test_Logic()
    t.test_add_rule()