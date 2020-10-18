import sys
sys.path.extend(["../","./","../../"])
import numpy as np

from logic import Logic
from utils.args import get_args


class Test_Logic:
    """ Testing gourp for ./logic/__init__.py
    """
    def get_logic(self, dataset_name="synthetic",logic_name="hawkes"):
        args = get_args()
        args.dataset_name = dataset_name
        args.synthetic_logic_name = logic_name
        return Logic(args)

    def test_template_hawkes(self):
        logic = self.get_logic()
        template = logic.get_template(target_predicate_ind=1)
        assert template[0]['predicate_ind'] == [0,1]
        assert template[0]['neighbor_ind'] == [0]
        assert template[0]['neighbor_combination'].tolist() == [1]
        assert template[0]['formula_effect'] == [1,-1]
        BEFORE = logic.logic.BEFORE
        assert template[0]['time_template'].tolist() == [0, BEFORE]