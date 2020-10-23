import sys
sys.path.extend(["../","./","../../"])
import numpy as np
from numpy.random import default_rng

from dataloader.synthetic import Synthetic
from utils.args import get_args


class Test_synthetic:
    """ Testing group for ./dataloader/synthetic.py
    """
    def get_synthetic(self, dataset_name="synthetic",logic_name="hawkes"):
        args = get_args()
        args.dataset_name = dataset_name
        args.synthetic_logic_name = logic_name
        return Synthetic(args)
        
    def test_hawkes(self):
        dataloader = self.get_synthetic()
        dataset = dataloader.generate_data(sample_ID_lb=0, sample_ID_ub=1, seed=1)
        