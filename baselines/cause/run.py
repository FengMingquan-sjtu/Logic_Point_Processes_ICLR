import numpy as np
import os.path as osp
input_path = "/home/fengmingquan/data/cause/input/mhp-1K-10"
data = np.load(osp.join(input_path, "data.npz"), allow_pickle=True)
n_types = int(data["n_types"])
print(n_types)
event_seqs = data["event_seqs"]
split_id = 0
train_event_seqs = event_seqs[data["train_test_splits"][split_id][0]]
test_event_seqs = event_seqs[data["train_test_splits"][split_id][1]]
print(len(test_event_seqs))
print(test_event_seqs[0])
# [(1094.5910501207484, 4), (1098.1079940088416, 7), (1100.4624087040852, 1), (1101.3163237708202, 1), (1105.8506944816406, 1), (1106.1821513888342, 0), (1106.1918308538147, 8)]