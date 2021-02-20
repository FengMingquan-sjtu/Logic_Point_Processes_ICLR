from scipy.io import loadmat
import pandas as pd 
import random

def load_mat():
    MAX_LEN = 1000
    TIME_DIVISOR = 86400 # 86400 second = 1 day
    filename = "/home/fengmingquan/data/Data_IPTV_subset.mat"
    data = loadmat(filename) #return a dict
    def convert(a):
        return list(map(lambda x: x.item(), a.flatten()))
    event_seqs_raw = data["seq"].flatten()
    event_type_names = convert(data["PID"])
    #print(event_type_names) #['others', 'drama', 'movie', 'news', 'entertainment', 'music', 'sports', 'military', 'records', 'kids', 'science', 'finance', 'daily life', 'education', 'laws', 'ads']
    #print(event_seqs_raw) #[array([[  436589,   436930,   462910, ..., 18109694, 18128867, 18131356],[       1,        2,        2, ...,        2,        1,        1]],dtype=int32), ..., ]

    event_seqs = list()
    for sample in event_seqs_raw:
        length = sample.shape[1]
        seq = list()
        for i in range(length):
            t = sample[0,i].astype(float) / TIME_DIVISOR
            pred_id = sample[1,i] - 1
            seq.append((t,pred_id))
            if i >= MAX_LEN:
                break
        event_seqs.append(seq)
    avg_length = sum([len(seq) for seq in event_seqs])/len(event_seqs)
    train_num = int(len(event_seqs) * 0.8)
    test_num = len(event_seqs) - train_num
    print("total {} seqs, avg length={}.".format(len(event_seqs), avg_length))
    print("train_num={}, test_num={}.".format(train_num,test_num))
    random.shuffle(event_seqs)
    train_event_seqs = event_seqs[:train_num]
    test_event_seqs = event_seqs[-test_num:]




if __name__ == "__main__":
    load_mat()