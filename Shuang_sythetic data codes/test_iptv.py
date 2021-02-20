from scipy.io import loadmat
import pandas as pd 


def load_mat():
    MAX_LEN = 1000
    TIME_DIVISOR = 86400 # 86400 second = 1 day
    filename = "/home/fengmingquan/data/Data_IPTV_subset.mat"
    data = loadmat(filename) #return a dict
    def convert(a):
        return list(map(lambda x: x.item(), a.flatten()))
    event_seqs_raw = data["seq"].flatten()
    event_type_names = convert(data["PID"])
    print(event_type_names) #['others', 'drama', 'movie', 'news', 'entertainment', 'music', 'sports', 'military', 'records', 'kids', 'science', 'finance', 'daily life', 'education', 'laws', 'ads']
    #print(event_seqs_raw) #[array([[  436589,   436930,   462910, ..., 18109694, 18128867, 18131356],[       1,        2,        2, ...,        2,        1,        1]],dtype=int32), ..., ]

    event_seqs = list()
    for sample in event_seqs_raw:
        length = sample.shape[1]
        seq = list()
        for i in range(length):
            t = sample[i,0].float() / TIME_DIVISOR
            pred_id = sample[i,1] - 1
            seq.append((t,pred_id))



if __name__ == "__main__":
    load_mat()