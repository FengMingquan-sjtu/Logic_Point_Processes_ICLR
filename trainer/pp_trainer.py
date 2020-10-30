import sys
sys.path.append("../")
import os

import torch
import torch.optim as optim
import numpy as np

from logic import Logic
from dataloader import get_dataset
from model.point_process import Point_Process

class PP_Trainer():
    def __init__(self,args):
        self.args = args
        
        logic = Logic(self.args)

        # data cache
        if self.args.dataset_name == "synthetic" or self.args.dataset_name == "handcrafted":
            data_cache_file = os.path.join(self.args.data_cache_folder, "{}_{}.pt".format(self.args.dataset_name, self.args.synthetic_logic_name))
        elif self.args.dataset_name == "mimic":
            data_cache_file = os.path.join(self.args.data_cache_folder, self.args.dataset_name + ".pt")
        else:
            print("the dataset {} does not define cache file.".format(self.args.dataset_name))
            data_cache_file = None

        if os.path.isfile(data_cache_file) and not self.args.update_data_cache:
            # load cache
            print("==> Load data from cache",data_cache_file)
            train_dataset, test_dataset = torch.load(f = data_cache_file)
        else:
            # preprocess data
            print("==> Not use cache, preprocess data")
            train_dataset, test_dataset = get_dataset(self.args)
            #print(self.args.data_cache_folder)
            if self.args.data_cache_folder: #only save cache when folder is non-empty
                if not os.path.exists(self.args.data_cache_folder):
                    os.makedirs(self.args.data_cache_folder)
                print("==> save cache to",data_cache_file)
                torch.save(obj=(train_dataset, test_dataset), f=data_cache_file)

        self.train_sample_ID_set = list(train_dataset.keys())
        self.test_sample_ID_set = list(test_dataset.keys())
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.model = Point_Process(args)

    def train(self):
        optimizer = optim.Adam(self.model._parameters.values(), lr=self.args.lr)
        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)
        num_batch = len(self.train_sample_ID_set) // self.args.batch_size_train
        for _iter in range(self.args.num_iter):
            
            print('iteration',_iter)
            np.random.shuffle(self.train_sample_ID_set)
            for batch in range(num_batch):
                print('batch',batch, flush=True)
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch

                size = self.args.batch_size_train
                rand_indices = self.train_sample_ID_set[size * batch : size * (batch+1)] # extract a batch of dataset
                
                log_like = self.model.log_likelihood(self.train_dataset, rand_indices)
                loss = -log_like
                print("loss=",loss)
                
                loss.backward()
                optimizer.step()

                print(self.model._parameters,flush=True)
            
            #lr_scheduler.step()

            #if _iter % self.args.test_period == 0:
            #    self.test(_iter)

    def test(self,_iter):
        if len(self.test_sample_ID_set) == 0:
            return
        # randomly select some test samples: (used when test dataset is large)
        if self.args.batch_size_test > 0:
            test_rand_indices = np.random.choice(self.test_sample_ID_set, self.args.batch_size_test, replace=False) # extract a batch of dataset
        else:
            # use whole test dataset
            test_rand_indices = self.test_sample_ID_set

        for p in self.args.target_predicate:
            print("===test iter {}, target predicate {}===".format(_iter,p))
            with torch.no_grad(): # testing phase no grad.
                pred, gt_label =  self.model.predict(sample_ID_batch=test_rand_indices, target_predicate=p)
            print(pred)
            print(gt_label)
            #baseline model that always outputs 1.
            #pred = np.ones(pred.shape,dtype=np.int64)
            test_acc, precision, recall, f1, conf_mat =  self.model.metrics_(pred,gt_label)

            print('test accuracy =', test_acc)
            print('test precision =', precision)
            print('recall =', recall)
            print('f1 =', f1)
            print('tn, fp, fn, tp =', conf_mat)



if __name__ == "__main__":
    from utils.args import get_args
    args = get_args()
    t = PP_Trainer(args)
    t.train()