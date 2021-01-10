import sys
sys.path.append('../')
import os
import torch
from dataloader.synthetic import Synthetic

def get_dataset(args):
    # data cache
    if args.dataset_name == "synthetic" or args.dataset_name == "handcrafted":
        data_cache_file = os.path.join(args.data_cache_folder, "{}_{}.pt".format(args.dataset_name, args.synthetic_logic_name))
    elif args.dataset_name == "mimic":
        data_cache_file = os.path.join(args.data_cache_folder, args.dataset_name + ".pt")
    else:
        print("the dataset {} does not define cache file.".format(args.dataset_name))
        data_cache_file = None

    if os.path.isfile(data_cache_file) and not args.update_data_cache:
        # load cache
        print("==> Load data from cache",data_cache_file)
        train_dataset, test_dataset = torch.load(f = data_cache_file)
    else:
        # preprocess data
        print("==> Not use cache, preprocess data")
        if args.dataset_name == 'synthetic':
            data = Synthetic(args=args)    
        elif args.dataset_name == 'mimic':
            data = Mimic(args=args)
        else:
            raise ValueError("dataset {} is not implemented".format(args.dataset_name))

        train_dataset = data.get_dataset(is_train=True)
        test_dataset = data.get_dataset(is_train=False)
        #print(args.data_cache_folder)
        if args.data_cache_folder: #only save cache when folder is non-empty
            if not os.path.exists(args.data_cache_folder):
                os.makedirs(args.data_cache_folder)
            print("==> save cache to",data_cache_file)
            torch.save(obj=(train_dataset, test_dataset), f=data_cache_file)
    

    return train_dataset, test_dataset

if __name__ == "__main__":
    
    from utils.args import get_args
    
    args = get_args()
    
    train_dataset,test_dataset = get_dataset(args)