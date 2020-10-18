import sys
sys.path.append('../')

from dataloader.synthetic import Synthetic

def get_dataset(args):
    if args.dataset_name == 'synthetic':
        data = Synthetic(args=args)    
    elif args.dataset_name == 'mimic':
        data = Mimic(args=args)
    elif args.dataset_name == 'handcrafted':
        data = Handcrafted(args=args)
    else:
        raise ValueError("dataset {} is not implemented".format(args.dataset_name))

    train_dataset = data.get_dataset(is_train=True)
    test_dataset = data.get_dataset(is_train=False)

    return train_dataset, test_dataset

if __name__ == "__main__":
    
    from utils.args import get_args
    
    args = get_args()
    
    train_dataset,test_dataset = get_dataset(args)