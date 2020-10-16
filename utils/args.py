import argparse
def get_args():
    """Get argument parser.
    Inputs: None
    Returns:
        args: argparse object that contains user-input arguments.
    """

    parser = argparse.ArgumentParser(description="Temporal Logic Point Process")
    #exp
    parser.add_argument('--exp_name', type=str, default="tlpp",
                        help="name of experiment",)
    
    #mimic dataset
    parser.add_argument('--time_window_drug', type=float, default=0.2,
                        help="time window of drug predicates. history earlier than t-time_window is ingored.")
    parser.add_argument('--time_window_sym', type=float, default=2.0,
                        help="time window of symptom predicates. history earlier than t-time_window is ingored.")

    #synthetic  dataset
    parser.add_argument('--synthetic_logic_name', type=str, default="hawkes",
                        choices = ["hawkes","self_correcting"],
                        help="which logic to synthesize.")
    parser.add_argument('--synthetic_training_sample_num', type=int, default=100,
                        help="")
    parser.add_argument('--synthetic_testing_sample_num', type=int, default=50,
                        help="")
    parser.add_argument('--synthetic_time_horizon', type=int, default=100,
                        help="Time horizon of generating data, i.e. maximum valid time.")
    parser.add_argument('--synthetic_time_window', type=int, default=1000,
                        help="time window of synthetic dataset. history earlier than t - time_window is ingored.")
    parser.add_argument('--synthetic_weight', type=float, default=0.1,
                        help="ground truth weight of synthetic data")
    parser.add_argument('--synthetic_base', type=float, default=0.2,
                        help="ground truth base of synthetic data")
    
    
    #dataset general setting
    parser.add_argument('--dataset_name', type=str, default="mimic",
                        choices = ["mimic","synthetic","handcrafted"],)
    parser.add_argument('--train_data_file', type=str, default="/home/fengmingquan/data/mimic/train_mimic_500_10.csv")
    #parser.add_argument('--train_data_file', type=str, default="/Users/fmq/Downloads/data/train_test_data_set/train_mimic_500_10.csv",
    #                    help="train data file.")
    parser.add_argument('--test_data_file', type=str, default="/home/fengmingquan/data/mimic/test_mimic_100_10.csv")
    #parser.add_argument('--test_data_file', type=str, default="/Users/fmq/Downloads/data/train_test_data_set/test_mimic_100_10.csv",
    #                    help="test data file.")
    parser.add_argument('--update_data_cache', action='store_true',
                        help="whether to update data cache, default=False.")
    parser.add_argument('--data_cache_folder',type=str,  default="/home/fengmingquan/output/logic_pp/cache",)
    #parser.add_argument('--data_cache_folder',type=str,  default="/Users/fmq/Downloads/cache",
    #                    help='''folder to save and to load dataset cache, if empty then not use cache.
    #                            cache file will be "data_cache_folder/dataset_name.pt"''')
    
    #trainer
    parser.add_argument('--num_iter', type=int, default=100,)
    parser.add_argument('--batch_size_test', type=int, default=0,
                        help = "batch size of testing. if >0, randomly select a batch of testing data, if <=0, use whole testing data.")
    parser.add_argument('--batch_size_train', type=int, default=50)

    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--test_period', type=int, default=10,
                        help="after every test_period iters of training, test and print.")
    
    
    #model
    parser.add_argument('--integral_grid', type=float, default=0.01,
                        help="dt in approximate integral")
    parser.add_argument('--time_tolerence', type=float, default=1e-6,
                        help="tolerence of time relation BEFORE,EQUAL.")
    parser.add_argument('--target_predicate', nargs='+',  type=int, default=[24],
                        help="int list, indices of target predicates.")
    parser.add_argument('--non_negative_map', type=str, default="max",
                        help="type of non_negative mapping in intenisty calculation.")
    parser.add_argument('--init_weight_range', type=float, default=0.3,
                        help="initial weight range is (0,x). Notice x should be small, like 0.001.")
    parser.add_argument('--time_decay_rate', type=float, default=0.2,
                        help="decay rate of history counts, the larger, the faster.")
    
                        

    
    args = parser.parse_args()
    return args