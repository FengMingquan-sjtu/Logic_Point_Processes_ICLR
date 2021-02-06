import argparse
import os
import os.path as osp
import sys
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.models.rnn import (
    EventSeqDataset,
    ExplainableRecurrentPointProcess,
    RecurrentMarkDensityEstimator,
)
from pkg.models.rppn import RecurrentPointProcessNet
from pkg.utils.argparser.training import add_subparser_arguments
from pkg.utils.evaluation import eval_fns
from pkg.utils.logging import get_logger, init_logging
from pkg.utils.misc import (
    Timer,
    compare_metric_value,
    export_csv,
    export_json,
    get_freer_gpu,
    makedirs,
    set_rand_seed,
    AverageMeter
)
from pkg.utils.pp import (
    eval_nll_hawkes_exp_kern,
    eval_nll_hawkes_sum_gaussians,
    event_seq_to_counting_proc,
)
from pkg.utils.torch import split_dataloader, convert_to_bucketed_dataloader


def get_parser():
    parser = argparse.ArgumentParser(description="Training different models. ")
    subparsers = parser.add_subparsers(
        description="Supported models", dest="model"
    )
    for model in ["ERPP", "RME", "RPPN", "HExp", "HSG", "NPHC"]:
        add_subparser_arguments(model, subparsers)

    return parser


def get_model(args, n_types):
    if args.model == "RME":
        model = RecurrentMarkDensityEstimator(n_types=n_types, **vars(args))
    elif args.model == "ERPP":
        model = ExplainableRecurrentPointProcess(n_types=n_types, **vars(args))
    elif args.model == "RPPN":
        model = RecurrentPointProcessNet(n_types=n_types, **vars(args))
    elif args.model == "HExp":
        from tick.hawkes import HawkesExpKern

        model = HawkesExpKern(args.decay, C=args.penalty, verbose=args.verbose)
    elif args.model == "HSG":
        from tick.hawkes import HawkesSumGaussians

        model = HawkesSumGaussians(
            args.max_mean,
            n_gaussians=args.n_gaussians,
            C=args.penalty,
            n_threads=args.n_threads,
            verbose=args.verbose,
        )
    elif args.model == "NPHC":
        from tick.hawkes import HawkesCumulantMatching

        model = HawkesCumulantMatching(
            integration_support=args.integration_support,
            C=args.penalty,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unsupported model={args.model}")

    return model


def get_device(cuda, dynamic=False):
    if torch.cuda.is_available() and args.cuda:
        if dynamic:
            device = torch.device("cuda", get_freer_gpu(by="n_proc"))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_hparam_str(args):
    if args.model == "ERPP":
        hparams = ["max_mean", "n_bases", "hidden_size", "lr"]
    else:
        hparams = []

    return ",".join("{}={}".format(p, getattr(args, p)) for p in hparams)








def predict_next_event(model, event_seqs, args):
    if args.model in ["ERPP", "RPPN"]:
        dataloader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=False, **dataloader_args
        )
        event_seqs_pred = model.predict_next_event(dataloader, device=device)
    elif args.model == "HExp":
        from pkg.utils.pp import predict_next_event_hawkes_exp_kern

        event_seqs_pred = predict_next_event_hawkes_exp_kern(
            event_seqs, model, verbose=True
        )
    else:
        print(
            "Predicting next event is not supported for "
            f"model={args.model} yet."
        )
        event_seqs_pred = None

    return event_seqs_pred


def get_infectivity_matrix(model, event_seqs, args):

    if args.model in ["RME", "ERPP", "RPPN"]:
        _dataloader_args = dataloader_args.copy()
        if "attr_batch_size" in args and args.attr_batch_size:
            _dataloader_args.update(batch_size=args.attr_batch_size)

        dataloader = DataLoader(
            EventSeqDataset(event_seqs), **_dataloader_args
        )
        dataloader = convert_to_bucketed_dataloader(dataloader, key_fn=len)
        infectivity = model.get_infectivity(dataloader, device, **vars(args))
    else:
        infectivity = model.get_kernel_norms()

    return infectivity

def calc_mean_absolute_error(event_seqs_true, event_seqs_pred, skip_first_n=0):
    """
    Args:
        event_seqs_true (List[List[Tuple]]):
        event_seqs_pred (List[List[Tuple]]):
        skip_first_n (int, optional): Skipe prediction for the first
          `skip_first_n` events. Defaults to 0.
    """
    mse = AverageMeter()
    for seq_true, seq_pred in zip(event_seqs_true, event_seqs_pred):
        if len(seq_true) <= skip_first_n:
            continue
        if skip_first_n == 0:
            ts_true = [0] + [t for t, _ in seq_true]
            ts_pred = [0] + [t for t, _ in seq_pred]
        else:
            ts_true = [t for t, _ in seq_true[skip_first_n - 1 :]]
            ts_pred = [t for t, _ in seq_pred[skip_first_n - 1 :]]

        mse.update(
            np.absolute(np.diff(ts_true) - np.diff(ts_pred)).mean(),
            len(ts_true) - 1,
        )

    return mse.avg

if __name__ == "__main__":

    args = get_parser().parse_args()
    assert args.model is not None, "`model` needs to be specified."

    output_path = osp.join(
        args.output_dir,
        args.dataset,
        f"split_id={args.split_id}",
        args.model,
        get_hparam_str(args),
    )
    makedirs([output_path])

    # initialization
    set_rand_seed(args.rand_seed, args.cuda)
    init_logging(output_path)
    logger = get_logger(__file__)

    logger.info(args)
    export_json(vars(args), osp.join(output_path, "config.json"))

    # load data
    input_path = osp.join(args.input_dir, args.dataset)

    if args.dataset.startswith("mimic"):
        data = np.load(osp.join(args.input_dir, "sepsis_logic_cause.npz"), allow_pickle=True)
        n_types = int(data["n_types"])
        train_event_seqs = data["train_event_seqs"]
        test_event_seqs =  data["test_event_seqs"]
        event_seqs = np.concatenate((train_event_seqs,test_event_seqs))
    else:
        data = np.load(osp.join(input_path, "data.npz"), allow_pickle=True)
        n_types = int(data["n_types"])
        event_seqs = data["event_seqs"]
        train_event_seqs = event_seqs[data["train_test_splits"][args.split_id][0]]
        test_event_seqs = event_seqs[data["train_test_splits"][args.split_id][1]]
        
    # sorted test_event_seqs by their length
    test_event_seqs = sorted(test_event_seqs, key=lambda seq: len(seq))

    if osp.exists(osp.join(input_path, "infectivity.txt")):
        A_true = np.loadtxt(osp.join(input_path, "infectivity.txt"))
    else:
        A_true = None

    with Timer("Loading trained model"):
        # define model
        model = get_model(args, n_types)

        if args.model in ["RME", "ERPP", "RPPN"]:
            dataloader_args = {
                "batch_size": args.batch_size,
                "collate_fn": EventSeqDataset.collate_fn,
                "num_workers": args.num_workers,
            }
            device = get_device(args.cuda)
            model.load_state_dict(torch.load(osp.join(output_path, "model.pt")))

            model = model.to(device)
            
            param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("param_num=",param_num)

        else:
            with open(osp.join(output_path, "model.pkl"), "rb") as f:
                model = pickle.load(f)

    
    # evaluate next event prediction
    results = {}
    if not args.skip_pred_next_event:
        with Timer("Predict the next event"):
            event_seqs_pred = predict_next_event(model, test_event_seqs, args)
            with open("result_{}.pkl".format(args.model),'wb') as f:
                pickle.dump((event_seqs_pred,test_event_seqs), f)
            if event_seqs_pred is not None:
                print(event_seqs_pred[0])
                print(test_event_seqs[0])
                mae = calc_mean_absolute_error(test_event_seqs, event_seqs_pred)
                print(mae)



