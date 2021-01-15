#!/bin/bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
WS=`pwd`

n_seqs=1000
n_types=10
dataset=pgem-$(($n_seqs / 1000))K-$n_types
shared_args="--dataset $dataset"
n_splits=5  # if modified, remember to modify below as well!!!

PY="python3.7"
LOCAL_RUN="xargs -L1 $PY"
CONDOR_RUN="./condor_wrapper.py $PY --notify_mode Always"
CONDOR_RUN_GPU="$CONDOR_RUN --gpu 1 --getenv"

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
    exit 1
fi

if [[ $* == *all* ]] || [[ $* == *preprocess* ]]; then
    $PY preprocessing/generate_events_by_pgem.py \
        --n_seqs $n_seqs \
        --n_copies 2 \
        --max_t 500
fi

if [[ $* == *all* ]] || [[ $* == *HExp* ]]; then
    printf "%s\n" "$WS/tasks/train.py HExp $shared_args --decay 0.1 --penalty 2.0 --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *NPHC* ]]; then
    printf "%s\n" "$WS/tasks/train.py NPHC $shared_args --integration_support 20 --verbose --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *HSG* ]]; then
    printf "%s\n" "$WS/tasks/train.py HSG $shared_args --max_mean 1000 --n_gaussians 5 --verbose --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *RPPN* ]]; then
    printf "%s\n" "$WS/tasks/train.py RPPN $shared_args --epoch 200 --cuda --split_id "{0..4} | $LOCAL_RUN
fi

if [[ $* == *all* ]] || [[ $* == *ERPP* ]]; then
    printf "%s\n" "$WS/tasks/train.py ERPP $shared_args --max_mean 20 --n_bases 7 --cuda --split_id "{0..4} | $LOCAL_RUN
fi

$PY postprocessing/summarize_results.py $dataset
# python postprocessing/compare_hparams.py $dataset HExp decay penalty
# python postprocessing/compare_hparams.py $dataset NPHC integration_support
# python postprocessing/compare_hparams.py $dataset HSG max_mean n_gaussians penalty
# python postprocessing/compare_hparams.py $dataset ERPP max_mean n_bases