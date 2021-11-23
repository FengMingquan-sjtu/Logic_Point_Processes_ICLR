# Learn_Logic_PP

## Environment and packages
Python 3.7.9 with following packages
```
numpy 1.18.5

torch 1.2.0
```

## Generate data 
```
python3 generate_synthetic_data.py --dataset_id 1
```
dataset_id: an integer in [1, 12], denoting the index of dataset. There are 12 datasets, see appendix in the paper for details.

## Train
```
python3 train_synthetic.py --dataset_id 1 --time_limit 20000 --algorithm REFS
```
dataset_id: index of dataset, same with data generation.

algorithm: one of {RAFS, REFS, Brute}. RAFS and REFS are efficient algorithm proposed in the paper, and Brute is a brute-force baseline.

time_limit: the maximum running time in seconds.

## Output
The stdout and stderr are redirected to files in following folds:
```
./log/out/
./log/err/
```
The file is named with date and time of program execution, e.g. 2021-11-17 16:52:13.526238.