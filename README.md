# Learn_Logic_PP

## packages

```
numpy 

torch

pandas
```

## data 
生成合成数据的命令：
```
python3 generate_synthetic_data.py
```
在generate_synthetic_data.py 的第1034行，可以指定参数：
```
generate(model_idx=8, num_sample=1200, time_horizon=10, worker_num=12)
```
这里model_idx 是数据集编号，是1到12的整数，对应论文中附录table7的12个合成数据集settings。
num_sample是数据集大小。
worker_num是并行计算核心数目。
time_horizon 是样本长度，一直固定为10，不需要调整。
## train

训练合成数据的命令：
```
python3 train_synthetic.py
```
在train_synthetic.py 第119行附近，可以指定参数：
```
fit(dataset_id=8, num_sample=1200, l1_coef=100, worker_num=12, num_iter=12, algorithm="BFS")
```
其中dataset_id是数据集编号，与前面的model_idx是同样的含义。
l1_coef是l1正则化项系数。
num_sample、worker_num也与之前含义相同。
num_iter是SGD迭代次数。
algorithm是使用的算法名称，"BFS"和"DFS"都是我们的模型，"Brute"是baseline。

如需调整其他参数，如学习率、初始值等，在train_synthetic.py 第54 行附近：

```
model.weight_lr = 0.001
if model.use_exp_kernel:
    model.init_base = 0.01
    model.init_weight = 0.1
```

## output
程序运行时的输出、错误日志被重定向到了以下文件夹的文件中：
```
./log/out/
./log/err/
```
这里的文件名称是程序被执行的瞬间的日期与时刻，例如2021-11-17 16:52:13.526238。

除了输出必要的学习规则轨迹以外，还输出了 （运行时间，log-like)，用于分析性能，例如：
```
time(s), log_likelihood =  199.05779314041138 -596.9934847900338
time(s), log_likelihood =  199.06428241729736 -609.1851544877351
```
如果不需要，可以在train_synthetic.py 第53 行令 model.print_time = False