import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.multiprocessing as tmp
import multiprocessing as mp
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        self.weights = torch.autograd.Variable((torch.tensor([1, 2]).double()), requires_grad=True)
    def loss(self, optimizer):
        optimizer.zero_grad()
        print(optimizer.param_groups[0]['params'])
        loss = torch.log(torch.abs(torch.sum(self.weights)) + 0.01)
        loss.backward()
        optimizer.step()

    def train(self):
        worker_num = 5
        
        #self.weights.share_memory_()
        print("start mp",flush=1)
        #with tmp.Pool(worker_num) as p:
        #    p.starmap(optimize, [(self,) for i in range(worker_num)])
        optimizer = optim.SGD([self.weights], lr=0.001)
        with mp.Pool(worker_num) as p:
            l = p.map(self.loss, [optimizer for i in range(worker_num)])
        
        print(self.weights)
            

def optimize(model):
    #print("enter optimize", flush=1)
    optimizer = optim.SGD([model.weights], lr=0.001)
    for epoch in range(1):
        for batch in range(1):
            #print("enter batch",flush=1)
            print("enter weights=",model.weights.data)
            optimizer.zero_grad()
            #print("cleaned grad",flush=1)
            loss = torch.log(torch.abs(torch.sum(model.weights)) + 0.01)
            #print("got loss",flush=1)
            loss.backward(retain_graph=True)
            #print("backwarded loss",flush=1)
            optimizer.step()
            #print("optimized weights",flush=1)
            print("loss=",loss.item())
            print("out weights=",model.weights.data)

def run1():
    m = Model()
    worker_num = 5
    
    m.weights.share_memory_()
    print("start mp",flush=1)
    with tmp.Pool(worker_num) as p:
        p.starmap(optimize, [(m,) for i in range(worker_num)])

def run2():
    m = Model()
    m.train()
if __name__ == "__main__":
    from logic_learning_2 import Timer
    with Timer("without share-mem"):
        run1()