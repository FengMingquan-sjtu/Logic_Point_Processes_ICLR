import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
from torch.autograd import Variable
from torch.multiprocessing import Pool, cpu_count
import time
class model:
    def __init__(self):
        self.v = Variable((torch.ones(1) * -1).double(), requires_grad=True)
        
    def forward(self, coef):
        return self.v * self.v * coef
    def zero(self, optimizer):
        for i in range(10):
           
            optimizer.zero_grad()
            
            #print("p-0: iter={}, v={:.4f}, v.grad={}".format(i, self.v.data[0], self.v.grad), flush=1)
            time.sleep(0.10)
            
    def one(self, optimizer):
        for i in range(10):
            print("p-1: iter={}, before forward, v={:.4f}, v.grad={}".format(i, self.v.data[0], self.v.grad), flush=1)
            l = self.forward(1)
            l.backward()
            print("p-1: iter={}, before update, v={:.4f}, v.grad={:.4f}".format(i, self.v.data[0], self.v.grad.data[0]), flush=1)
            optimizer.step()
            #print("proc-1: optimizer step=",optimizer.state[self.v]["step"])
            print("p-1: iter={}, after update, v={:.4f}, v.grad={:.4f}".format(i, self.v.data[0], self.v.grad.data[0]), flush=1)
            time.sleep(0.1)
    
    def two(self, optimizer):
        for i in range(10):
            print("p-2: iter={}, before forward, v={:.4f}, v.grad={}".format(i, self.v.data[0], self.v.grad), flush=1)
            l = self.forward(2)
            l.backward()
            print("p-2: iter={}, before update, v={:.4f}, v.grad={:.4f}".format(i, self.v.data[0], self.v.grad.data[0]), flush=1)
            optimizer.step()
            #print("proc-2: optimizer step=",optimizer.state[self.v]["step"])
            print("p-2: iter={}, after update, v={:.4f}, v.grad={:.4f}".format(i, self.v.data[0], self.v.grad.data[0]), flush=1)
            time.sleep(0.2)

    def run(self):
        optimizer = optim.Adam([self.v,], lr=0.1)
        
        p0 = mp.Process(target=self.zero, args=(optimizer,))
        p1 = mp.Process(target=self.one, args=(optimizer,))
        p2 = mp.Process(target=self.two, args=(optimizer,))
        
        procs = [p0, p1,p2]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print("main proc v =", self.v)
    
    def run_pool(self):
        optimizer = optim.Adam([self.v,], lr=0.1)
        arg_list = list()
        for i in range(12):
            arg_list.append(optimizer)
        with Pool(3) as p:
            p.map(self.one, arg_list)

    
    def share_memory(self):
        self.v.share_memory_()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    m = model()
    #m.share_memory()
    #m.run_pool()
    m.run()