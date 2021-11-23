import os 
import sys
import datetime
import time 
import argparse

import torch
import numpy as np




    
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[%s] " % self.name, end="")
        dt = time.time() - self.tstart
        if dt < 60:
            print("Elapsed: {:.4f} sec.".format(dt))
        elif dt < 3600:
            print("Elapsed: {:.4f} min.".format(dt / 60))
        elif dt < 86400:
            print("Elapsed: {:.4f} hour.".format(dt / 3600))
        else:
            print("Elapsed: {:.4f} day.".format(dt / 86400))


def redirect_log_file():
    log_root = ["./log/out","./log/err"]   
    for root in log_root:     
        if not os.path.exists(root):
            os.makedirs(root)
    t = str(datetime.datetime.now())
    out_file = os.path.join(log_root[0], t)
    err_file = os.path.join(log_root[1], t)
    sys.stdout = open(out_file, 'w')
    sys.stderr = open(err_file, 'w')

