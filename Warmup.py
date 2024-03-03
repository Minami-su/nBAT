import matplotlib.pyplot as plt
import math
import torch
from math import cos, pi


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    warmup_epoch = max_epoch/5 if warmup else 0
    
    need=max_epoch/50
    print(current_epoch, max_epoch,warmup_epoch,lr_max)
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
        print('lr1: %.10f'% lr)
    else:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        print('lr2: %.10f'% lr)            
    for param_group in optimizer.param_groups:
    
        print('lr3: %.10f'% lr)
        param_group['lr'] = lr
        

    