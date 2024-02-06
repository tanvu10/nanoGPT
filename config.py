import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Config():
    def __init__(self,
            batch_size = 16, # how many independent sequences will we process in parallel?
            block_size = 256, # what is the maximum context length for predictions? (T)
            max_iters = 5000,
            eval_interval = 1,
            learning_rate = 3e-4,
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
            eval_iters = 200,
            n_embd = 64,
            n_head = 8,
            n_layer = 2,
            dropout = 0.2):
        
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.device = device
        self.eval_iters = eval_iters
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout