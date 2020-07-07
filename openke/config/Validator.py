# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
from openke.config import Tester

class Validator(Tester):
    def __init__(self, model = None, data_loader = None):
        super(Validator, self).__init__(data_loader=data_loader, use_gpu=torch.cuda.is_available())

        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)

        self.model = model
        self.valid_dataloader = data_loader

        self.lib.validHead.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.validTail.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.getValidHit10.restype = ctypes.c_float

        # self.valid_steps = valid_steps
        self.early_stopping_patience = 10
        self.bad_counts = 0
        self.best_hit10 = 0

    def valid(self):
        self.lib.validInit()
        validation_range = tqdm(self.valid_dataloader)
        for index, [valid_head_batch, valid_tail_batch] in enumerate(validation_range):
            score = self.valid_one_step(valid_head_batch)
            self.lib.validHead(score.__array_interface__["data"][0], index)
            score = self.valid_one_step(valid_tail_batch)
            self.lib.validTail(score.__array_interface__["data"][0], index)
        return self.lib.getValidHit10()

    def valid_one_step(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
