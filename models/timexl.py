import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM
# from darts.models import ARIMA
from darts import TimeSeries
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        model_name, ckpt_path, device = configs.NAME, configs.ckpt_path, configs.device
        self.model_name = model_name
        self.patch_len = 96
        self.device = self.choose_device(device)
        print(f'self.device: {self.device}')
        self.model = AutoModelForCausalLM.from_pretrained(
            #'/data/qiuyunzhong/Training-LTSM/checkpoints/models--thuml--timer-base/snapshots/35a991e1a21f8437c6d784465e87f24f5cc2b395',
            ckpt_path,
            trust_remote_code=True).to(self.device)
    
    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            idx = int(device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')
    def forcast(self, data, pred_len):
        
        if len(data.shape) == 3:
            data = torch.tensor(data[:,:,-1]).squeeze().float().to(self.device)
            print('data.shape=', data.shape)
        pred = self.model.generate(data, max_new_tokens=pred_len)
        pred = pred.unsqueeze(2).detach().to('cpu').numpy()
        return pred
