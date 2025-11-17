import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from darts import TimeSeries
from darts.models import ARIMA
import numpy as np

class Model(nn.Module):
    # def __init__(self, model_name, ckpt_path, device='cpu'):
    #     super(Model, self).__init__()
    #     self.model_name = model_name
    #     self.ckpt_path = ckpt_path
    #     self.device = device
    #     self.patch_len = 96
    #     self.model = ARIMA()
    
    # def forcast(self, data, pred_len):
    #     # import pdb;
    #     # pdb.set_trace()
    #     B,S,C = data.shape
    #     all_preds=[]
    #     for b in range(B):
    #         preds_for_b=[]
    #         for c in range(C):
    #             data_series = TimeSeries.from_values(data[b,:,c])
    #             self.model.fit(data_series)
    #             pred = self.model.predict(pred_len).values()
    #             pred = pred.reshape(1, pred_len, 1)
    #             preds_for_b.append(pred)
    #         preds_for_b = np.concatenate(preds_for_b, axis=2)
    #         all_preds.append(preds_for_b)
    #     final_preds = np.concatenate(all_preds, axis=0)
    #     final_preds = final_preds.reshape((B, pred_len, C))
    #     return final_preds
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # 可放其他超参

    def forecast(self, data: torch.Tensor, pred_len: int):
        """
        data: [B, S, C]  torch.Tensor
        return: [B, pred_len, C]  torch.Tensor  (numpy-based, no grad)
        """
        B, S, C = data.shape
        all_preds = []
        data_np = data.detach().cpu().numpy()   # 脱离图，转 numpy
        for b in range(B):
            preds_c = []
            for c in range(C):
                ts = TimeSeries.from_values(data_np[b, :, c])
                ts_pred = ts  # 避免覆盖
                ts_pred.fit()
                pred = ts_pred.predict(pred_len)
                pred = pred.values() if hasattr(pred, 'values') else pred
                preds_c.append(pred.reshape(1, pred_len, 1))
            preds_c = np.concatenate(preds_c, axis=2)  # [1, pred_len, C]
            all_preds.append(preds_c)
        final = np.concatenate(all_preds, axis=0)  # [B, pred_len, C]
        return torch.from_numpy(final).float().to(self.device)