import logging
import os
import time
from torch import nn
import torch.cuda.amp as amp


import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import numpy as np


class Model(nn.Module):  # 速度对batch敏感，几乎成倍增加时间
    def __init__(self, configs):
        super().__init__()
        # autocast:
        # cpu
        # none->float16->bfloat16
        # 0.17->0.14->0.143
        # gpu
        # none->float16->float32 (不支持bfloat)
        # 0.63->0.42->0.42->
        # -》float32
        model_name, ckpt_path, device = configs.NAME, configs.ckpt_path, configs.device
        self.model_name = model_name
        self.dtype = torch.float32  # 16节省内存 32最快？
        self.device = self.choose_device(device)
        # 10配128不错: org合理 our也有提升
        # for min: 20 128???
        # FIXME：sample=100没有提升了，10就可以
        self.num_samples = 20  # FIXME: 多次预测取median... 10效果不好试试30..
        # FIXME：！！！！针对small而言：64效果差0 128效果可10 8效果极好60 16效果极好50 （提升主要来源于波动减小稳定性高
        # 问题：patch=8 org效果太差离谱不能用，
        self.patch_size = 128  # FIXME: !!!!比auto的速度快很多！ # 96会有严重问题 必须是{8, 16, 32, 64, 128}中的一个 # 64效果不好试试128！！！！！！！！有效
        self.patch_len = self.patch_size  # FIXME:
        # 时间消耗：10：fast train: Timer-10s Uni2ts-large-100s Uni2ts-base-21s Uni2ts-large-7s
        self.module = MoiraiModule.from_pretrained(ckpt_path, local_files_only=True)
        # self.module = torch.jit.script(self.module)  # FIXME: 使用JIT加速  observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"] = None,
        logging.info(f'num_samples={self.num_samples}, patch_size={self.patch_size}')
        # small 100->0.3s, 30->0.18s, 10->0.15s, 5->0.13s, 1->0.14s

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
        batch_size, seq_len, feature = data.shape
        # assert feature == 1, f'feature={feature}'

        # real_seq_l = len(data)
        real_seq_l = seq_len
        real_pred_l = pred_len

        # 重新拼接了同一个batch内的data！！！ 由此test_data的生成方式也会变！！！
        # _data = data.reshape(batch_size * real_seq_l)
        _data = data.reshape(batch_size * real_seq_l * feature)
        seq_with_zero_pred = np.concatenate([_data, np.zeros(real_pred_l)])
        date_range = pd.date_range(start='1900-01-01', periods=len(seq_with_zero_pred), freq='s')
        data_pd = pd.DataFrame(seq_with_zero_pred, index=date_range, columns=['target'])
        ds = PandasDataset(dict(data=data_pd))
        train, test_template = split(ds, offset=real_seq_l)
        test_data = test_template.generate_instances(
            prediction_length=real_pred_l,
            windows=batch_size,
            distance=real_seq_l,
        )

        with torch.no_grad(), amp.autocast(dtype=self.dtype):  # FIXME
            # with torch.no_grad():
            predictor = MoiraiForecast(
                module=self.module,
                prediction_length=real_pred_l,
                context_length=real_seq_l,
                patch_size=self.patch_size,  # FIXME: auto
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).create_predictor(batch_size=batch_size, device=self.device)  # FIXME:batch_size=batch_size!!!
            forecasts = predictor.predict(test_data.input)
            forecast_list = list(forecasts)
        # assert len(forecast_list) == 1, f'len(forcast_list)={len(forecast_list)}'
        # forecast = forecast_list[0]
        # pred = forecast.quantile(0.5)  # median
        # assert len(pred) == real_pred_l, f'len(pred)={len(pred)}'
        # return pred
        assert len(forecast_list) == batch_size, f'len(forcast_list)={len(forecast_list)}'
        preds = np.array([forecast.quantile(0.5) for forecast in forecast_list])
        
        # ? Modification for covariate setting
        # preds = preds.reshape((batch_size, real_pred_l, 1))
        preds = preds.reshape((batch_size, real_pred_l, feature))
        return preds
