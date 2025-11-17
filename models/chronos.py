import logging
import os
import time
from torch import nn
import torch.cuda.amp as amp

import torch
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import numpy as np


from chronos import ChronosPipeline


class Model(nn.Module):  # pred较长时时间巨长...
    def __init__(self, configs):
        super().__init__()
        model_name, ckpt_path, device = configs.NAME, configs.ckpt_path, configs.device
        self.model_name = model_name
        self.device = self.choose_device(device)
        self.org_device = self.device
        self.ckpt_path = ckpt_path
        # gpu gpu autocast
        # none->float32->float16
        # 2.16->1.44->很慢->2.3
        # -> float32
        self.dtype = torch.float16  # 16节省内存 32最快？
        self.pipeline = ChronosPipeline.from_pretrained(
            self.ckpt_path,
            device_map=self.device,
            torch_dtype=self.dtype,
            # torch_dtype=torch.float64,  # 快
            # torch_dtype=torch.float32,  # 最快的！！！
            # torch_dtype=torch.bfloat16, # 不能用
            # torch_dtype=torch.float16,  # 更快
        )
        self.pipeline.model = self.pipeline.model.to(self.device)  # Ensure the model is on the correct device
        self.pipeline.model.eval()
        self.num_samples = 3  # FIXME: 多次预测取median... default=20 目测一个也能用 (多了CUDA内存爆炸
        # bfloat16,float16,float32,float64
        # 1->13s 7s 1.2s 1.7s
        # 3->19s 11s
        # 10->39s 23s

        # 相比Timer：17s->170s
        # 真相：因为Chronos内置Patch很短！！！！
        # 调整pred_len
        # 192->96->48->24->12->6
        # 2.3->1.2s->0.68->0.43->0.39->0.29
        self.patch_len = 512

    def reinit(self, device, dtype):
        self.device = self.choose_device(device)
        self.pipeline = None
        self.pipeline = ChronosPipeline.from_pretrained(
            self.ckpt_path,
            device_map=device,
            torch_dtype=dtype
        )
        self.pipeline.model.eval()

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
        assert feature == 1, f'feature={feature}'
        with torch.no_grad(), amp.autocast(dtype=self.dtype):  # FIXME:
            # with torch.no_grad():
            max_repeat = 5
            while max_repeat > 0:
                try:
                    if self.device != self.org_device:
                        logging.info(f'Chronos device changed, reinit...')
                        self.reinit(self.org_device, self.dtype)
                    # FIXME：既不能to dtype 也不能to device 都会报错
                    # data = torch.Tensor(data.reshape(batch_size, seq_len)).to(self.device)
                    # data = torch.Tensor(data.reshape(batch_size, seq_len)).to(self.dtype)
                    data = torch.Tensor(data.reshape(batch_size, seq_len))
                    forecast = self.pipeline.predict(
                        context=data,
                        prediction_length=pred_len,
                        num_samples=self.num_samples,
                        limit_prediction_length=False,
                    )
                    break
                except Exception as e:
                    logging.error(e)
                    logging.info(f'Chronos predict failed, max_repeat={max_repeat}, reinit...')
                    time.sleep(3)
                    # device = 'cuda:0' if max_repeat != 1 else 'cpu'
                    # dtype = random.choice([torch.float16, torch.float32, torch.float64])
                    device, dtype = self.device, self.dtype
                    logging.info(f'device={device}, dtype={dtype}')
                    try:
                        self.reinit(device, dtype)  # 也会失败
                    except Exception as e:
                        logging.error(e)
                        logging.info(f'Chronos reinit failed, max_repeat={max_repeat}, reinit...')
                    max_repeat -= 1
                    if max_repeat == 0:
                        raise ValueError(f'Chronos predict failed, with error: {e}')
            assert forecast.shape == (batch_size, self.num_samples, pred_len), f'forecast.shape={forecast.shape}'
            preds = np.median(forecast.numpy(), axis=1).reshape((batch_size, pred_len, 1))
            return preds