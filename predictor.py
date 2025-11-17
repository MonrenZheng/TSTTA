# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2025-present, Kim et al.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import os
from typing import Dict, Optional
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.forecast import forecast
from datasets.loader import get_train_dataloader, get_test_dataloader
from utils.misc import prepare_inputs
from utils.misc import mkdir
from config import get_norm_method


class Predictor:
    def __init__(self, cfg, model, norm_module: Optional[torch.nn.Module] = None):
        self.cfg = cfg

        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module

        cfg.TRAIN.SHUFFLE, cfg.TRAIN.DROP_LAST = False, False
        self.train_loader = get_train_dataloader(cfg)
        self.test_loader = get_test_dataloader(cfg)

        self.mse_all = []
        self.mae_all = []

        self.test_errors, self.train_errors = self._get_test_errors(), self._get_train_errors()

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        log_dict = {}
        
        self.errors_all = {
            "test_mse": self.test_errors['mse'], 
            "test_mae": self.test_errors['mae'], 
            "train_mse": self.train_errors['mse'], 
            "train_mae": self.train_errors['mae'], 
        }

        results = self.get_results()
        self.save_results(results)

        self.errors_all["test_mse_all"] = self.test_errors['mse_all'].astype(float)
        
        self.save_to_npy(**self.errors_all)

        # log to W&B
        log_dict.update({f"Test/{metric}": value for metric, value in results.items()})

    @torch.no_grad()
    def _get_errors_from_dataloader(self, dataloader, tta=False, split='test') -> Dict[str, np.ndarray]:
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        mse_all = []
        mae_all = []
        
        for inputs in tqdm(dataloader, desc='Calculating Errors'):
            enc_window_raw, dec_window = prepare_inputs(inputs)
            if self.norm_method == 'SAN':
                enc_window, statistics_pred = self.norm_module.normalize(enc_window_raw)
            else:  # Normalization from Non-stationary Transformer
                means = enc_window_raw.mean(1, keepdim=True).detach()
                enc_window = enc_window_raw - means
                stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
                enc_window /= stdev
            
            ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
            # dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
            # dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
            
            dtype = torch.float32
            patch_len = self.cfg.MODEL.patch_len
            seq_len = self.cfg.MODEL.seq_len
            pred_len = self.cfg.DATA.PRED_LEN
            batch_size = enc_window.shape[0]
            pad_len = (patch_len - seq_len % patch_len) % patch_len
            if seq_len % patch_len != 0:
                enc_window = np.concatenate((np.zeros((batch_size, pad_len, 1)), enc_window), axis=1)
            enc_window = torch.tensor(enc_window, dtype=dtype).cuda()
            inference_step = pred_len // patch_len
            dis = inference_step * patch_len - pred_len
            if dis != 0:
                inference_step += 1
                dis = dis + patch_len

            seq = torch.tensor(enc_window, dtype=dtype).cuda()
            assert seq.shape[1] % patch_len == 0, f'seq_len {seq.shape[1]} is not a multiple of patch_len {patch_len}'

            # gpu autocast
            # none->float16->float32
            # 32->36->53
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                # with torch.no_grad():
                for j in range(inference_step):
                    outputs = self.model(seq, None, None, None)  
                    seq = torch.cat([seq, outputs[:, -patch_len:, :]], dim=1)
            if dis != 0:
                seq = seq[:, :-dis, :]  # 去掉pred多余的部分
            _pred_total = seq   # .detach().cpu().numpy()
            pred_total = _pred_total[:, pad_len:, :]    

            pred = pred_total[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
            
            if self.norm_method == 'SAN':
                pred = self.norm_module.de_normalize(pred, statistics_pred)
            else:  # De-Normalization from Non-stationary Transformer
                pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.cfg.DATA.PRED_LEN, 1))
                pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.cfg.DATA.PRED_LEN, 1))
            
            mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            
            mse_all.append(mse)
            mae_all.append(mae)
                
        mse_all = torch.flatten(torch.concat(mse_all, dim=0)).cpu().numpy()
        mae_all = torch.flatten(torch.concat(mae_all, dim=0)).cpu().numpy()

        return {'mse': mse_all, 'mae': mae_all}

    def _get_train_errors(self):
        return self._get_errors_from_dataloader(self.train_loader, tta=False, split='train')

    def _get_test_errors(self):

        self.cur_step = self.cfg.DATA.SEQ_LEN - 2
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        is_last = False
        test_len = len(self.test_loader.dataset)

        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, dec_window_all = prepare_inputs(inputs)
            while batch_end < len(enc_window_all):
                enc_window_first = enc_window_all[batch_start]
                
                batch_size = self.cfg.TTA.TAFAS.BATCH_SIZE
                period = batch_size - 1
                batch_end = batch_start + batch_size

                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                    is_last = True

                self.cur_step += batch_size
    
                inputs = enc_window_all[batch_start:batch_end], dec_window_all[batch_start:batch_end]
                
                pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)

                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)
                
                batch_start = batch_end
                batch_idx += 1
        
        assert self.cur_step == len(self.test_loader.dataset.test) - self.cfg.DATA.PRED_LEN - 1
        
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        return {'mse': self.mse_all.mean(), 'mae': self.mae_all.mean(), 'mse_all': self.mse_all}

    def get_results(self) -> Dict[str, float]:
        test_mse = self.test_errors['mse'].mean().astype(float)
        test_mae = self.test_errors['mae'].mean().astype(float)
        train_mse = self.train_errors['mse'].mean().astype(float)
        train_mae = self.train_errors['mae'].mean().astype(float)
        
        return {"test_mse": test_mse, 
                "test_mae": test_mae, 
                "train_mse": train_mse, 
                "train_mae": train_mae

            }

    def save_results(self, results):
        results_string = ", ".join([f"{metric}: {value:.04f}" for metric, value in results.items()])
        print("Results without TSF-TTA:")
        print(results_string)

        with open(os.path.join(mkdir(self.cfg.RESULT_DIR) / "test.txt"), "w") as f:
            f.write(results_string)

    def save_to_npy(self, **kwargs):
        for key, value in kwargs.items():
            np.save(os.path.join(self.cfg.RESULT_DIR, f"{key}.npy"), value)
