##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

from typing import Tuple, Optional

import torch
import torch.nn as nn

from config import get_norm_method
from utils.misc import prepare_inputs
import numpy as np

def forecast(
    cfg, 
    inputs: Tuple[torch.Tensor, torch.Tensor], 
    model: nn.Module,
    norm_module: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    enc_window, dec_window = prepare_inputs(inputs)
    norm_method = get_norm_method(cfg)
    if norm_method == 'SAN':
        enc_window, statistics = norm_module.normalize(enc_window)
    elif norm_method == 'RevIN':
        enc_window = norm_module(enc_window, 'norm')
    elif norm_method == 'DishTS':
        enc_window, _ = norm_module(enc_window, 'forward')
    else:  # Normalization from Non-stationary Transformer
        means = enc_window.mean(1, keepdim=True).detach()
        enc_window = enc_window - means
        # stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        enc_window /= stdev
    
    ground_truth = dec_window[:, -cfg.DATA.PRED_LEN:, cfg.DATA.TARGET_START_IDX:].float()
    
    dtype = torch.float32
    patch_len = cfg.MODEL.patch_len
    seq_len = cfg.MODEL.seq_len
    pred_len = cfg.DATA.PRED_LEN
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
            outputs = model(seq, None, None, None)  # 使用了内置的scaler！！！
            seq = torch.cat([seq, outputs[:, -patch_len:, :]], dim=1)
    if dis != 0:
        seq = seq[:, :-dis, :]  # 去掉pred多余的部分
    _pred_total = seq   # .detach().cpu().numpy()
    pred_total = _pred_total[:, pad_len:, :]    

    pred = pred_total[:, -cfg.DATA.PRED_LEN:, cfg.DATA.TARGET_START_IDX:]
    
    if norm_method == 'SAN':
        pred = norm_module.de_normalize(pred, statistics)
    elif norm_method == 'RevIN':
        pred = norm_module(pred, 'denorm')
    elif norm_method == 'DishTS':
        pred = norm_module(pred, 'inverse')
    else:  # De-Normalization from Non-stationary Transformer
        pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
        pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
    
    return pred, ground_truth
