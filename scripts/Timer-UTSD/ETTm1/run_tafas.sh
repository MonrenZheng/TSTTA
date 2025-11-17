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

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

echo $CUDA_VISIBLE_DEVICES

ckpt_path="/data/qiuyunzhong/ts_adaptive_inference/Timer/ckpt/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt"
MODEL="Timer-UTSD"
TTA=TAFAS
DATASET="ETTm1"
datafold="ETT-small"
datapath="ETTm1.csv"
PRED_LEN=96
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
RESULT_DIR="./results/${TTA}/"
BASE_LR=0.001
WEIGHT_DECAY=0.0
GATING_INIT=0.05

OUTPUT_DIR="logs/${TTA}/${MODEL}/${DATASET}"
mkdir -p "${OUTPUT_DIR}"

OUTPUT="${OUTPUT_DIR}/res.txt"

# ---------- 打印所有参数 ----------
{
echo "========== Parameter Dump =========="
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "DATASET              : $DATASET"
echo "datafold             : $datafold"
echo "datapath             : $datapath"
echo "PRED_LEN             : $PRED_LEN"
echo "MODEL                : $MODEL"
echo "RESULT_DIR           : $RESULT_DIR"
echo "OUTPUT               : $OUTPUT"
echo "BASE_LR              : $BASE_LR"
echo "WEIGHT_DECAY         : $WEIGHT_DECAY"
echo "GATING_INIT          : $GATING_INIT"
echo "===================================="
} >> "${OUTPUT}"
# -----------------------------------


for PRED_LEN in 24 48 96 192; do # 
printf '\n\n========== PRED_LEN: %s ==========\n' "${PRED_LEN}" >> "${OUTPUT}" 2>&1
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
echo "CHECKPOINT_DIR       : $CHECKPOINT_DIR"
python main.py DATA.NAME ${DATASET} \
    VISIBLE_DEVICES 7 \
    device 'cuda:7' \
    DATA.PRED_LEN ${PRED_LEN} \
    DATA.fold ${datafold} \
    DATA.path ${datapath} \
    MODEL.NAME ${MODEL} \
    MODEL.ckpt_path ${ckpt_path} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.TAFAS.GATING_INIT ${GATING_INIT} \
    RESULT_DIR ${RESULT_DIR} >> ${OUTPUT}
done