# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

echo $CUDA_VISIBLE_DEVICES

GATING_INIT=0.05
LOSS_ALPHA=0.1
LOW_RANK=16
TTA=PETSA
echo $CUDA_VISIBLE_DEVICES


ckpt_path="/data/qiuyunzhong/CKPT/Timer_forecast_1.0.ckpt"
DATASET="ETTh1"
datafold="ETT-small"
datapath="ETTh1.csv"
PRED_LEN=96
MODEL="Timer-LOTSA"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
RESULT_DIR="./results/${TTA}/"
BASE_LR=0.001
WEIGHT_DECAY=0.0001

OUTPUT_DIR="logs-wd0.0001/${TTA}/${MODEL}/${DATASET}"
mkdir -p "${OUTPUT_DIR}"

OUTPUT="${OUTPUT_DIR}/res.txt"

# ---------- 打印所有参数 ----------
{
echo "========== Parameter Dump =========="
echo ckpt_path             : $ckpt_path
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
echo "LOSS_ALPHA           : $LOSS_ALPHA"
echo "GATING_INIT          : $GATING_INIT"
echo "===================================="
} >> "${OUTPUT}"
# -----------------------------------

for PRED_LEN in 24 48 96 192 336 720; do
# for GATING_INIT in 0.01 0.05 0.1 0.3; do #0.01,0.05,0.1,and0.3
# for BASE_LR in 0.005 0.003 0.001 0.0005 0.0001; do
printf '\n\n========== PRED_LEN: %s ==========\n' "${PRED_LEN}" >> "${OUTPUT}" 2>&1
# printf '========== GATING_INIT: %s ==========\n' "${GATING_INIT}" >> "${OUTPUT}" 2>&1
# printf '========== BASE_LR: %s ==========\n' "${BASE_LR}" >> "${OUTPUT}" 2>&1
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
echo "CHECKPOINT_DIR       : $CHECKPOINT_DIR"
python main.py DATA.NAME ${DATASET} \
    SEED 1 \
    VISIBLE_DEVICES 4 \
    device 'cuda:4' \
    DATA.PRED_LEN ${PRED_LEN} \
    DATA.fold ${datafold} \
    DATA.path ${datapath} \
    MODEL.NAME ${MODEL} \
    MODEL.ckpt_path ${ckpt_path} \
    MODEL.pred_len ${PRED_LEN} \
    MODEL.ckpt_path ${ckpt_path} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.PETSA.GATING_INIT ${GATING_INIT} \
    TTA.PETSA.RANK ${LOW_RANK} \
    TTA.PETSA.LOSS_ALPHA ${LOSS_ALPHA} \
    RESULT_DIR ${RESULT_DIR} >> ${OUTPUT}
done
# done
# done