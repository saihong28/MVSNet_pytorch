#!/usr/bin/env bash
DTU_TESTING="/home/liboyang/桌面/SLAM/MVSNet_pytorch/dtu_mvs/processed/mvs_testing/dtu/"
CKPT_FILE="./pretrained/model_000014.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
