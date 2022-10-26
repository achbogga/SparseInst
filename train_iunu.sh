#!/bin/bash
config_file="/home/aboggaram/projects/SparseInst/configs/sparse_inst_r50_giam_fp16.yaml"
cd /home/aboggaram/projects/SparseInst && \
python3 tools/train_net.py \
--config-file ${config_file} \
--num-gpus 2
