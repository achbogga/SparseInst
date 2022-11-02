#!/bin/bash
config_file="/home/aboggaram/projects/SparseInst/configs/sparse_inst_r50_giam_fp16_for_onnx.yaml"
cd /home/aboggaram/projects/SparseInst && \
python3 onnx/convert_onnx.py \
--config-file ${config_file}
