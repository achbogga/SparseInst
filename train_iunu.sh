#!/bin/bash
input_size=512
NUM_CLASSES=3
weights_path="/home/aboggaram/models/sparse_inst_r50_giam_aug_2b7d68.pth"
config_file="/home/aboggaram/projects/SparseInst/configs/sparse_inst_r50_giam_fp16.yaml"
input_folder="/home/aboggaram/data/Octiva/test_images_for_sparse_inst/*"
output_folder="/home/aboggaram/data/Octiva/output_images_from_sparse_inst"
python3 /home/aboggaram/projects/SparseInst/tools/train_net.py \
--config-file ${config_file} \
--num-gpus 2 \
--input ${input_folder} \
--output ${output_folder} \
--opt MODEL.WEIGHTS ${weights_path} INPUT.MIN_SIZE_TEST ${input_size} SOLVER.AMP.ENABLED True
