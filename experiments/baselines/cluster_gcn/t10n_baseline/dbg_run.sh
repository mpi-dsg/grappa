# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

python train.py --dataset ogbpr --data_prefix ./data/ --nomultilabel --num_layers 2  --hidden1 128 --learning_rate 0.003 --num_clusters 1500 --bsize 20 --dropout 0.5 --weight_decay 0.0001  --early_stopping 1000 --num_clusters_val 20 --num_clusters_test 1 --epochs 500 --save_name ./ckpt/ogbprmodel --diag_lambda 0.0001 --novalidation 2>&1 | tee dbg_run.txt
