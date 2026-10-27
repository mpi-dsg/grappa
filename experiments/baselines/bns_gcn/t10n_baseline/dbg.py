#!/usr/bin/python3

import multiprocessing as mp
import subprocess  # for run command
import glob  # for enumetating files
import os  # create directory

import re  # for regex
import numpy as np
import pandas as pd  # for data processing


def shell_or(cmd, default=None):
    # https://stackoverflow.com/questions/4256107/running-bash-commands-in-python
    try:
        normal = subprocess.run([cmd],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True,
                                shell=True,
                                text=True)
        return normal
    except:
        print(f"FAIL  : {cmd}")
        return default


def run_rcmd(cmd, hosts):
    pl = []
    for h in hosts:
        proc = mp.Process(target=shell_or, args=(f"ssh {h} \"{cmd}\"",))
        pl.append(proc)
        proc.start()

    for p in pl:
        p.join()


def start_train(N_PARTITIONS, MODEL, SAMPLING_RATE, HOSTS, DATASET, REPEAT=1):
    assert DATASET in ['reddit', 'ogbn-products', 'ogbn-arxiv']
    assert MODEL in ['gcn', 'gat', 'graphsage']
    pl = []
    rank = 0
    N_LAYER = 2
    for h in HOSTS:
        T_CMD=f"cd /workspace/t10n/exp/bns_gcn/build/BNS-GCN/ ; python3 main.py" \
            f" --run_idx {REPEAT}" \
            f" --dataset {DATASET}" \
            f" --skip-partition" \
            f" --data_path /data/ds_ori/" \
            f" --part_path /data/bns_gcn/" \
            f" --dropout 0.5" \
            f" --lr 0.003" \
            f" --weight-decay 0.0001" \
            f" --n-partitions {N_PARTITIONS}" \
            f" --n-epochs 500" \
            f" --model {MODEL}" \
            f" --sampling-rate {SAMPLING_RATE}" \
            f" --n-layers {N_LAYER}" \
            f" --n-hidden 128" \
            f" --log-every 3" \
            f" --use-pp" \
            f" --fix-seed" \
            f" --parts-per-node 1" \
            f" --master-addr {HOSTS[0]}" \
            f" --node-rank {rank}" \
            f" 2>&1 | tee /workspace/t10n/exp/bns_gcn/t10n_baseline/log/bnsgcndbg_{DATASET}_{MODEL}_{N_LAYER}_{N_PARTITIONS}_R{REPEAT}_{rank}.txt"

        rank = rank + 1
        proc = mp.Process(target=shell_or, args=(f"ssh {h} \"{T_CMD}\"",))
        pl.append(proc)
        proc.start()

    for p in pl:
        p.join()


def start_train_ogbpa(N_PARTITIONS,
                      MODEL,
                      SAMPLING_RATE,
                      HOSTS,
                      DATASET,
                      REPEAT=1):
    assert DATASET == "ogbn-papers100m"
    assert MODEL in ['gcn', 'gat', 'graphsage']
    pl = []
    rank = 0
    for h in HOSTS:
        T_CMD=f"cd /workspace/t10n/exp/bns_gcn/build/BNS-GCN/ ; python3 main.py" \
            f" --run_idx {REPEAT}" \
            f" --dataset {DATASET}" \
            f" --partition-method random" \
            f" --skip-partition" \
            f" --data_path /data/ds_ori/" \
            f" --part_path /data/bns_gcn/" \
            f" --dropout 0.5" \
            f" --lr 0.003" \
            f" --weight-decay 0.0001" \
            f" --n-partitions {N_PARTITIONS}" \
            f" --n-epochs 500" \
            f" --model {MODEL}" \
            f" --sampling-rate {SAMPLING_RATE}" \
            f" --n-layers 3" \
            f" --n-hidden 128" \
            f" --log-every 1" \
            f" --use-pp" \
            f" --fix-seed" \
            f" --parts-per-node 1" \
            f" --master-addr {HOSTS[0]}" \
            f" --node-rank {rank}" \
            f" 2>&1 | tee /workspace/t10n/exp/bns_gcn/t10n_baseline/log/bnsgcn_{DATASET}_{MODEL}_3_{N_PARTITIONS}_R{REPEAT}_{rank}.txt"
        # use default instead of --inductive
        rank = rank + 1
        proc = mp.Process(
            target=shell_or,
            args=
            (f"ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ServerAliveCountMax=3 {h} \"{T_CMD}\"",
            ))
        pl.append(proc)
        proc.start()


#for N_PARTITIONS in [16, 8, 4, 2, 1]:
for N_PARTITIONS in [16]:
    host_f = f"/workspace/t10n/exp/t10n/hosts_docker_{N_PARTITIONS}"
    HOSTS = []
    with open(host_f, 'r') as f:
        for r in f.readlines():
            HOSTS.append(r.strip())
    print(HOSTS)
    #for REPEAT in [1, 2, 3]:
    for REPEAT in [1]:
        #for DATASET in ['ogbn-arxiv', 'reddit', 'ogbn-products']:
        for DATASET in [ 'reddit', 'ogbn-products']:
            for MODEL in ['gcn']:
                SAMPLING_RATE = 0.1
                start_train(N_PARTITIONS,
                            MODEL,
                            SAMPLING_RATE,
                            HOSTS,
                            DATASET,
                            REPEAT=REPEAT)

        if DATASET == "ogbn-papers100m":
            pass
