import os
import time

MPI_RUN_CMD_PREFIX = f"mpirun --allow-run-as-root --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include eth0"
#ROC="/workspace/t10n/exp/roc/t10n_baseline/gnn"
ROC = "gnn"
HOSTS = "/workspace/t10n/exp/neutron_star/t10n_baseline/hosts_docker"

LR = 0.003
DECAY = 0.0001
DECAY_RATE = 0.97
DROPOUT = 0.5
EPOCH = 300

DATASET = 'reddit'
LAYERS = {'reddit': '602-128-41'}
DATA_PATH = f"/data/roc/{DATASET}"

PARAMS = f"-ll:gpu 1 -ll:cpu 8 -ll:fsize 12000 -ll:zsize 30000 -lr {LR} -decay {DECAY} -decay-rate {DECAY_RATE} -dropout {DROPOUT} "

for REPEAT in [1, 2, 3]:
    for DATASET in ['reddit', 'ogbar', 'ogbpr']:
        NP = 16
        CMD = f"{MPI_RUN_CMD_PREFIX} -np {NP} --hostfile {HOSTS} {ROC} {PARAMS} -layers {LAYERS[DATASET]} -file {DATA_PATH} -e {EPOCH} 2>&1 | tee ./log/roc_{DATASET}_roc_2_{NP}_R{REPEAT}_0.log"
        os.system(CMD)
        exit(0)
