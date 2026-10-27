import os
import time

MPI_RUN_CMD_PREFIX = f"mpirun --allow-run-as-root --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include eth0"
NTS = "/workspace/t10n/exp/neutron_star/build/NeutronStarLite/build/nts"

#for REPEAT in [1, 2, 3]:
for REPEAT in [1]:
    for DATASET in ['reddit', 'ogbpr', 'ogbar']:  # 'cora',
        for MODEL in ['gcn', 'gat']:
            #for NP in [16, 8]:
            for NP in [16]:
                HOSTS = f"hosts_docker_{NP}"
                CMD = f"{MPI_RUN_CMD_PREFIX} -np {NP} --hostfile {HOSTS} {NTS} cfg_file/{MODEL}_{DATASET}.cfg 2>&1 | tee ./log/dbg_nts_{DATASET}_{MODEL}_2_{NP}_R{REPEAT}_0.log"
                os.system(CMD)
                exit(0)
#############################33
# NP = 4 ?
