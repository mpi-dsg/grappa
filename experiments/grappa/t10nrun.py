#!/usr/bin/python3
#  Copyright (c) 2024-2026 by MPI-SWS, Germany. All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

import argparse, os, socket

from t10n.util import cluster_run, shell_or


def test(host_file, tag_id):

    def cmd_func(hosts, host_idx):
        test_cmd = f"hostname 2>&1 | tee ./log/{tag_id}_{host_idx}.txt"
        return test_cmd

    work_dir = os.path.abspath(os.path.dirname(__file__))
    cluster_run(host_file, tag_id, cmd_func, work_dir=work_dir)


def kill_all(host_file, tag_id):

    kill_cmd = f"pgrep -f {tag_id} | xargs kill -9; exit"

    def cmd_func(hosts, host_idx):
        if hosts[host_idx] == socket.gethostname():
            return "hostname"
        return kill_cmd

    kill_tag_id = f"kill_all_{tag_id}"
    work_dir = os.path.abspath(os.path.dirname(__file__))
    cluster_run(host_file, kill_tag_id, cmd_func, work_dir=work_dir)

    shell_or(kill_cmd, tag_id)


def start_inline(host_file, tag_id, cmd):

    def cmd_func(hosts, idx):
        return cmd

    work_dir = os.path.abspath(os.path.dirname(__file__))
    cluster_run(host_file, tag_id, cmd_func, work_dir=work_dir)

def start_train_slurm(host_file, tag_id, run_train_sh, num_proc):
    #WHICH_PYTHON = "$PYTHON"
    WHICH_PYTHON = "python3"
    # WHICH_PYTHON = "$PYTHON3"
    # DATA_PATH = "$WORK_ROOT/exp/t10n/test"
    # DATA_PATH = '$DATA_PATH/t10n'
    DATA_PATH = '/data/t10n/tmp/'
    #PART_METHOD = "metis"
    NUM_PART = 16

    N_EPOCH = 51
    N_EVAL = 50
    # make sure #batch is the same across ranks both for train and infer
    BS = 1000
    EVAL_BS = 200000
    #EVAL_BS = 20000
    GPU_PER_NODE = num_proc  # 1 in docker, create_swarm_cluster uses 1

    #for PART_METHOD in ["metis", "random"]:
    for PART_METHOD in ["random"]:

        for GRAPH_NAME in [
                'reddit',
                #'cora',
                #'ogbpr',
                #'ogbar',
                #'ogbpa',
                #'igb260m',
                #'rmat_26_part1'
        ]:
            for RUN in [1]:  # repeat 3 runs
                for MODEL in [
                    #'gin'
                    'gcn',
                    #'sage',
                    #'gat'
                ]:
                    kk = [
                    #   (1, 10),
                        (2, "25,10"),
                    #   (3, "15,10,5"),
                    #   (4, "20,15,10,5"),
                    ]
                    for N_LAYER, FAN_OUT in kk:
                        #tag_id = f"t10nonlyinrandom_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        #tag_id = "dbg_reddit"
                        #tag_id = f"t10nresampling{PART_METHOD}_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        # tag_id = f"gcnfull_{PART_METHOD}_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        tag_id = f"slurm_loaddata_{PART_METHOD}_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        #tag_id = f"t10npresentation_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"

                        def cmd_func(hosts, host_idx):
                            with open(run_train_sh, 'r') as sh_cmd:
                                lines = sh_cmd.readlines()
                            cmd = "".join(lines)
                            cmd = cmd.replace("\\", "")
                            cmd = cmd.replace("$PROC_PER_NODE", f"{num_proc}")
                            cmd = cmd.replace("$N_NODES", str(len(hosts)))
                            cmd = cmd.replace("$NODE_RANK", str(host_idx))
                            cmd = cmd.replace("$MASTER_ADDR", hosts[0])
                            cmd = cmd.replace("$MASTER_PORT", "9977")

                            cmd = cmd.replace("$WHICH_PYTHON", f"{WHICH_PYTHON}")
                            cmd = cmd.replace("$DATA_PATH", f"{DATA_PATH}")
                            cmd = cmd.replace("$GRAPH_NAME", f"{GRAPH_NAME}")
                            cmd = cmd.replace("$PART_METHOD", f"{PART_METHOD}")
                            cmd = cmd.replace("$NUM_PART", f"{NUM_PART}")
                            cmd = cmd.replace("$MODEL", f"{MODEL}")
                            cmd = cmd.replace("$N_LAYER", f"{N_LAYER}")
                            cmd = cmd.replace("$FAN_OUT", f"{FAN_OUT}")
                            cmd = cmd.replace("$N_EPOCH", f"{N_EPOCH}")
                            cmd = cmd.replace("$BS", f"{BS}")
                            cmd = cmd.replace("$EVAL_BS", f"{EVAL_BS}")
                            cmd = cmd.replace("$GPU_PER_NODE", f"{GPU_PER_NODE}")
                            cmd = cmd.replace("$N_EVAL", f"{N_EVAL}")

                            cmd = cmd.replace("$TAG_ID", f"{tag_id}")
                            #cur_dir = os.path.abspath(os.path.dirname(__file__))
                            #cmd = cmd.replace("$LOG_DIR", f"{cur_dir}/log")
                            #print(cmd)
                            return cmd

                        work_dir = os.path.abspath(os.path.dirname(__file__))
                        cluster_run(host_file,
                                    tag_id,
                                    cmd_func,
                                    work_dir=work_dir,
                                    timeout=True)
                        kill_all(host_file, tag_id)
                        kill_all(host_file, tag_id)
                        kill_all(host_file, tag_id)

                        # exit(0)
                    # for N_LAYER end
                # for MODEL end
            # for RUN end

def start_train(host_file, tag_id, run_train_sh, num_proc):
    # WHICH_PYTHON = "$PYTHON"
    WHICH_PYTHON = "python3"
    # WHICH_PYTHON = "$PYTHON3"
    # DATA_PATH = "$WORK_ROOT/exp/t10n/test"
    # DATA_PATH = '$DATA_PATH/t10n'
    DATA_PATH = '/data/t10n/tmp/'
    #PART_METHOD = "metis"
    NUM_PART = 16

    N_EPOCH = 500
    N_EVAL = 50
    # make sure #batch is the same across ranks both for train and infer
    BS = 1000
    EVAL_BS = 200000
    #EVAL_BS = 20000
    GPU_PER_NODE = num_proc  # 1 in docker, create_swarm_cluster uses 1

    #for PART_METHOD in ["metis", "random"]:
    for PART_METHOD in ["random"]:

        for GRAPH_NAME in [
                'reddit',
                #'cora',
                #'ogbpr',
                #'ogbar',
                #'ogbpa',
                #'igb260m',
                #'rmat_31_part1'
        ]:
            for RUN in [1, 2, 3]:  # repeat 3 runs
                for MODEL in [
                    #'gin'
                    'gcn',
                    # 'sage',
                    #'gat'
                ]:
                    kk = [
                    #   (1, 10),
                    #    (2, "25,10"),
                       (3, "15,10,5"),
                    #   (4, "20,15,10,5"),
                    ]
                    for N_LAYER, FAN_OUT in kk:
                        #tag_id = f"t10nonlyinrandom_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        #tag_id = "dbg_reddit"
                        #tag_id = f"t10nresampling{PART_METHOD}_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        # tag_id = f"gcnfull_{PART_METHOD}_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        tag_id = f"eurosys26_{PART_METHOD}_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"
                        #tag_id = f"t10npresentation_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{NUM_PART}_R{RUN}"

                        def cmd_func(hosts, host_idx):
                            with open(run_train_sh, 'r') as sh_cmd:
                                lines = sh_cmd.readlines()
                            cmd = "".join(lines)
                            cmd = cmd.replace("\\", "")
                            cmd = cmd.replace("$PROC_PER_NODE", f"{num_proc}")
                            cmd = cmd.replace("$N_NODES", str(len(hosts)))
                            cmd = cmd.replace("$NODE_RANK", str(host_idx))
                            cmd = cmd.replace("$MASTER_ADDR", hosts[0])
                            cmd = cmd.replace("$MASTER_PORT", "9977")

                            cmd = cmd.replace("$WHICH_PYTHON", f"{WHICH_PYTHON}")
                            cmd = cmd.replace("$DATA_PATH", f"{DATA_PATH}")
                            cmd = cmd.replace("$GRAPH_NAME", f"{GRAPH_NAME}")
                            cmd = cmd.replace("$PART_METHOD", f"{PART_METHOD}")
                            cmd = cmd.replace("$NUM_PART", f"{NUM_PART}")
                            cmd = cmd.replace("$MODEL", f"{MODEL}")
                            cmd = cmd.replace("$N_LAYER", f"{N_LAYER}")
                            cmd = cmd.replace("$FAN_OUT", f"{FAN_OUT}")
                            cmd = cmd.replace("$N_EPOCH", f"{N_EPOCH}")
                            cmd = cmd.replace("$BS", f"{BS}")
                            cmd = cmd.replace("$EVAL_BS", f"{EVAL_BS}")
                            cmd = cmd.replace("$GPU_PER_NODE", f"{GPU_PER_NODE}")
                            cmd = cmd.replace("$N_EVAL", f"{N_EVAL}")

                            cmd = cmd.replace("$TAG_ID", f"{tag_id}")
                            #cur_dir = os.path.abspath(os.path.dirname(__file__))
                            #cmd = cmd.replace("$LOG_DIR", f"{cur_dir}/log")
                            #print(cmd)
                            return cmd

                        work_dir = os.path.abspath(os.path.dirname(__file__))
                        cluster_run(host_file,
                                    tag_id,
                                    cmd_func,
                                    work_dir=work_dir,
                                    timeout=True)
                        kill_all(host_file, tag_id)
                        kill_all(host_file, tag_id)
                        kill_all(host_file, tag_id)

                        # exit(0)
                    # for N_LAYER end
                # for MODEL end
            # for RUN end


def start_partition(host_file, tag_id, partition_script, num_proc):
    #DEST_PATH = '/more_data/'
    DEST_PATH = '/data/t10n/tmp/'
    SRC_PREFIX = '/data/t10n/'
    #DEST_PATH = '$TMP_ROOT/pa_tmp/igb/'
    #SRC_PREFIX = '$DATA_PATH/t10n'

    WHICH_PYTHON = "python3"
    #WHICH_PYTHON = "$PYTHON"

    #for GRAPH_NAME in ['igb260m']:
    for GRAPH_NAME in ['ogbpr']:
        for NUM_PART in [16]:
            tag_id = f"partition_{GRAPH_NAME}_{NUM_PART}"

            def cmd_func(hosts, host_idx):
                num_proc = NUM_PART // len(hosts)
                with open(partition_script, 'r') as sh_cmd:
                    lines = sh_cmd.readlines()
                    cmd = "".join(lines)
                    cmd = cmd.replace("\\", "")
                    cmd = cmd.replace("$PROC_PER_NODE", f"{num_proc}")
                    cmd = cmd.replace("$N_NODES", str(len(hosts)))
                    cmd = cmd.replace("$NODE_RANK", str(host_idx))
                    cmd = cmd.replace("$MASTER_ADDR", hosts[0])
                    cmd = cmd.replace("$MASTER_PORT", "9977")

                    cmd = cmd.replace("$WHICH_PYTHON", f"{WHICH_PYTHON}")

                    cmd = cmd.replace("$DEST_PATH", f"{DEST_PATH}")
                    cmd = cmd.replace("$GRAPH_NAME", f"{GRAPH_NAME}")
                    cmd = cmd.replace("$SRC_PREFIX", f"{SRC_PREFIX}")
                    cmd = cmd.replace("$NUM_PART", f"{NUM_PART}")

                    cmd = cmd.replace("$TAG_ID", f"{tag_id}")

                    return cmd

            work_dir = os.path.abspath(os.path.dirname(__file__))
            cluster_run(host_file,
                        tag_id,
                        cmd_func,
                        work_dir=work_dir,
                        timeout=True)


def main(args):
    if args.task == "test":
        test(args.host_file, args.tag_id)
    elif args.task == "kill":
        kill_all(args.host_file, args.tag_id)
    elif args.task == "inline":
        assert args.inline_cmd is not None, "need --inline_cmd"
        start_inline(args.host_file, args.tag_id, args.inline_cmd)
    elif args.task == "train":
        assert args.train_script is not None
        assert args.train_num_proc > -1
        start_train(args.host_file, args.tag_id, args.train_script,
                    args.train_num_proc)
    elif args.task == "train_slurm":
        assert args.train_script is not None
        assert args.train_num_proc > -1
        start_train_slurm(args.host_file, args.tag_id, args.train_script,
                          args.train_num_proc)
    elif args.task == "partition":
        start_partition(args.host_file, args.tag_id, args.partition_script,
                        args.partition_num_proc)
    else:
        assert False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t10n_launcher")
    ######################
    # dataset parameters
    ######################
    parser.add_argument("--host_file",
                        type=str,
                        default="hosts",
                        help="the files contains hosts in this cluster")

    parser.add_argument("--tag_id",
                        type=str,
                        required=True,
                        help="a unique id for this task across the cluster")

    parser.add_argument("--task",
                        required=True,
                        default='train',
                        choices=['train', 'kill', 'inline', 'partition'],
                        help="name of the partition method")

    parser.add_argument("--train_script",
                        type=str,
                        default="train.sh.template",
                        help="the .sh file that contains train command")
    parser.add_argument("--train_num_proc",
                        type=int,
                        default=-1,
                        help="= n_proc_per_node, num of gpus per node")

    parser.add_argument("--partition_script",
                        type=str,
                        default="partition.sh.template",
                        help="the .sh file that contains train command")
    parser.add_argument("--partition_num_proc",
                        type=int,
                        default=-1,
                        help="= n_proc_per_node, num of proc per node")

    parser.add_argument("--inline_cmd", type=str, help="run a inline command")

    args = parser.parse_args()

    print(args)
    main(args)
