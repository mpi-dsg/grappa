#! /usr/bin/python3
import os

from t10n.dataset.meta import name_to_meta

DEFAULT_FANOUT_OF_LAYER = {
    2: "25,10",
    3: "15,10,5",
    4: "20,15,10,5",
    5: "25,20,15,10,5"
}


def cmd_full(tag_id: str, graph_name: str, ip_config: str, part_config: str,
             n_classes: int, backend: str, num_gpus: int, model: int,
             stop_at_border: bool, disable_backup_server: bool, num_epochs: int,
             num_hidden: int, num_layers: int, fan_out: str, batch_size: int,
             batch_size_eval: int, log_every: int, eval_every: int, lr: float,
             dropout: float, local_rank: int, standalone: bool, pad_data: bool,
             net_type: bool, resume_path: str, checkpoint_path: str,
             checkpoint_every: str):
    pass


def cmd(tag_id: str,
        graph_name: str,
        part_config: str,
        model: str,
        num_layers: int,
        batch_size: int,
        batch_size_eval: int,
        eval_every: int,
        num_epochs: int,
        disable_backup_server: bool,
        stop_at_the_border: bool,
        ip_config: str,
        n_gpu_per_machine: int = 1,
        resume_path=None,
        checkpoint_path=None,
        checkpoint_every=-1):
    fan_out = DEFAULT_FANOUT_OF_LAYER[num_layers]
    meta_o = name_to_meta(graph_name)

    #job_cmd = "$PYTHON train/dgl_train.py"
    job_cmd = "python3 train/dgl_train.py"
    job_cmd += f" --model {model}"
    job_cmd += f" --num_layers {num_layers}"
    job_cmd += f" --fan_out {fan_out}"
    job_cmd += f" --n_classes {meta_o.n_label}"

    job_cmd += f" --num_epochs {num_epochs}"
    job_cmd += f" --batch_size {batch_size}"

    job_cmd += f" --eval_every {eval_every}"
    job_cmd += f" --batch_size_eval {batch_size_eval}"

    job_cmd += f" --graph_name {graph_name}"
    job_cmd += f" --part_config {part_config}"

    if resume_path is not None:
        job_cmd += f" --resume_path {resume_path}"
    if checkpoint_path is not None:
        job_cmd += f" --checkpoint_path {checkpoint_path}"
    if checkpoint_every > -1:
        job_cmd += f" --checkpoint_every {checkpoint_every}"

    job_cmd += f" --num_gpus {n_gpu_per_machine}"
    job_cmd += f" --ip_config {ip_config}"

    return job_cmd


def cmd_logging_suffix(tag_id: str, log_dir: str):
    suffix = f" 2>&1 | tee -a {log_dir}/{tag_id}.log"
    return suffix


def cmd_launcher_prefix(
    tag_id: str,
    workspace: str,
    n_parts: int,
    part_config: str,
    use_first_n_parts: int,
    num_machine: int,
    ip_config: str,
    n_gpu_per_machine: int,
):
    #prefix = "python3 $DATA_ROOT/work/ds4gnn/WS/DGL/launch.py"
    prefix = "python3 launch_ea48ce7.py"
    prefix += f" --workspace {workspace}"

    actual_part_n = n_parts
    N_SERVER_PER_MACHINE = int(actual_part_n / num_machine)
    N_TRAINER_PER_MACHINE = int(N_SERVER_PER_MACHINE) * n_gpu_per_machine
    N_SAMPLER_PER_TRAINER = 0  # launch.py uses N + 1

    prefix += f" --part_config {part_config}"
    prefix += f" --ip_config {ip_config}"

    prefix += f" --num_servers {N_SERVER_PER_MACHINE}"
    prefix += f" --num_trainers {N_TRAINER_PER_MACHINE}"
    prefix += f" --num_samplers {N_SAMPLER_PER_TRAINER}"

    return prefix


def cmd_part_conf_path(
    graph_name: str,
    part_algo: str,
    part_hop: int,
    n_parts: int,
    use_first_n_parts: int = -1,
):
    actual_part_n = use_first_n_parts if use_first_n_parts > 0 else n_parts
    conf = f"/data/dgl/{graph_name}"
    #conf = f"$DATA_PATH/dgl/{graph_name}"
    conf += f"/data_part_n{n_parts}_{part_algo}"
    conf += f"_{part_hop}_{actual_part_n}/{graph_name}.json"
    return conf


def cmd_ip_config(
    work_dir: str,
    num_machine: int,
):
    ip_conf = f"{work_dir}/hosts_docker_a_{num_machine}"
    return ip_conf


################################
# launch a training job
################################
N_MACHINE = 4
N_GPU_PER_MACHINE = 1

PART_ALGO = 'metis'
N_PART = 4
USE_FIRST_N = -1
N_HALO_HOP = 1

BATCH_SIZE = 1000
BATCH_SIZE_EVAL = 1000

N_EPOCH = 500
EVAL_EVERY = 100

DIS_BACKUP_SERVER = True
STOP_AT_THE_BORDER = False

for REPEAT in [1 , 2, 3]:
    for GRAPH_NAME in ['reddit', 'ogbpr', 'ogbar', 'cora', 'ogbpa']:
        for MODEL in ['gcn', 'sage', 'gat']:
            for N_LAYER in [2, 3, 4]:
                if GRAPH_NAME == 'ogbpa' and REPEAT > 1:
                    continue
                if GRAPH_NAME == 'ogbpa' and N_LAYER > 3:
                    continue

                tag_id = f"dgl{PART_ALGO}_{GRAPH_NAME}_{MODEL}_{N_LAYER}_{N_PART}_R{REPEAT}"
                work_dir = os.path.abspath(os.path.dirname(__file__))

                ip_config = cmd_ip_config(work_dir=work_dir,
                                          num_machine=N_MACHINE)
                part_conf = cmd_part_conf_path(graph_name=GRAPH_NAME,
                                               part_algo=PART_ALGO,
                                               part_hop=N_HALO_HOP,
                                               n_parts=N_PART,
                                               use_first_n_parts=-1)

                prefix = cmd_launcher_prefix(
                    tag_id=tag_id,
                    workspace=work_dir,
                    n_parts=N_PART,
                    part_config=part_conf,
                    use_first_n_parts=USE_FIRST_N,
                    num_machine=N_MACHINE,
                    ip_config=ip_config,
                    n_gpu_per_machine=N_GPU_PER_MACHINE)

                job_cmd = cmd(tag_id=tag_id,
                              graph_name=GRAPH_NAME,
                              part_config=part_conf,
                              model=MODEL,
                              num_layers=N_LAYER,
                              batch_size=BATCH_SIZE,
                              batch_size_eval=BATCH_SIZE_EVAL,
                              eval_every=EVAL_EVERY,
                              num_epochs=N_EPOCH,
                              disable_backup_server=DIS_BACKUP_SERVER,
                              stop_at_the_border=STOP_AT_THE_BORDER,
                              ip_config=ip_config,
                              n_gpu_per_machine=N_GPU_PER_MACHINE)

                suffix = cmd_logging_suffix(tag_id=tag_id,
                                            log_dir=f"{work_dir}/log")

                cluster_launch_job = f"{prefix} \"{job_cmd}\" {suffix}"
                os.system(cluster_launch_job)
