#!/usr/bin/python3
#  Copyright (c) 2024-2026 by MPI-SWS, Germany. All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

import os
import argparse
import time
import json
import shutil

from t10n.util import shell_or, printg, timing
from t10n.util import cluster_run, parse_hosts_file

SSH_CMD = "ssh -o StrictHostKeyChecking=no"
SCP_CMD = "scp -o StrictHostKeyChecking=no"

CLUSTER_MANAGER_IP = os.environ['T10N_MANAGER_IP']
DATASET_PATH = os.environ['T10N_DATASET_PATH']
MORE_DATA_PATH = os.environ['T10N_MORE_DATA_PATH']
WORKSPACE_PATH = os.environ['T10N_WORKSPACE_PATH']

CLUSTER_IMG_NAME = "t10n_cu118"
DOCKER_FILE = "Dockerfile_cu118"
DUP = 1  # number of docker per GPU

CLUSTER_NET_NAME = f"{CLUSTER_IMG_NAME}_net"

SUBMIT_NODE_NAME = f"{CLUSTER_IMG_NAME}_submit"
WORKER_NODE_PRE = f"{CLUSTER_IMG_NAME}_w"


def check_requirements():
    maybe = shell_or("whoami")
    assert maybe is not None
    assert maybe.stdout.strip(
    ) == "root", f"docker swarm need root privilege to create cluster across physical nodes"
    disk_free = shutil.disk_usage(".")[2] / 1024 / 1024 / 1024
    assert disk_free > 16.0, "16GB free disk space is required for saved docker image file"


@timing
def cleanup_swarm_cluster(phy_host_file):
    printg("Cleaning up previous containers")
    shell_or(f"{SSH_CMD} {CLUSTER_MANAGER_IP} docker stop {SUBMIT_NODE_NAME}")
    shell_or(f"{SSH_CMD} {CLUSTER_MANAGER_IP} docker rm {SUBMIT_NODE_NAME}")

    hosts, n_gpus = parse_hosts_file(phy_host_file)
    worker_idx = 0
    for h, n_gpu in zip(hosts, n_gpus):
        print(f"cleaning {n_gpu*DUP} containers in host {h}")
        for li in range(n_gpu * DUP):
            WORKER = f"{WORKER_NODE_PRE}{worker_idx}"
            shell_or(f"{SSH_CMD} {h} docker stop {WORKER}")
            shell_or(f"{SSH_CMD} {h} docker rm {WORKER}")
            worker_idx += 1
        shell_or(
            f"{SSH_CMD} {h} docker swarm leave --force && docker network rm {CLUSTER_NET_NAME}"
        )


@timing
def cleanup_cluster_all(phy_host_file):
    printg("Cleaning up all containers in cluster")
    STOP_CMD = "docker ps -q -a | xargs docker stop"
    CLEAN_CMD = "docker ps -q -a | xargs docker rm"
    shell_or(f"{SSH_CMD} {CLUSTER_MANAGER_IP} {STOP_CMD}")
    shell_or(f"{SSH_CMD} {CLUSTER_MANAGER_IP} {CLEAN_CMD}")
    hosts, n_gpus = parse_hosts_file(phy_host_file)
    for h, n_gpu in zip(hosts, n_gpus):
        print(f"cleaning {n_gpu*DUP} containers in host {h}")
        shell_or(f"{SSH_CMD} {h} {STOP_CMD}")
        shell_or(f"{SSH_CMD} {h} {CLEAN_CMD}")

        shell_or(
            f"{SSH_CMD} {h} docker swarm leave --force && docker network rm {CLUSTER_NET_NAME}"
        )


@timing
def create_swarm_network(phy_host_file):
    printg("Creating swarm network")
    hosts, _ = parse_hosts_file(phy_host_file)
    for idx, h in enumerate(hosts):
        if idx == 0:
            maybe = shell_or(
                f"{SSH_CMD} {h} docker swarm init --advertise-addr {CLUSTER_MANAGER_IP}"
            )
            assert maybe is not None
            maybe = shell_or(f"{SSH_CMD} {h} docker swarm join-token worker")
            assert maybe is not None
            res = maybe.stdout.split("\n")
            worker_join = res[2].strip()
            #print(worker_join)
            maybe = shell_or(
                f"{SSH_CMD} {h} docker network create -d overlay --attachable {CLUSTER_NET_NAME}"
            )
            assert maybe is not None
        else:
            maybe = shell_or(f"{SSH_CMD} {h} {worker_join}")
            assert maybe is not None
            print(maybe.stdout)


@timing
def create_docker_image(
    build_dir,
    output_dir,
    image_name,
    tag="latest",
):
    printg("Creating docker containers")
    maybe = shell_or(
        f"cd {build_dir}; docker build -f ./{DOCKER_FILE} -t {image_name}:{tag} ."
    )
    assert maybe is not None
    maybe = shell_or(
        f"cd {build_dir}; docker save --output {output_dir}/{image_name}_{tag}.tar {image_name}:{tag}"
    )
    assert maybe is not None


@timing
def start_swarm_containers(phy_host_file, img_dir, image_name, tag="latest"):
    printg("Creating swarm contrainers")

    work_dir = os.path.abspath(os.path.dirname(__file__))

    def cmd_func(hosts, host_idx):
        load_cmd = f"docker load --input {img_dir}/{image_name}_{tag}.tar"
        return load_cmd

    cluster_run(host_file=phy_host_file,
                tag_id="create_swarm_container",
                cmd_func=cmd_func,
                work_dir=work_dir)

    hosts, n_gpus = parse_hosts_file(phy_host_file)
    worker_idx = 0
    for h, n_gpu in zip(hosts, n_gpus):
        print(f"creating {n_gpu*DUP} containers in host {h}")
        for li in range(n_gpu * DUP):
            WORKER = f"{WORKER_NODE_PRE}{worker_idx}"
            G_ID = f"{li%n_gpu}"
            NV_DEVICE_SPEC = "" #f"--device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia{G_ID}:/dev/nvidia{G_ID}"
            printg(f"{WORKER}: {NV_DEVICE_SPEC}")
            #RUN_IMG = f"docker run --cap-add SYS_NICE -v {DATASET_PATH}:/data  -v {MORE_DATA_PATH}:/more_data -v {WORKSPACE_PATH}:/workspace {NV_DEVICE_SPEC} --gpus device={G_ID} --shm-size=700g --name {WORKER} --network {CLUSTER_NET_NAME} -dit {image_name}:{tag}"
            RUN_IMG = f"docker run --cap-add SYS_NICE -v {DATASET_PATH}:/data  -v {MORE_DATA_PATH}:/more_data -v {WORKSPACE_PATH}:/workspace --shm-size=700g --name {WORKER} --network {CLUSTER_NET_NAME} -dit {image_name}:{tag}"
            SSH_RUN_IMG = f"{SSH_CMD} {h} {RUN_IMG}"
            maybe = shell_or(SSH_RUN_IMG, "docker_run_img")
            assert maybe is not None
            START_SSHD = f"{SSH_CMD} {h} docker exec {WORKER} bash -c \"/usr/sbin/sshd -D\""
            maybe = shell_or(START_SSHD, "docker_exec_sshd")
            assert maybe is not None
            worker_idx += 1


@timing
def dump_swarm_cluster_ip(phy_host_file, dump_file="hosts_docker"):
    work_dir = os.path.abspath(os.path.dirname(__file__))
    dump_file_path = work_dir + "/" + dump_file
    maybe = shell_or(f"> {dump_file_path}")
    assert maybe is not None

    hosts, _ = parse_hosts_file(phy_host_file)
    for h in hosts:
        inspect = f"{SSH_CMD} {h} docker network inspect {CLUSTER_NET_NAME}"
        maybe = shell_or(inspect, "docker_network_inspect")
        assert maybe is not None
        config = json.loads(maybe.stdout)
        config = config[0]['Containers']
        with open(dump_file_path, "a+") as hf:
            for k, v in config.items():
                name = v['Name']
                ipv4 = v['IPv4Address']
                if name.startswith(WORKER_NODE_PRE):
                    hf.write(ipv4.split('/')[0] + "\n")


@timing
def submit_node_start(image_name, tag="latest"):
    printg("Starting submit-node container...")
    RUN_SUBMIT_NODE = f"docker run --cap-add SYS_NICE -v {DATASET_PATH}:/data -v {WORKSPACE_PATH}:/workspace --shm-size=512g --name {SUBMIT_NODE_NAME} --network {CLUSTER_NET_NAME} -dit {image_name}:{tag}"
    maybe = shell_or(RUN_SUBMIT_NODE)
    assert maybe is not None


@timing
def submit_node_run(CMD):
    printg("Downloading ds4gnn")
    assert maybe is not None
    DOCKER_EXEC = f"docker exec {SUBMIT_NODE_NAME} bash -c \"{CMD}\""
    maybe = shell_or(DOCKER_EXEC)
    assert maybe is not None


parser = argparse.ArgumentParser()
parser.add_argument("--phy_hosts",
                    type=str,
                    required=True,
                    help="host file of physical cluster")
parser.add_argument("--build_dir",
                    type=str,
                    default=None,
                    required=False,
                    help="dockerfile directory where docker build runs")
parser.add_argument("--docker_img_dir",
                    type=str,
                    default=None,
                    required=False,
                    help="dockerfile directory where docker build runs")

args = parser.parse_args()

check_requirements()
cleanup_swarm_cluster(args.phy_hosts)
cleanup_cluster_all(args.phy_hosts)
exit(0)
create_swarm_network(args.phy_hosts)
if args.build_dir is not None:
    create_docker_image(args.build_dir,
                        output_dir=args.docker_img_dir,
                        image_name=CLUSTER_IMG_NAME)
if args.docker_img_dir is not None:
    start_swarm_containers(args.phy_hosts,
                           img_dir=args.docker_img_dir,
                           image_name=CLUSTER_IMG_NAME)
    dump_swarm_cluster_ip(args.phy_hosts)

    submit_node_start(image_name=CLUSTER_IMG_NAME)
