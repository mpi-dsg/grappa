#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

import os, time
import subprocess  # for run command
import multiprocessing as mp

import psutil, numa

import traceback
import pynvml


def timing(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Execution of {func.__name__} took {end_time - start_time:.6f} seconds"
        )
        return result

    return wrapper


def shell_or(cmd, tag_id="misc", default=None):
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
        print(f"FAIL: {tag_id}: {cmd}")
        return default


def green_str(in_str):
    GREEN = '\033[0;32m'
    NC = '\033[0m'  # No Color
    return f"{GREEN}{in_str}{NC}"


def printg(in_str):
    print(green_str(in_str))


def parse_hosts_file(hosts_file):
    hosts = []
    n_gpu = []
    with open(hosts_file, 'r') as hosts_f:
        for line in hosts_f.readlines():
            h = line.strip().split(" ")[0]  # vaolta09 n_gpu=2
            hosts.append(h)
            num = int(line.split("=")[1]) if "=" in line else -1
            n_gpu.append(num)
    return hosts, n_gpu


def cluster_run(host_file, tag_id, cmd_func, work_dir=".", timeout=True):
    #mp.set_start_method('spawn', force=True)
    print('-' * 50)
    print(f"{tag_id} starting...")
    procs = []
    hosts, _ = parse_hosts_file(host_file)
    for idx in range(len(hosts)):
        cmd = cmd_func(hosts, idx)
        cmd = cmd.replace("\n", " ")
        log_suffix = f"2>&1 | tee log/{tag_id}_{idx}.txt"
        full_cmd = f"cd {work_dir}; {cmd} {log_suffix}; exit"
        if timeout:
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ServerAliveCountMax=3 {hosts[idx]} \"{full_cmd}\""
        else:
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o {hosts[idx]} \"{full_cmd}\""

        #print(ssh_cmd)
        p = mp.Process(target=shell_or, args=(
            ssh_cmd,
            tag_id,
        ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print(f"{tag_id} end")


def mp_run(num_process, tag_id, cmd_func, work_dir="."):
    print('-' * 50)
    print(f"{tag_id} starting...")
    procs = []
    for idx in range(num_process):
        cmd = cmd_func(idx)
        cmd = cmd.replace("\n", " ")
        log_suffix = f"2>&1 | tee log/{tag_id}_{idx}.txt"
        mp_cmd = f"cd {work_dir}; {cmd} {log_suffix}; exit"
        p = mp.Process(target=shell_or, args=(
            mp_cmd,
            tag_id,
        ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print(f"{tag_id} end")


def pprint(any, *args, **kwargs):
    rank = any if isinstance(any, int) else any.rank
    if rank == 0:
        print(*args, **kwargs)


def backtrace(rank=0):
    if rank > 0:
        return
    for line in traceback.format_stack():
        print(line.strip())

def is_in_slurm():
    # Check for common Slurm environment variables
    slurm_vars = ["SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_JOB_NODELIST"]
    return any(var in os.environ for var in slurm_vars)

def set_py_affinity(node):
    return
    if not numa.available():
        print("NUMA is not available on this system")
    else:
        # Get the CPU mask for the specified NUMA node
        cpu_mask = numa.node_to_cpus(node)

        # Set CPU affinity for the process using psutil
        p = psutil.Process()
        p.cpu_affinity(cpu_mask)

        # Set memory policy to bind memory to the specified NUMA node
        numa.set_interleave_mask([node])


class PyNVML:

    def __init__(self) -> None:
        self.handles = None
        try:
            pynvml.nvmlInit()
            self.dev_n = pynvml.nvmlDeviceGetCount()
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.dev_n)
            ]
        except:
            pass

    def __del__(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def report_memory(self, rank, tag=None):
        if self.handles == None:
            return

        def b2m(number):
            return f"{float(number)/1024/1024:.2f}"

        if rank % self.dev_n == 0:
            infos = [
                pynvml.nvmlDeviceGetMemoryInfo(self.handles[i])
                for i in range(self.dev_n)
            ]
            headline = "memory\t" + "\t".join(
                [f"gpu{i}" for i in range(self.dev_n)])
            tot = "total \t" + "\t".join([b2m(e.total) for e in infos])
            usd = "used  \t" + "\t".join([b2m(e.used) for e in infos])
            fre = "free  \t" + "\t".join([b2m(e.free) for e in infos])
            print('-' * 50)
            if tag is not None:
                print(tag)
                print('-' * 50)
            print(headline)
            print(tot)
            print(usd)
            print(fre)


try:
    pyml_handle = PyNVML()
except:
    pass
