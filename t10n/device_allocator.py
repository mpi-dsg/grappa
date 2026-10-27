import torch as th


class DeviceAllocCPUServing:

    def __init__(self, train_device, backend) -> None:
        assert str(train_device)[:4] == 'cuda'
        # https://pytorch.org/docs/stable/distributed.html#backends
        self.device = th.device("cpu")
        self.train_device = train_device
        self.backend = backend

    def part_topo_dev(self):
        return self.device

    def part_feat_dev(self):
        # feat include label, mask as well
        return self.device

    def comm_dev(self):
        if self.backend == 'nccl':
            return self.train_device
        else:
            return self.device

class DeviceAllocGPUServing:

    def __init__(self, train_device, backend) -> None:
        assert str(train_device)[:4] == 'cuda'
        # https://pytorch.org/docs/stable/distributed.html#backends
        self.device = train_device
        self.train_device = train_device
        self.backend = backend

    def part_topo_dev(self):
        return self.device

    def part_feat_dev(self):
        # feat include label, mask as well
        return self.device

    def comm_dev(self):
        if self.backend == 'nccl':
            return self.train_device
        else:
            return self.device


import pynvml  # https://pypi.org/project/nvidia-ml-py/
import psutil
import os
import math
from t10n.util import shell_or


class T10nDeviceAllocator:

    def __init__(self, gpus_per_node) -> None:
        pynvml.nvmlInit()
        self.gpn = gpus_per_node
        gpu_n = pynvml.nvmlDeviceGetCount()
        assert self.gpn == gpu_n
        self.mode = "isolated"
        pynvml.nvmlShutdown()

    def get_local_gpu_id(self, world_size, rank):
        return rank % self.gpn

    def get_numa_affinity(self, local_gpu_id):
        # pynvml.nvmlDeviceGetNumaNodeId (handle) is not supported
        cmd_str = f"nvidia-smi topo -C -i {local_gpu_id}"
        maybe = shell_or(cmd_str)
        if maybe is None:
            return -1
        else:
            nid = -1
            try:
                nid = maybe.stdout.split(':')[1]
                nid = int(nid)
            except Execution as e:
                pass
            finally:
                return nid

    def get_local_cpu_num(self):
        num_logical_cpus = psutil.cpu_count(logical=True)
        return num_logical_cpus

    def get_local_cpu_ids(self, world_size, rank):
        # return [0, 1, 2, 3, 4, 5, 6, 7]
        gpu_id = self.get_local_gpu_id(world_size, rank)
        # print(f"get_local_cpu_ids: gpu_id={gpu_id}")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        try:
            num_ele = math.ceil(self.get_local_cpu_num() / 64)
            affinity = pynvml.nvmlDeviceGetCpuAffinity(handle, num_ele)
            cpu_affinity = ''.join(f"{x:064b}" for x in affinity)
            al = [int(x) for x in cpu_affinity]
            al.reverse()
        except pynvml.NVMLError as e:
            pynvml.nvmlShutdown()
            cpu_affinity = "N/A"

        cpu_num = self.get_local_cpu_num()
        local_cpus = al[:cpu_num]
        results = []
        for idx, e in enumerate(local_cpus):
            if e > 0:
                results.append(idx)
        pynvml.nvmlShutdown()
        return results

    def set_omp_dev_assignments(self,
                                world_size,
                                rank,
                                cg_idx=0,
                                dup_per_gpu=1):
        lgid = self.get_local_gpu_id(world_size, rank)
        assert lgid == rank % self.gpn
        assert world_size % self.gpn == 0

        lcpus = self.get_local_cpu_ids(world_size, rank)
        # print(lcpus)
        if self.mode == "isolated":
            if dup_per_gpu > 1:
                stride = len(lcpus) // dup_per_gpu
                assigned_cpus = lcpus[cg_idx * stride:cg_idx * stride + stride]
            else:
                stride = len(lcpus)
                assigned_cpus = lcpus
            # print(f"assigned_cpus: {assigned_cpus}")
            os.environ['OMP_PROC_BIND'] = 'spread'
            os.environ['OMP_PLACES'] = ",".join([str(e) for e in assigned_cpus])
            os.environ['OMP_NUM_THREADS'] = str(stride)
        else:
            assert False
