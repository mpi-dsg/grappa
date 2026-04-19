#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

from typing import Union, Optional

from t10n._C.graph import Batch
from t10n._C.graph import HostGraph
from t10n._C.graph import GpuGraph
from t10n._C.graph import HostSampler
from t10n._C.graph import GpuSampler

from t10n.dgl_dsg.compliant import to_dgl_batch
from t10n.util import timing

import torch

class BatchSampler:

    def __init__(self,
                 graph: Union[HostGraph, GpuGraph, None],
                 target_nodes,
                 fan_outs: list[int],
                 repeated: Optional[bool] = None,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 use_batch_n=-1):

        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

class HostBatchSampler(BatchSampler):

    def __init__(self,
                 graph: Union[HostGraph, GpuGraph, None],
                 target_nodes,
                 fan_outs: list[int],
                 repeated: Optional[bool] = None,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 use_batch_n=-1):

        assert torch.is_tensor(target_nodes), "only handle tensor input"

        self.graph = graph
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_nodes = target_nodes
        self.fan_outs = fan_outs
        self.repeated = repeated
        self.total_nodes_n = self.target_nodes.shape[0]
        self.total_batch = 0
        self.use_batch_n = use_batch_n

        self.batch_idx = 0

        self.host_sampler = HostSampler(graph, fan_outs, repeated)

    def __del__(self):
        self.host_sampler = None

    def _reinit_from(self, target_nodes, batch_size, use_batch_n):
        assert torch.is_tensor(target_nodes), "only handle tensor input"
        self.host_sampler.c_reinit_from(target_nodes.numpy(), batch_size, use_batch_n)

    #@timing
    def _next_batch(self):
        return self.host_sampler.c_next_batch_py()

    def _cur_batch_opt_nodes(self):
        off_start = self.batch_idx * self.batch_size
        off_end = off_start + self.batch_size
        if off_end > self.total_nodes_n:
            off_end = self.total_nodes_n
        output_nodes = self.target_nodes[off_start:off_end]
        return output_nodes

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.total_nodes_n)
            self.target_nodes = self.target_nodes[idx]
        self.batch_idx = 0
        if self.use_batch_n <= 0:
            self.total_batch = (self.total_nodes_n + self.batch_size -
                            1) // self.batch_size
        else:
            self.total_batch = self.use_batch_n

        self._reinit_from(self.target_nodes, self.batch_size, self.total_batch)
        return self

    def __next__(self):
        if self.batch_idx >= self.total_batch:
            raise StopIteration
        else:
            # all_layers, all_inputs = self._next_batch()
            all_layers = self._next_batch()
            output_nodes = self._cur_batch_opt_nodes()
            self.batch_idx += 1
            return to_dgl_batch(output_nodes, all_layers)


class HostBatchSamplerWithCounter(BatchSampler):

    def __init__(self,
                 graph,
                 target_nodes,
                 fan_outs: list[int],
                 repeated: Optional[bool] = None,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 target_nodes_gideg=None,
                 use_batch_n=-1):
        self.graph = graph
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_nodes = target_nodes
        self.fan_outs = fan_outs
        self.repeated = repeated
        self.total_nodes_n = self.target_nodes.shape[0]
        self.total_batch = 0
        self.use_batch_n = use_batch_n
        self.batch_idx = 0

        self.count_resampling = True
        self.target_nodes_gideg = target_nodes_gideg
        self.host_sampler = HostSampler(graph, fan_outs, repeated,
                                        self.count_resampling)
        self.cur_resample_count = 0

    def __del__(self):
        self.host_sampler = None

    def _cur_batch_opt_nodes(self):
        off_start = self.batch_idx * self.batch_size
        off_end = off_start + self.batch_size
        if off_end > self.total_nodes_n:
            off_end = self.total_nodes_n
        output_nodes = self.target_nodes[off_start:off_end]
        return output_nodes

    def __iter__(self):
        assert torch.is_tensor(self.target_nodes), "only handle tensor input"
        assert torch.is_tensor(
            self.target_nodes_gideg), "only handle tensor input"

        if self.shuffle:
            idx = torch.randperm(self.total_nodes_n)
            self.target_nodes = self.target_nodes[idx]
            self.target_nodes_gideg = self.target_nodes_gideg[idx]
        self.batch_idx = 0

        if self.use_batch_n < 0:
            self.total_batch = (self.total_nodes_n + self.batch_size -
                            1) // self.batch_size
        else:
            self.total_batch = self.use_batch_n

        self.host_sampler.c_reinit_with_gideg_from(
            self.target_nodes.numpy(), self.target_nodes_gideg.numpy(),
            self.batch_size, self.total_batch)

        return self

    def __next__(self):
        if self.batch_idx >= self.total_batch:
            raise StopIteration
        else:
            # when count_resampling is True,
            all_layers, correct = self.host_sampler.c_next_batch_py()
            # print(f"{correct}")
            self.cur_resample_count = correct[0] / correct[1]
            output_nodes = self._cur_batch_opt_nodes()
            self.batch_idx += 1
            return to_dgl_batch(output_nodes, all_layers)

    def get_cur_resampling_cnt(self):
        return self.cur_resample_count


class GpuBatchSampler:

    def __init__(self, fan_outs: list[int]):
        pass

    def _reinit_from(self, target_nodes, batch_size):
        pass

    def _next_batch(self):
        pass
