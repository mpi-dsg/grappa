from typing import Union, Optional

from t10n.sampler import BatchSampler
from t10n._C.graph import HostNeighborSampler

from t10n.xborder import XBorder
from t10n.xborder import xb_group_by_partition, xb_gather
from t10n.cl import t10n_all_to_all_1d_int64_nccl as t10n_all_to_all_1d_int64

import torch
import dgl
from dgl.base import NID, EID
from dgl.transforms import to_block
from copy import deepcopy


class HostXBBatchSampler(BatchSampler):

    def __init__(self,
                 graph,
                 target_nodes,
                 fan_outs: list[int],
                 repeated: Optional[bool] = None,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 xb=None,
                 rank: int = -1,
                 num_part: int = 0,
                 device_allocator=None):
        self.graph = graph
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_nodes = target_nodes
        self.fan_outs = fan_outs
        self.repeated = repeated
        self.total_nodes_n = self.target_nodes.shape[0]
        self.total_batch = 0
        self.use_batch_n = -1
        self.batch_idx = 0

        self.host_nbr_sampler = HostNeighborSampler(graph, repeated)
        self.xb = xb
        self.rank = rank
        self.num_part = num_part
        self.da = device_allocator

    def __del__(self):
        self.host_nbr_sampler = None

    def _reinit_from(self, target_nodes, batch_size, use_batch_n):
        assert torch.is_tensor(target_nodes), "only handle tensor input"

    def _cur_batch_opt_nodes(self):
        off_start = self.batch_idx * self.batch_size
        off_end = off_start + self.batch_size
        if off_end > self.total_nodes_n:
            off_end = self.total_nodes_n
        output_nodes = self.target_nodes[off_start:off_end]
        return output_nodes

    def _next_batch(self):
        output_nodes = self._cur_batch_opt_nodes()
        blocks = []

        seed_nodes = output_nodes
        for fanout in reversed(self.fan_outs):
            ########################################
            # get all seed_nodes from every partition
            ########################################
            seed_nodes_list_th, tmp_mask = xb_group_by_partition(
                self.xb, seed_nodes, self.num_part)
            del tmp_mask
            gather_seed_list = t10n_all_to_all_1d_int64(
                seed_nodes_list_th,
                self.rank,
                self.num_part,
                device=self.da.comm_dev())

            #from where they requested:
            #    do dampling
            #    extract feature
            #send back for each of the graph and features
            src_list = []
            dst_list = []
            for idx, seeds_in_this_rank in enumerate(gather_seed_list):
                seeds_in_this_rank = seeds_in_this_rank.to(
                    self.da.part_topo_dev())

                uv, _ = self.host_nbr_sampler.c_next_batch_py(
                    seeds_in_this_rank.numpy(), fanout)

                src_list.append(torch.as_tensor(uv[0]))
                dst_list.append(torch.as_tensor(uv[1]))

            gather_src_list = t10n_all_to_all_1d_int64(
                src_list,
                self.rank,
                world_size=self.num_part,
                device=self.da.comm_dev())
            gather_dst_list = t10n_all_to_all_1d_int64(
                dst_list,
                self.rank,
                world_size=self.num_part,
                device=self.da.comm_dev())

            all_u = torch.concat(tuple(gather_src_list))
            all_v = torch.concat(tuple(gather_dst_list))
            all_u = all_u.to(self.da.part_topo_dev())
            all_v = all_v.to(self.da.part_topo_dev())
            sub_g = dgl.graph((all_u, all_v))

            block = to_block(sub_g, seed_nodes)
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

            # trigger gc
            for i in range(self.num_part):
                del seed_nodes_list_th[0]
                del gather_seed_list[0]
                del gather_src_list[0]
                del gather_dst_list[0]
            del all_u
            del all_v
            del sub_g

        self.batch_idx += 1
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks

    def __next__(self):
        if self.batch_idx >= self.total_batch:
            raise StopIteration
        else:
            return self._next_batch()

    def __iter__(self):
        self.batch_idx = 0
        self.total_batch = (self.total_nodes_n + self.batch_size -
                        1) // self.batch_size

        self._reinit_from(self.target_nodes, self.batch_size, self.total_batch)
        return self
