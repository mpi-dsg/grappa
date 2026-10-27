import numpy as np
import torch as th

from t10n.xborder import XBorder
from t10n.xborder import xb_group_by_partition, xb_gather
from t10n.cl import t10n_all_to_all_1d_int64_nccl as t10n_all_to_all_1d_int64
from t10n.util import pprint

import dgl
import dgl.backend as FF
from dgl.base import NID, EID
from dgl.transforms import to_block
from dgl.dataloading.base import BlockSampler
from dgl.utils import toindex as dgl_toindex



def to_dgl_tensor(th_tensor):
    idxobj = dgl_toindex(th_tensor)
    return idxobj.todgltensor()

class IsolatedSamplerWithPreciseCounter(BlockSampler):

    def __init__(self,
                 fanouts,
                 rank,
                 nbr_idx,
                 num_part,
                 prefix_sum,
                 device,
                 da,
                 isloader,
                 edge_dir='in',
                 prob=None,
                 mask=None,
                 replace=False,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.rank = rank
        self.nbr_idx = nbr_idx
        self.num_part = num_part
        self.prefix_sum = prefix_sum
        self.device = device
        self.da = da
        self.edge_dir = edge_dir
        self.isloader = isloader
        if mask is not None and prob is not None:
            raise ValueError(
                'Mask and probability arguments are mutually exclusive. '
                'Consider multiplying the probability with the mask '
                'to achieve the same goal.')
        self.prob = prob or mask
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []

        bs = float(seed_nodes.shape[0])

        precise_count = 0
        for idx, fanout in enumerate(reversed(self.fanouts)):
            gideg = th.zeros(seed_nodes.shape[0], dtype=th.int64)
            self.isloader.get_batch_gideg(gideg, seed_nodes)
            gideg = FF.zerocopy_to_dgl_ndarray(gideg)
            #print(f"rank:{self.rank}, idx:{idx}, gideg:{gideg}")
            lideg = FF.zerocopy_to_dgl_ndarray(
                th.zeros(seed_nodes.shape[0], dtype=th.int64))
            #print(f"rank:{self.rank}, idx:{idx}, lideg:{lideg}")
            resample_count = FF.zerocopy_to_dgl_ndarray(
                th.zeros(1, dtype=th.int64))
            #print(f"rank:{self.rank}, idx:{idx}, resample_count:{resample_count}")

            frontier = g.sample_neighbors(seed_nodes,
                                          fanout,
                                          edge_dir=self.edge_dir,
                                          prob=self.prob,
                                          replace=False,
                                          output_device=self.output_device,
                                          exclude_edges=exclude_eids,
                                          degs=(gideg, lideg),
                                          resample_count=resample_count)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
            tmp_tensor = FF.zerocopy_from_dgl_ndarray(resample_count)
            precise_count += tmp_tensor.item()
            #print(f"rank:{self.rank}, idx:{idx}, lideg2:{lideg}")
        #print(f"rank:{self.rank}, idx:{idx}, precise_count:{precise_count}")
        # end for
        self.isloader.set_coarse_count(precise_count / bs)

        return seed_nodes, output_nodes, blocks


class IsolatedSamplerOrCoarseCounter(BlockSampler):

    def __init__(self,
                 fanouts,
                 rank,
                 nbr_idx,
                 num_part,
                 prefix_sum,
                 device,
                 da,
                 isloader,
                 edge_dir='in',
                 prob=None,
                 mask=None,
                 replace=False,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.rank = rank
        self.nbr_idx = nbr_idx
        self.num_part = num_part
        self.prefix_sum = prefix_sum
        self.device = device
        self.da = da
        self.edge_dir = edge_dir
        self.isloader = isloader
        if mask is not None and prob is not None:
            raise ValueError(
                'Mask and probability arguments are mutually exclusive. '
                'Consider multiplying the probability with the mask '
                'to achieve the same goal.')
        self.prob = prob or mask
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        #import traceback
        #for line in traceback.format_stack():
        #    pprint(self, ("3.14159", line.strip())
        #pprint(self, ("pi ", g.__class__.__name__)
        output_nodes = seed_nodes
        bs = float(seed_nodes.shape[0])
        blocks = []

        use_coarse_counter = False
        use_isolated_sampler = not use_coarse_counter

        if use_isolated_sampler:
            for idx, fanout in enumerate(reversed(self.fanouts)):
                frontier = g.sample_neighbors(seed_nodes,
                                                fanout,
                                                edge_dir=self.edge_dir,
                                                prob=self.prob,
                                                replace=self.replace,
                                                output_device=self.output_device,
                                                exclude_edges=exclude_eids)
                eid = frontier.edata[EID]
                block = to_block(frontier, seed_nodes)
                block.edata[EID] = eid
                seed_nodes = block.srcdata[NID]
                blocks.insert(0, block)
            return seed_nodes, output_nodes, blocks

        else:
            factor = [1] * (len(self.fanouts))
            r_f = [e for e in reversed(self.fanouts)]
            for i in range(len(r_f) - 1, 0, -1):
                factor[i - 1] = r_f[i] * factor[i]

            coarse_count = 0

            for idx, fanout in enumerate(reversed(self.fanouts)):
                frontier = g.sample_neighbors(seed_nodes,
                                              fanout,
                                              edge_dir=self.edge_dir,
                                              prob=self.prob,
                                              replace=self.replace,
                                              output_device=self.output_device,
                                              exclude_edges=exclude_eids)

                eid = frontier.edata[EID]
                block = to_block(frontier, seed_nodes)
                block.edata[EID] = eid
                seed_nodes = block.srcdata[NID]
                blocks.insert(0, block)

                rank_mask = (seed_nodes >= self.prefix_sum[self.rank]) & (
                    seed_nodes < self.prefix_sum[self.rank])
                nbr_mask = (seed_nodes >= self.prefix_sum[self.nbr_idx]) & (
                    seed_nodes < self.prefix_sum[self.nbr_idx])
                count = th.count_nonzero(rank_mask | nbr_mask)
                out_count = seed_nodes.shape[0] - count
                coarse_count += out_count * factor[idx]
            # end for
            self.isloader.set_coarse_count(coarse_count / bs)

            return seed_nodes, output_nodes, blocks


class XBorderSampler(BlockSampler):

    def __init__(self,
                 fanouts,
                 rank,
                 num_part,
                 prefix_sum,
                 xborder: XBorder,
                 device,
                 da,
                 edge_dir='in',
                 prob=None,
                 mask=None,
                 replace=False,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.rank = rank
        self.prefix_sum = prefix_sum
        self.num_part = num_part
        self.xb = xborder
        self.device = device
        self.da = da
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                'Mask and probability arguments are mutually exclusive. '
                'Consider multiplying the probability with the mask '
                'to achieve the same goal.')
        self.prob = prob or mask
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        #import traceback
        #for line in traceback.format_stack():
        #    pp(self, ("3.14159", line.strip())
        #pp(self, ("pi ", g.__class__.__name__)
        output_nodes = seed_nodes
        blocks = []
        layer_counter = 0
        for fanout in reversed(self.fanouts):
            layer_counter += 1
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
                frontier = g.sample_neighbors(seeds_in_this_rank,
                                              fanout,
                                              edge_dir=self.edge_dir,
                                              prob=self.prob,
                                              replace=self.replace,
                                              output_device=self.output_device,
                                              exclude_edges=exclude_eids)
                u, v = frontier.edges()
                src_list.append(u)
                dst_list.append(v)

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

            all_u = th.concat(tuple(gather_src_list))
            all_v = th.concat(tuple(gather_dst_list))
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

        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks

class PinSAGESampler(BlockSampler):

    def __init__(self,
                 g,
                 num_layer,
                 random_walk_length=3,
                 random_walk_restart_prob=0.5,
                 num_random_walk=4,
                 num_neighbor=5,
                 edge_dir='in',
                 prob=None,
                 mask=None,
                 replace=False,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):

        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)

        if isinstance(g, dgl.distributed.dist_graph.DistGraph):
            self.g = g.local_partition
        else:
            self.g = g
        self.num_layer = num_layer
        self.sampler = dgl.sampling.RandomWalkNeighborSampler(
            self.g, random_walk_length, random_walk_restart_prob,
            num_random_walk, num_neighbor)

    def sample_blocks(self, _, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes

        blocks = []
        for _ in range(self.num_layer):
            frontier = self.sampler(seed_nodes)
            block = dgl.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes

        return input_nodes, output_nodes, blocks
