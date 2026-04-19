#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

import time
import numpy as np

from t10n._C import xborder as c_xborder
from t10n.cl import t10n_all_to_all_1d_int64_nccl as t10n_all_to_all_1d_int64
from t10n.cl import t10n_all_to_all_2d_float32_nccl as t10n_all_to_all_2d_float32
from t10n.util import pprint, timing, pyml_handle

XBorder = c_xborder.XBorder

import torch as th


def xbdbg(any, *args, **kwargs):
    pass
    #pprint(any, f"rank{any.rank:02d}:", *args, **kwargs)


def xb_mask_by_partition(xb: XBorder, ids_th, n_part):
    mask_by_part = th.zeros([n_part, ids_th.shape[0]],
                            dtype=th.int8,
                            device=ids_th.device)
    xb.mask_by_partition(ids_th.numpy(), mask_by_part.numpy())
    return mask_by_part


def xb_group_by_partition_v1(xb: XBorder, ids_th, n_part):
    mask_by_part = xb_mask_by_partition(xb, ids_th, n_part)
    grouped_ids = []
    for i in range(n_part):
        mask = mask_by_part[i]
        nonzero = th.flatten(th.nonzero(mask))
        grouped_ids.append(ids_th[nonzero])
    return grouped_ids, mask_by_part


def xb_group_by_partition(xb: XBorder, ids_th, n_part):
    mask_by_part = th.zeros([n_part, ids_th.shape[0]],
                            dtype=th.int8,
                            device=ids_th.device)
    grouped_ids = xb.group_by_partition(ids_th.numpy(), mask_by_part.numpy())
    grouped_ids_ret = [th.from_numpy(e) for e in grouped_ids]
    return grouped_ids_ret, mask_by_part


# this function is used to fill a batch_feat input for input_nodes
# the caller should make sure input_nodes in [node of this part + halo node]
def xb_ad_hoc_fill_batch_feat(xb: XBorder, device, input_nodes, local_feat_n,
                              rank, nbr_idx, n_part, prefix_sum):
    if n_part == 1:
        batch_inputs = local_feat_n[rank][input_nodes]
        return batch_inputs

    batch_inputs = th.zeros([input_nodes.size(dim=0), local_feat_n[0].shape[1]],
                            dtype=th.float32,
                            device=device)

    mask_by_part = xb_mask_by_partition(xb, input_nodes, n_part)
    mask_r = mask_by_part[rank]
    mask_r_idx = th.flatten(th.nonzero(mask_r))
    batch_inputs[mask_r_idx] = local_feat_n[rank][input_nodes[mask_r_idx] -
                                                  prefix_sum[rank]]

    if n_part == 1:
        return batch_inputs

    mask_n = mask_by_part[nbr_idx]
    mask_n_idx = th.flatten(th.nonzero(mask_n))
    batch_inputs[mask_n_idx] = local_feat_n[nbr_idx][input_nodes[mask_n_idx] -
                                                     prefix_sum[nbr_idx]]

    others_mask = ~(mask_r | mask_n)
    others_idx = th.flatten(th.nonzero(others_mask))
    others_ids = input_nodes[others_idx]

    xb.ad_hoc_fill_batch_feat_v2(batch_inputs, local_feat_n, others_ids,
                                 others_idx)
    return batch_inputs


#
# emb_self is a shard of embedding for all nodes
# emb_self is the shard of current rank, corresponds to id in [low, up)
# target_nodes could be nodes out of [low, up)
#
# rank is the rank of my self
# world_size === number of part
# prefix_sum is how the ids divided into ranks
#
# result is the featched emb on this local machine
# this is purposed for pull batch feature, but can be used for other as well
def xb_gather(xb: XBorder, comm_dev, batch_input, target_nodes, emb_self, rank,
              world_size, prefix_sum):
    # prepare batch input
    input_nodes = target_nodes  # the input nodes at most far to training node

    # only cpu version now
    part_ids_list, mask_by_part = xb_group_by_partition(xb, input_nodes,
                                                        world_size)

    mask_rank = mask_by_part[rank]
    mask_rank_idx = th.flatten(th.nonzero(mask_rank))  # on cpu
    rank_lcoal_ids = (part_ids_list[rank] - prefix_sum[rank])  # on cpu
    batch_input[mask_rank_idx] = (emb_self[rank_lcoal_ids]).to(comm_dev)
    del mask_rank_idx
    del rank_lcoal_ids
    xbdbg(rank, f"part_ids{[e.shape for e in part_ids_list]}")
    gather_other_ids = t10n_all_to_all_1d_int64(part_ids_list,
                                                rank,
                                                world_size,
                                                device=comm_dev)
    xbdbg(rank, f"gather_other_ids{[ e.shape for e in gather_other_ids]}")

    #print(gather_other_ids)
    node_feat_n = []
    for idx, gids in enumerate(gather_other_ids):
        if idx == rank or gids.shape[0] == 0:
            node_feat_n.append(th.empty(0, dtype=th.float32))
        else:
            lids = gids - prefix_sum[rank]
            llids = lids.to(emb_self.device)  # move from gpu to gpu or cpu
            node_feat_n.append(emb_self[llids])

    #pyml_handle.report_memory(rank, "before all_to_all")
    xbdbg(rank, f" node_feat_n{[e.shape for e in node_feat_n]}")
    gather_node_feat_n = t10n_all_to_all_2d_float32(node_feat_n,
                                                    rank,
                                                    world_size,
                                                    device=comm_dev,
                                                    dim_2nd=emb_self.shape[1])
    #pyml_handle.report_memory(rank, "after all_to_all")

    for i in range(world_size):
        if i == rank:
            continue
        mask_i = mask_by_part[i]
        mask_i_idx = th.flatten(th.nonzero(mask_i))
        batch_input[mask_i_idx] = gather_node_feat_n[i]
    # trigger gc
    del mask_by_part
    for i in range(world_size):
        del part_ids_list[0]
        del gather_other_ids[0]
        del gather_node_feat_n[0]
    #pyml_handle.report_memory(rank, "after None")

    return batch_input


#
# emb_self is a shard of embedding for all nodes
# emb_self is the shard of current rank, corresponds to id in [low, up)
# target_nodes could be nodes out of [low, up)
# target_nodes_emb is thr corresponding emb of target_nodes
#
# This function will update each emb_self with target_nodes_emb
#
# The caller should make sure target_nodes do not overlap
# across ranks so that no concurrent write to emb_self
def xb_scatter(xb: XBorder, comm_dev, target_nodes_emb, target_nodes, emb_self,
               rank, world_size, prefix_sum):
    pass


class MiniBatchXborderSampler:
    pass
    # NeighborSampler
