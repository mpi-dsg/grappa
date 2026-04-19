#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

import faulthandler

# Enable fault handler to print tracebacks on crashes
faulthandler.enable()

from t10n._C import c_dist
from t10n.util import timing, pprint
from t10n.cl import t10n_gather_gloo as t10n_gather
from .meta import name_to_meta, t10n_chunk_path
from .io_dist import get_dataset_io, DatasetIO, SplitConfig
from .io_dist import per_rank_stride, per_rank_local_num, per_rank_padding

import os
import pathlib

import torch as th


@timing
def random_assign_pid_for_nid(world_size, rank, nid2pid, num_part):
    assert th.is_tensor(nid2pid)
    assert nid2pid.dtype == th.int32
    assert nid2pid.shape[0] % world_size == 0

    local_num = nid2pid.shape[0] // world_size
    local_slice = th.zeros(local_num, dtype=nid2pid.dtype)
    c_dist.random_assign_pid(rank, local_slice.numpy(), num_part)

    # th.distributed.all_gather_into_tensor(nid2pid, local_slice) # gloo not work
    th.distributed.all_gather(list(nid2pid.chunk(num_part)),
                              local_slice)  # works for gloo


@timing
def relabel_nid(world_size, rank, nid2lid, nid2pid, num_part):
    assert th.is_tensor(nid2pid)
    assert nid2pid.dtype == th.int32
    assert th.is_tensor(nid2lid)
    assert nid2lid.dtype == th.int64
    nodes_per_part = [0] * num_part
    for pid in range(num_part):
        mask = nid2pid == pid
        idx = th.flatten(th.nonzero(mask))
        nid2lid[idx] = th.arange(idx.shape[0])
        nodes_per_part[pid] = idx.shape[0]
    return nodes_per_part


def assert_mask_true(input_mask):
    pass
    #nozero = th.flatten(th.nonzero(input_mask))
    #assert input_mask.shape[0] == nozero.shape[0]

@timing
def split_ideg(sc: SplitConfig, ds_io: DatasetIO):
    assert sc.nid2pid.dtype == th.int32

    ds_name = sc.ds_meta.name
    if ds_name == 'igb260m':
        # igb is not augmented
        total_n_e = sc.ds_meta.origin_n_e
    elif ds_name == 'ogbpa':
        total_n_e = sc.ds_meta.n_e
    elif ds_name == 'ogbpr':
        total_n_e = sc.ds_meta.n_e
    else:
        assert False, "not tested"

    # how many parts each rank handles
    N_PART_PER_RANK = sc.num_part // sc.world_size

    stride = per_rank_stride(sc.world_size, sc.rank, total_n_e)
    rank_start = sc.rank * stride
    rank_end = rank_start + per_rank_local_num(sc.world_size, sc.rank,
                                               total_n_e)
    rank_u, rank_v = ds_io.read_edge(rank_start, rank_end)
    node_ids, in_deg = th.unique(rank_v, return_counts=True)

    dst_rank = 0
    all_node_ids = t10n_gather(node_ids, sc.rank, sc.world_size, dst_rank)
    all_in_deg   = t10n_gather(in_deg,   sc.rank, sc.world_size, dst_rank)

    if sc.rank == dst_rank:
        final_indeg  = th.zeros(sc.nid2pid.shape[0], dtype=th.int64)
        import time
        print("calculating degree @", time.strftime("%H:%M:%S"))
        for nid, ideg in zip(all_node_ids, all_in_deg):
            final_indeg[nid] += ideg
        print("calculating degree finish @", time.strftime("%H:%M:%S"))

        ideg_list = [ th.zeros(l, dtype=th.int64) for l in sc.nodes_per_part ]
        for i in range(sc.num_part):
            print( f"dump for partition {i} @", time.strftime("%H:%M:%S") )
            part_i_mask = sc.nid2pid == i
            part_i_nid = th.flatten(th.nonzero(part_i_mask))
            lid = sc.nid2lid[part_i_nid]
            lideg = final_indeg[part_i_nid]
            ideg_list[i][lid] = lideg
            ds_io.write('ideg', i, ideg_list[i])

    th.distributed.barrier()

@timing
def split_edges(sc: SplitConfig, ds_io: DatasetIO):
    assert sc.nid2pid.dtype == th.int32

    ds_name = sc.ds_meta.name
    if ds_name == 'igb260m':
        # igb is not augmented
        total_n_e = sc.ds_meta.origin_n_e
    elif ds_name == 'ogbpa':
        total_n_e = sc.ds_meta.n_e
    elif ds_name == 'ogbpr':
        total_n_e = sc.ds_meta.n_e
    else:
        assert False, "not tested"

    # how many parts each rank handles
    N_PART_PER_RANK = sc.num_part // sc.world_size

    stride = per_rank_stride(sc.world_size, sc.rank, total_n_e)
    rank_start = sc.rank * stride
    rank_end = rank_start + per_rank_local_num(sc.world_size, sc.rank,
                                               total_n_e)
    rank_u, rank_v = ds_io.read_edge(rank_start, rank_end)

    #p_u = [None] * sc.num_part
    #p_v = [None] * sc.num_part
    #b_u = [[None] * sc.num_part for i in range(sc.num_part)]
    #b_v = [[None] * sc.num_part for i in range(sc.num_part)]

    rank_u_pid = sc.nid2pid[rank_u]
    rank_v_pid = sc.nid2pid[rank_v]

    mask_u_eq_v = rank_u_pid == rank_v_pid
    mask_u_eq_v_n = ~mask_u_eq_v

    for pidx in range(sc.num_part):
        dst_rank = pidx // N_PART_PER_RANK

        mask_this_part = rank_u_pid == pidx
        ############
        # p edges
        ############
        # u, v are both in part pidx
        local_mask = mask_this_part & mask_u_eq_v
        idx_this_part = th.flatten(th.nonzero(local_mask))

        pp_u = rank_u[idx_this_part]
        pp_v = rank_v[idx_this_part]
        assert_mask_true(sc.nid2pid[pp_u] == pidx)
        assert_mask_true(sc.nid2pid[pp_u] == pidx)

        # gather data
        # p_u, p_v
        final_p_u = t10n_gather(rank_u[idx_this_part], sc.rank, sc.world_size,
                                dst_rank)
        final_p_v = t10n_gather(rank_v[idx_this_part], sc.rank, sc.world_size,
                                dst_rank)
        if dst_rank == sc.rank:
            ##################
            # debug assert
            # should remove later
            ##################
            assert_mask_true(sc.nid2pid[final_p_u] == pidx)
            assert_mask_true(sc.nid2pid[final_p_v] == pidx)

            final_p_u = sc.nid2lid[final_p_u]
            final_p_v = sc.nid2lid[final_p_v]
            ds_io.write_primary(pidx, final_p_u, final_p_v)

        ############
        # bridges edges
        ############
        # u in pidx, v is not
        bridge_mask_bu = mask_this_part & mask_u_eq_v_n
        for pjdx in range(sc.num_part):
            mask_pjdx = rank_v_pid == pjdx
            bridge_mask_bv = bridge_mask_bu & mask_pjdx
            bridge_idx = th.flatten(th.nonzero(bridge_mask_bv))
            if pidx == pjdx:
                assert bridge_idx.shape[0] == 0
                continue
            #b_u[pidx][pjdx] = rank_u[bridge_idx]
            #b_v[pidx][pjdx] = rank_v[bridge_idx]

            final_b_u = t10n_gather(rank_u[bridge_idx], sc.rank, sc.world_size,
                                    dst_rank)
            final_b_v = t10n_gather(rank_v[bridge_idx], sc.rank, sc.world_size,
                                    dst_rank)
            if dst_rank == sc.rank:
                ##################
                # debug assert
                # should remove later
                ##################
                assert_mask_true(sc.nid2pid[final_b_u] == pidx)
                assert_mask_true(sc.nid2pid[final_b_v] == pjdx)

                final_b_u = sc.nid2lid[final_b_u]
                final_b_v = sc.nid2lid[final_b_v]
                ds_io.write_bridge(pidx, final_b_u, pjdx, final_b_v)


def split_node_data(sc: SplitConfig, ds_io: DatasetIO, key: str):
    assert sc.nid2pid.dtype == th.int32
    assert sc.num_part % sc.world_size == 0
    # how many parts each rank handles
    N_PART_PER_RANK = sc.num_part // sc.world_size
    # the start and end pid this rank handles
    pid_beg = N_PART_PER_RANK * sc.rank
    pid_end = N_PART_PER_RANK * (sc.rank + 1)
    # get original node id range this rank loads
    oid_start = sum(sc.nodes_per_part[:pid_beg])
    oid_end = sum(sc.nodes_per_part[:pid_end])

    # get original nid to pid this rank uses
    assert (oid_end - oid_start) == sc.nodes_per_part[sc.rank]
    local_nid2pid = sc.nid2pid[oid_start:oid_end]
    # get node of original nid [start, end) this rank uses
    local_slice = ds_io.read(key, oid_start, oid_end)

    # alloc space for parts handled by this rank
    # gathered from net and dumped to file
    parts_per_rank = [None] * sc.num_part
    data_type = None
    for i in range(sc.num_part):
        prank = i // N_PART_PER_RANK
        if prank == sc.rank:
            parts_per_rank[i] = ds_io.zero(key, sc.nodes_per_part[i])
            data_type = parts_per_rank[i].dtype

    for pidx in range(sc.num_part):
        # find original nid that was assigned to partition_pidx
        mask = local_nid2pid == pidx
        idx = th.flatten(th.nonzero(mask))
        # original nids
        idx_oid = idx + oid_start

        # the rank that will gather node data
        dst_rank = pidx // N_PART_PER_RANK

        # local_slice: node data of range [oid_start, oid_end)
        # local_slice[idx]: data of partition pidx
        slice_all_rank = t10n_gather(local_slice[idx], sc.rank, sc.world_size,
                                     dst_rank)
        # the coresponding original nid for local_slice[idx]
        o_nid_all_rank = t10n_gather(idx_oid, sc.rank, sc.world_size, dst_rank)

        if dst_rank == sc.rank:
            ##################
            # debug assert
            # should remove later
            ##################
            assert_mask_true(sc.nid2pid[o_nid_all_rank] == pidx)
            assert (slice_all_rank.shape[0] == sc.nodes_per_part[pidx])
            assert (o_nid_all_rank.shape[0] == sc.nodes_per_part[pidx])

            the_lid = sc.nid2lid[o_nid_all_rank]
            parts_per_rank[pidx][the_lid] = slice_all_rank

    # end for pidx in range(sc.num_part):
    for i in range(sc.num_part):
        prank = i // N_PART_PER_RANK
        if prank == sc.rank:
            ds_io.write(key, i, parts_per_rank[i])


@timing
def split_1d(sc: SplitConfig, ds_io: DatasetIO, key: str):
    split_node_data(sc, ds_io, key)


@timing
def split_2d(sc: SplitConfig, ds_io: DatasetIO, key: str):
    split_node_data(sc, ds_io, key)


@timing
def build_chunks_dist(world_size, rank, dest_path, graph_name, src_path,
                      part_method, num_part):
    assert num_part % world_size == 0

    ds_meta = name_to_meta(graph_name)
    # global nid2pid -- original id to partition id
    padded_size = per_rank_padding(world_size, rank, ds_meta.n_n)
    pprint(
        rank,
        f"n_n, padded_size, world_size = {ds_meta.n_n}, {padded_size}, {world_size}"
    )
    nid2pid = th.zeros(padded_size, dtype=th.int32)
    random_assign_pid_for_nid(world_size, rank, nid2pid, num_part)

    nid2lid = th.zeros(ds_meta.n_n, dtype=th.int64)
    nodes_per_part = relabel_nid(world_size, rank, nid2lid,
                                 nid2pid[:ds_meta.n_n], num_part)

    sc = SplitConfig(world_size, rank, nid2pid[:ds_meta.n_n], nid2lid, num_part, ds_meta,
                     nodes_per_part)

    output_dir = os.path.join(
        dest_path, t10n_chunk_path(graph_name, part_method, num_part))
    if rank == 0:
        if os.path.exists(output_dir):
            assert False, f"please delete {output_dir} manually"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    th.distributed.barrier()

    io = get_dataset_io(src_path, output_dir, graph_name)

    split_edges(sc, io)
    split_ideg(sc, io)

    split_1d(sc, io, key="node_label_file")
    split_1d(sc, io, key="train_mask_file_bin")
    split_1d(sc, io, key="val_mask_file_bin")
    split_1d(sc, io, key="test_mask_file_bin")

    split_2d(sc, io, key="node_feats_file_bin")
