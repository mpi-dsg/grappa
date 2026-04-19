#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

import argparse
import socket
import os, time, datetime

import numpy as np

from t10n.dataset.meta import name_to_meta, t10n_chunk_path

from t10n._C.graph import HostGraph
from t10n.sampler import HostBatchSampler, HostBatchSamplerWithCounter
from t10n.xbsampler import HostXBBatchSampler

from t10n.device_allocator import DeviceAllocGPUServing, T10nDeviceAllocator
from t10n.xborder import XBorder
from t10n.xborder import xb_group_by_partition, xb_gather
from t10n.util import pprint, timing, set_py_affinity

import torch as th
import dgl
import dgl.backend as FF


def pp(self, *args, **kwargs):
    pprint(0, f"rank{self.rank:02d}:", *args, **kwargs)


class IsolatedDataloaderGPU:

    def __init__(self, device, args) -> None:
        tic = time.time()
        meta_o = name_to_meta(args.graph_name)

        self.device = device
        self.da = DeviceAllocGPUServing(device, args.backend)
        self.in_feats = meta_o.n_dim
        self.n_classes = meta_o.n_label

        self.use_dgl_dsg_sampler = not args.close_dd

        self.num_part = args.n_part
        self.feat_n = [None] * self.num_part
        self.ideg_n = [None] * self.num_part

        self.part_src = None
        self.part_dst = None

        # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.
        self.data_path = os.path.join(
            args.data_path,
            t10n_chunk_path(args.graph_name, args.part_method, args.n_part))

        rank = th.distributed.get_rank()
        world = th.distributed.get_world_size()
        self.rank = rank

        self.world = world
        self.nbr_idx = self.rank

        self.max_coarse_count = args.batch_size * float(args.fan_out.split(",")[0])
        ta = T10nDeviceAllocator(args.num_gpus)
        local_gid = ta.get_local_gpu_id(self.world, self.rank)
        self.node_id = ta.get_numa_affinity(local_gid)
        set_py_affinity(self.node_id)

        p0_u = f"{self.data_path}/p{rank}.u.bin"
        p0_v = f"{self.data_path}/p{rank}.v.bin"

        feat_0 = f"{self.data_path}/p{rank}.nfeat.bin"
        label_0 = f"{self.data_path}/p{rank}.label.bin"
        ideg_0 = f"{self.data_path}/p{rank}.ideg.bin"

        train_mask_0 = f"{self.data_path}/p{rank}.train.bin"
        valid_mask_0 = f"{self.data_path}/p{rank}.valid.bin"
        test_mask_0 = f"{self.data_path}/p{rank}.test.bin"

        pp(self, '-' * 50)
        p0_u = np.fromfile(p0_u, dtype=np.int64)
        pp(self, f"p0_u, shape={p0_u.shape}, {p0_u[0]}")

        p0_v = np.fromfile(p0_v, dtype=np.int64)
        pp(self, f"p0_v, shape={p0_v.shape}, {p0_v[0]}")

        feat_0 = np.fromfile(feat_0, dtype=np.float32)
        feat_0 = np.reshape(feat_0, (-1, self.in_feats))
        pp(self, f"feat_0, shape={feat_0.shape}, {feat_0[0][:1]}")

        label_0 = np.fromfile(label_0, dtype=np.float32)
        pp(self, f"label_0, shape={label_0.shape}, {label_0[0]}")

        if os.path.isfile(ideg_0):
            ideg_0 = np.fromfile(ideg_0, dtype=np.int64)
            self.ideg_0 = FF.zerocopy_from_numpy(ideg_0)
            pp(self, f"ideg_0, shape={ideg_0.shape}, {ideg_0[0]}")
        else:
            pp(self, f"ideg_0, not available")
            self.ideg_0 = th.empty((0,), dtype=th.int64)

        train_mask_0 = np.fromfile(train_mask_0, dtype=np.int8)
        valid_mask_0 = np.fromfile(valid_mask_0, dtype=np.int8)
        test_mask_0 = np.fromfile(test_mask_0, dtype=np.int8)

        assert self.da.part_feat_dev() == th.device("cpu"), "only tested cpu"
        train_mask_0 = FF.zerocopy_from_numpy(train_mask_0)
        valid_mask_0 = FF.zerocopy_from_numpy(valid_mask_0)
        test_mask_0 = FF.zerocopy_from_numpy(test_mask_0)

        self.p0_u = FF.zerocopy_from_numpy(p0_u)
        self.p0_v = FF.zerocopy_from_numpy(p0_v)
        self.label_0 = FF.zerocopy_from_numpy(label_0)
        self.feat_0 = FF.zerocopy_from_numpy(feat_0)

        train_nid0 = th.flatten(th.nonzero(train_mask_0))
        #train_nid1 = th.flatten(th.nonzero(train_mask_1)) + label_0.size(dim=0)
        self.train_nid = train_nid0  # th.concat((train_nid0, train_nid1))
        self.train_mask = train_mask_0

        valid_nid0 = th.flatten(th.nonzero(valid_mask_0))
        #valid_nid1 = th.flatten(th.nonzero(valid_mask_1)) + label_0.size(dim=0)
        self.valid_nid = valid_nid0  #th.concat((valid_nid0, valid_nid1))
        self.valid_mask = valid_mask_0

        test_nid0 = th.flatten(th.nonzero(test_mask_0))
        # test_nid1 = th.flatten(th.nonzero(test_mask_1)) + label_0.size(dim=0)
        self.test_nid = test_nid0  # th.concat((test_nid0, test_nid1))
        self.test_mask = test_mask_0

        self.node_label = self.label_0

        pp(self, f"train_mask_0, shape={train_mask_0.shape}, {train_mask_0[0]}")

        pp(self, f"rank:{self.rank:02d} INIT_LOAD: {time.time() - tic:.2f} s")
        batch_num = (train_nid0.size()[0] + int(args.batch_size) - 1) // int(
            args.batch_size)

        local_batch = th.tensor([batch_num, batch_num],
                                device=self.da.comm_dev())
        th.distributed.all_reduce(local_batch, op=th.distributed.ReduceOp.MIN)

        max_batch = th.tensor([batch_num, batch_num], device=self.da.comm_dev())
        th.distributed.all_reduce(max_batch, op=th.distributed.ReduceOp.MAX)

        self.local_batch = local_batch[0].item()
        pp(self, f"rank:{self.rank:02d} NUM_BATCH: {self.local_batch}")
        pp(
            self,
            f"rank:{self.rank:02d} N_TRAIN  : {train_nid0.shape[0]} ({batch_num}/{max_batch[0].item()})"
        )
        pp(self, f"rank:{self.rank:02d} N_VALID  : {valid_nid0.shape[0]}")
        pp(self, f"rank:{self.rank:02d} N_TEST   : {test_nid0.shape[0]}")
        pp(
            self,
            f"rank:{self.rank:02d} N_ALL    : {train_nid0.shape[0] + valid_nid0.shape[0] + test_nid0.shape[0]}"
        )

        pp(self, '-' * 50)

        self.explore_mode = 1

        ############################################
        # a new mode, in this mode,
        # p0 and sum(b0i) are collected as p0',
        # p0' and p1' are the new partition
        ############################################
        # get number of nodes in each primary chunk
        # this is used to calcula
        self.explore_mode = 2
        self.part_num_node = []
        for pid in range(self.num_part):
            # used to count node number
            train_mask_f = f"{self.data_path}/p{pid}.train.bin"
            blob = np.memmap(train_mask_f, dtype=np.int8, mode='r')
            self.part_num_node.append(blob.size)

        self.prefix_sum = [0] * (self.num_part + 1)
        for i in range(1, self.num_part + 1):
            self.prefix_sum[i] = (self.part_num_node[i - 1] +
                                  self.prefix_sum[i - 1])
        self.xb = XBorder(self.rank, self.num_part, self.prefix_sum)

        self.train_nid = self.train_nid + self.prefix_sum[self.rank]
        self.valid_nid = self.valid_nid + self.prefix_sum[self.rank]
        self.test_nid = self.test_nid + self.prefix_sum[self.rank]

        self.batch_coarse_count = 1.0  # coarse count of current batch

        self.all_pu = []
        self.all_pv = []
        self.all_ideg = []
        self.all_nfeat = []

        self.all_bu = []
        self.all_bv = []

        for i in range(self.num_part):
            fpu = f"{self.data_path}/p{i}.u.bin"
            fpv = f"{self.data_path}/p{i}.v.bin"
            fideg = f"{self.data_path}/p{i}.ideg.bin"
            fnfeat = f"{self.data_path}/p{i}.nfeat.bin"

            if i == rank:
                self.all_pu.append(self.p0_u)
                self.all_pv.append(self.p0_v)
                self.all_ideg.append(self.ideg_0)
                self.all_nfeat.append(self.feat_0)
            else:
                punp = np.fromfile(fpu, dtype=np.int64)
                pvnp = np.fromfile(fpv, dtype=np.int64)
                idegnp = np.fromfile(fideg, dtype=np.int64)
                fnfeatnp = np.fromfile(fnfeat, dtype=np.int64)

                punp = FF.zerocopy_from_numpy(punp) + self.prefix_sum[i]
                pvnp = FF.zerocopy_from_numpy(pvnp) + self.prefix_sum[i]
                idegnp = FF.zerocopy_from_numpy(idegnp)
                fnfeatnp = FF.zerocopy_from_numpy(fnfeatnp)

                self.all_pu.append(punp)
                self.all_pv.append(pvnp)
                self.all_ideg.append(idegnp)
                self.all_nfeat.append(fnfeatnp)

        for p_src in range(self.num_part):
            bu_0n = []
            bv_0n = []
            for p_dst in range(self.num_part):
                bufile = f"{self.data_path}/b{p_src}_{p_dst}.u.bin"
                bvfile = f"{self.data_path}/b{p_src}_{p_dst}.u.bin"

                if p_dst == p_src:
                    bu_0n.append(None)
                    bv_0n.append(None)
                else:
                    bunp = np.fromfile(bufile, dtype=np.int64)
                    bvnp = np.fromfile(bvfile, dtype=np.int64)

                    bunp = FF.zerocopy_from_numpy(bunp) + self.prefix_sum[p_src]
                    bvnp = FF.zerocopy_from_numpy(bvnp) + self.prefix_sum[p_dst]

                    bu_0n.append(bunp)
                    bv_0n.append(bvnp)

            self.all_bu.append(bu_0n)
            self.all_bv.append(bv_0n)

    def load_nbr_chunk_v2(self, nbr_idx):
        tic = time.time()
        self.xb.ad_hoc_clear()
        self.xb.set_nbr_idx(nbr_idx)

        # load second primary
        p1_u = f"{self.data_path}/p{nbr_idx}.u.bin"
        p1_v = f"{self.data_path}/p{nbr_idx}.v.bin"

        p1_u = np.fromfile(p1_u, dtype=np.int64)
        p1_v = np.fromfile(p1_v, dtype=np.int64)
        # update to the global node id
        p1_u = FF.zerocopy_from_numpy(p1_u) + self.prefix_sum[nbr_idx]
        p1_v = FF.zerocopy_from_numpy(p1_v) + self.prefix_sum[nbr_idx]

        feat_1 = f"{self.data_path}/p{nbr_idx}.nfeat.bin"
        label_1 = f"{self.data_path}/p{nbr_idx}.label.bin"
        ideg_1 = f"{self.data_path}/p{nbr_idx}.ideg.bin"

        feat_1 = np.fromfile(feat_1, dtype=np.float32)
        feat_1 = np.reshape(feat_1, (-1, self.in_feats))
        label_1 = np.fromfile(label_1, dtype=np.float32)

        if os.path.isfile(ideg_1):
            ideg_1 = np.fromfile(ideg_1, dtype=np.int64)
            self.ideg_1 = FF.zerocopy_from_numpy(ideg_1)
        else:
            self.ideg_1 = th.empty((0,), dtype=th.int64)

        self.feat_1 = FF.zerocopy_from_numpy(feat_1)
        self.label_1 = FF.zerocopy_from_numpy(label_1)

        self.feat_n[self.rank] = self.feat_0
        self.feat_n[nbr_idx] = self.feat_1

        self.ideg_n[self.rank] = self.ideg_0
        self.ideg_n[nbr_idx] = self.ideg_1

        #  It does the following things
        #  0) work in a new global node id space: new_id in pi + sum_0_(i-1){num(pj)}
        #  1) it take bi_j.u or bi_j.v (local id)
        #  2) create a map from bi_j.u -> reid or -1 into bi_j.u.reid.bin
        #  3) coupled with 2), create a feature file from pj.feat.bin used by bi_j.u and rerrange into bij.u.reid.feat.bin,
        #  the workflow is batch_ids -> gids in range p -> look up id from bij_u.reid.bin -> look up feature from bij.u.reid.feat.bin

        # idx-relabel by nbr_idx

        # b01_u = f"{data_path}/b{nbr_idx}_{self.rank}.u.bin"
        # b01_v = f"{data_path}/b{nbr_idx}_{self.rank}.v.bin"
        tensor_src = None
        tensor_dest = None
        if self.num_part > 1:
            tensor_src = th.concat((p1_u,))
            tensor_dest = th.concat((p1_v,))
        tensor_ids = None
        for bidx in range(self.num_part):
            for prim_idx in [self.rank, nbr_idx]:
                if bidx == prim_idx:
                    continue
                #b01_u = f"{self.data_path}/b{prim_idx}_{bidx}.u.bin"
                #b01_v = f"{self.data_path}/b{prim_idx}_{bidx}.v.bin"

                # sampling: train_node <<-- random neighbors
                # so gather bridging edges start from neighbor to train node
                b01_u = f"{self.data_path}/b{bidx}_{prim_idx}.u.bin"
                b01_v = f"{self.data_path}/b{bidx}_{prim_idx}.v.bin"
                if not os.path.isfile(b01_u):
                    continue

                b01_u = np.fromfile(b01_u, dtype=np.int64)
                b01_v = np.fromfile(b01_v, dtype=np.int64)
                #print(f"{self.rank} bidx={bidx}, prim_idx={prim_idx}, b01_u.shape = {b01_u.shape}")

                b01_u = FF.zerocopy_from_numpy(b01_u) + self.prefix_sum[
                    bidx]  # be careful with the index bidx or prim_idx
                b01_v = FF.zerocopy_from_numpy(
                    b01_v) + self.prefix_sum[prim_idx]

                tensor_src = th.concat((tensor_src, b01_u))
                tensor_dest = th.concat((tensor_dest, b01_v))
                if tensor_ids is None:
                    tensor_ids = b01_u  # add in edge only
                else:
                    tensor_ids = th.concat(
                        (tensor_ids, b01_u))  # add in edge only


#        for bidx in range(self.num_part):
#            for prim_idx in [self.rank, nbr_idx]:
#                if bidx == self.rank or bidx == nbr_idx:  # added already in 'in'
#                    continue
#                b01_u = f"{self.data_path}/b{prim_idx}_{bidx}.u.bin"
#                b01_v = f"{self.data_path}/b{prim_idx}_{bidx}.v.bin"
#
#                b01_u = np.fromfile(b01_u, dtype=np.int64)
#                b01_v = np.fromfile(b01_v, dtype=np.int64)
#
#                b01_u = FF.zerocopy_from_numpy(b01_u) + self.prefix_sum[
#                    prim_idx]  # be careful with the index bidx or prim_idx
#                b01_v = FF.zerocopy_from_numpy(b01_v) + self.prefix_sum[bidx]
#
#                tensor_src = th.concat((tensor_src, b01_u))
#                tensor_dest = th.concat((tensor_dest, b01_v))
#                if tensor_ids is None:
#                    tensor_ids = b01_v  # add in edge only
#                else:
#                    tensor_ids = th.concat(
#                        (tensor_ids, b01_v))  # add in edge only

# build cache
        if tensor_ids is not None:
            uniq_ids = th.unique(tensor_ids)  # mapping 0... to global nid
            pp(self, f"rank:{self.rank:02d}  NUM_HALO: {uniq_ids.shape[0]}")
            left = th.arange(self.prefix_sum[self.rank],
                             self.prefix_sum[self.rank + 1])
            right = th.arange(self.prefix_sum[self.nbr_idx],
                              self.prefix_sum[self.nbr_idx + 1])
            total_n = th.unique(th.concat((uniq_ids, left, right))).shape[0]
            left = None
            right = None
            pp(self, f"rank:{self.rank:02d}  NODE_SUM: {total_n}")
            pp(
                self,
                f"rank:{self.rank:02d}  NODE_FEA: {float(total_n*self.in_feats*4)/1024.0/1024.0/1024.0:.4f} GB"
            )

            ids_list, _ = xb_group_by_partition(self.xb, uniq_ids,
                                                self.num_part)
            for bidx in range(self.num_part):
                if bidx == self.rank or bidx == self.nbr_idx:
                    continue

                # sorting is required for faster copy
                bidx_ids, _ = th.sort(ids_list[bidx])

                # deal with feature vector
                feat = f"{self.data_path}/p{bidx}.nfeat.bin"
                dumm = np.memmap(feat, dtype=np.float32, mode='r')
                dim0 = dumm.size // self.in_feats
                assert dim0 == self.part_num_node[bidx]
                dim0 = self.part_num_node[bidx]
                feat = np.memmap(feat,
                                 dtype=np.float32,
                                 mode='r',
                                 shape=(dim0, self.in_feats))

                feat = feat[bidx_ids - self.prefix_sum[bidx]]

                self.feat_n[bidx] = FF.zerocopy_from_numpy(feat)

                # deal with in degree
                ideg = f"{self.data_path}/p{bidx}.ideg.bin"
                if os.path.isfile(ideg):
                    dumm = np.memmap(ideg, dtype=np.int64, mode='r')
                    ideg = dumm[bidx_ids - self.prefix_sum[bidx]]
                    self.ideg_n[bidx] = FF.zerocopy_from_numpy(ideg)
                else:
                    self.ideg_n[bidx] = th.empty((0,), dtype=th.int64)
                self.xb.ad_hoc_build_id_mapping(bidx, bidx_ids.numpy())

        p0_u = self.p0_u + self.prefix_sum[self.rank]
        p0_v = self.p0_v + self.prefix_sum[self.rank]
        if self.num_part > 1:
            self.part_src = th.concat((tensor_src, p0_u))
            self.part_dst = th.concat((tensor_dest, p0_v))
        else:
            self.part_src = p0_u
            self.part_dst = p0_v

        tensor_src = None
        tensor_dest = None

        pp(self, '-' * 50)
        pp(self, f"rank:{self.rank:02d} RELOADING: {time.time() - tic:.2f} s")
        pp(self, '-' * 50)
        #th.distributed.barrier(timeout=datetime.timedelta(seconds=7200))
        th.distributed.barrier()
        # barrier timeout: store_util.barrier()                 : torch/distributed/elastic/agent/server/api.py#L959
        # torch :class LocalElasticAgent(SimpleElasticAgent)    : torch/distributed/elastic/agent/server/local_elastic_agent.py
        # torch :agent = LocalElasticAgent(                     : torch/distributed/elastic/agent/server/api.py
        pp(self, f"rank:{self.rank:02d} BARRIER  : {time.time() - tic:.2f} s")

    def load_nbr_chunk(self, nbr_idx):
        tic = time.time()
        p1_u = f"{self.data_path}/p{nbr_idx}.u.bin"
        p1_v = f"{self.data_path}/p{nbr_idx}.v.bin"

        # sampling: train_node <<-- random neighbors
        # so gather bridging edges start from neighbor to train node
        b01_u = f"{self.data_path}/b{nbr_idx}_{self.rank}.u.bin"
        b01_v = f"{self.data_path}/b{nbr_idx}_{self.rank}.v.bin"

        feat_1 = f"{self.data_path}/p{nbr_idx}.nfeat.bin"
        label_1 = f"{self.data_path}/p{nbr_idx}.label.bin"

        p1_u = np.fromfile(p1_u, dtype=np.int64)
        p1_v = np.fromfile(p1_v, dtype=np.int64)

        if os.path.isfile(b01_u):
            b01_u = np.fromfile(b01_u, dtype=np.int64)
            b01_v = np.fromfile(b01_v, dtype=np.int64)
            self.b01_u = FF.zerocopy_from_numpy(b01_u) + self.label_0.size(
                dim=0)
            self.b01_v = FF.zerocopy_from_numpy(b01_v)
        else:
            self.b01_u = th.empty((0,), dtype=th.int64)
            self.b01_v = th.empty((0,), dtype=th.int64)

        feat_1 = np.fromfile(feat_1, dtype=np.float32)
        feat_1 = np.reshape(feat_1, (-1, self.in_feats))
        label_1 = np.fromfile(label_1, dtype=np.float32)

        self.p1_u = FF.zerocopy_from_numpy(p1_u) + self.label_0.size(dim=0)
        self.p1_v = FF.zerocopy_from_numpy(p1_v) + self.label_0.size(dim=0)
        self.label_1 = FF.zerocopy_from_numpy(label_1)
        self.feat_1 = FF.zerocopy_from_numpy(feat_1)

        pp(self, '-' * 50)
        pp(self,
           f"rank:{self.rank:02d} RELOADING_v1: {time.time() - tic:.2f} s")
        pp(self, '-' * 50)

    def build_partition(self):
        tic = time.time()
        self.cur_g = dgl.graph((th.concat((self.p0_u, self.b01_u, self.p1_u)),
                                th.concat((self.p0_v, self.b01_v, self.p1_v))))
        self.cur_node_feat = th.concat((self.feat_0, self.feat_1), 0)
        #self.cur_node_label = th.concat((self.label_0, self.label_1), 0)

        pp(self, '-' * 50)
        pp(self,
           f"rank:{self.rank:02d} BLD_GRAPH_v1: {time.time() - tic:.2f} s")
        pp(self, '-' * 50)

    def build_partition_v2_t10n(self):
        tic = time.time()
        self.cur_g = HostGraph(self.part_src.numpy(), self.part_dst.numpy(),
                               self.node_id)

        pp(self, '-' * 50)
        num_edges = self.part_src.shape[0]
        coo_gb = float(num_edges * 8 * 2) / 1024.0 / 1024.0 / 1024.0
        pp(self,
           f"rank:{self.rank:02d} NUM_EDGES: {num_edges}: {coo_gb:.4f}:GB")
        pp(self, f"rank:{self.rank:02d} BLD_GRAPH: {time.time() - tic:.2f} s")
        pp(self, '-' * 50)

    def build_partition_v2(self):
        tic = time.time()
        self.cur_g = dgl.graph((self.part_src, self.part_dst))
        pp(self, self.cur_g)
        #self.cur_node_label = th.concat((self.label_0, self.label_1), 0)
        #counter = 0
        #mask = self.part_src > self.prefix_sum[2]
        #counter = th.count_nonzero(mask)
        #pp(self, f"counter = {counter}")
        #mask2 = self.part_dst > self.prefix_sum[2]
        #counter = th.count_nonzero(mask2)
        #pp(self, f"counter = {counter}")
        #mask = mask & mask2
        #counter = th.count_nonzero(mask)
        #pp(self, f"counter = {counter}")
        #pp(self, self.part_src.shape)
        pp(self, '-' * 50)
        num_edges = self.cur_g.edges()[0].shape[0]
        coo_gb = float(num_edges * 8 * 2) / 1024.0 / 1024.0 / 1024.0
        pp(self,
           f"rank:{self.rank:02d} NUM_EDGES: {num_edges}: {coo_gb:.4f}:GB")
        pp(self, f"rank:{self.rank:02d} BLD_GRAPH: {time.time() - tic:.2f} s")
        pp(self, '-' * 50)

    def get_cur_dataloader(self, args, epoch):

        if epoch % args.repart_every == 0:
            # the simple rr sched
            self.nbr_idx = (self.nbr_idx + 1) % self.world

            if self.nbr_idx == self.rank:
                self.nbr_idx = (self.nbr_idx + 1) % self.world

            # self.cur_g = dgl.graph((self.p0_u, self.p0_v))
            # self.cur_node_feat  = self.feat_0

            # a collator will be created from sampler
            # #### self.collator = NodeCollator(g, nids, graph_sampler, **collator_kwargs)
            # the work is done at self.collator
            # #### self.graph_sampler.sample_blocks(self.g, items)

            # this is for inference
            if self.use_dgl_dsg_sampler:
                # build partition begin
                if self.explore_mode == 1:
                    self.load_nbr_chunk(self.nbr_idx)
                    self.build_partition()
                if self.explore_mode == 2:
                    self.load_nbr_chunk_v2(self.nbr_idx)
                    self.build_partition_v2()
                # build partition end

                if args.model == "pinsage":
                    from t10n.dgl_dsg.sampler import PinSAGESampler
                    self.sampler = PinSAGESampler(
                        self.cur_g,
                        num_layer=args.num_layers,
                        random_walk_length=args.num_layers)
                else:
                    from t10n.dgl_dsg.sampler import IsolatedSamplerOrCoarseCounter as DDIsolatedSampler
                    #from t10n.dgl_dsg.sampler import IsolatedSamplerWithPreciseCounter as DDIsolatedSampler
                    self.sampler = DDIsolatedSampler(
                        [int(fanout) for fanout in args.fan_out.split(",")],
                        self.rank, self.nbr_idx, self.num_part, self.prefix_sum,
                        self.device, self.da, self)

                self.dataloader = dgl.dataloading.DataLoader(
                    self.cur_g,
                    self.train_nid,
                    self.sampler,
                    batch_size=args.batch_size,
                    shuffle=True,  # False for debug
                    drop_last=False,
                )
            else:
                if args.model == "pinsage":
                    assert False
                # build partition begin
                if self.explore_mode == 1:
                    assert False, "not tested"
                if self.explore_mode == 2:
                    self.load_nbr_chunk_v2(self.nbr_idx)
                    self.build_partition_v2_t10n()
                # build partition begin
                self.sampler = None
                self.dataloader = None

                #self.dataloader = HostBatchSampler (
                #    self.cur_g,
                #    self.train_nid,
                #    [int(fanout) for fanout in args.fan_out.split(",")],
                #    repeated=False,
                #    batch_size=args.batch_size,
                #    shuffle=True,
                #    use_batch_n=self.get_max_step())

                # start from local train nodes self.train_nid
                assert self.ideg_0.shape[0] == self.train_mask.shape[0]
                self.dataloader = HostBatchSamplerWithCounter(
                    self.cur_g,
                    self.train_nid,
                    [int(fanout) for fanout in args.fan_out.split(",")],
                    repeated=False,
                    batch_size=args.batch_size,
                    shuffle=True,
                    target_nodes_gideg=self.ideg_0[self.train_nid -
                                                   self.prefix_sum[self.rank]],
                    use_batch_n=self.get_max_step())

                pp(self, self.dataloader)

        return self.dataloader

    def set_coarse_count(self, coarse_count):
        assert self.use_dgl_dsg_sampler
        #pprint(self.rank % 2, f"rank{self.rank}: coarse_count={coarse_count}")
        self.batch_coarse_count = coarse_count

    def get_coarse_count(self):
        if self.use_dgl_dsg_sampler:
            if float(self.batch_coarse_count) > self.max_coarse_count:
                return self.max_coarse_count
        else:
            if isinstance(self.dataloader, HostBatchSamplerWithCounter):
                val = float(self.dataloader.get_cur_resampling_cnt())
                if val > self.max_coarse_count:
                    val =  self.max_coarse_count
                return val
            else:
                return 0.0

    # input_nodes could be halo nodes as well
    # batch_input is to be written
    def get_batch_input(self, batch_input, input_nodes):
        # th.set_printoptions(profile="full")
        if self.explore_mode == 1:
            return self.cur_node_feat[input_nodes]

        if self.num_part == 1:
            batch_input = self.feat_n[self.rank][input_nodes]
            return
        refer_feat_n = [ele.numpy() for ele in self.feat_n]
        self.xb.ad_hoc_fill_batch_feat(batch_input.numpy(), refer_feat_n,
                                       input_nodes.numpy())

        #batch_input = xb_ad_hoc_fill_batch_feat(self.xb,
        #                                        self.da.part_feat_dev(),
        #                                        input_nodes, self.feat_n,
        #                                        self.rank, self.nbr_idx,
        #                                        self.num_part, self.prefix_sum)
        return

    def get_batch_gideg(self, batch_input, input_nodes):
        # th.set_printoptions(profile="full")
        if self.explore_mode == 1:
            assert False

        refer_ideg_n = [ele.numpy() for ele in self.ideg_n]
        self.xb.ad_hoc_fill_batch_ideg(batch_input.numpy(), refer_ideg_n,
                                       input_nodes.numpy())

        #batch_input = xb_ad_hoc_fill_batch_feat(self.xb,
        #                                        self.da.part_feat_dev(),
        #                                        input_nodes, self.feat_n,
        #                                        self.rank, self.nbr_idx,
        #                                        self.num_part, self.prefix_sum)
        return batch_input

    def get_batch_labels(self, seed):
        if self.explore_mode == 1:
            return self.node_label[seed]
        else:
            index = seed - self.prefix_sum[self.rank]
            return self.node_label[index]

    def get_node_label(self):
        return self.node_label

    def get_rank(self):
        return self.rank

    def get_max_step(self):
        return self.local_batch

    def get_comm_dev(self):
        return self.da.comm_dev()

    def get_num_node(self):
        return self.part_num_node[self.rank]

    def get_nid_offset(self):
        return self.prefix_sum[self.rank]

    def get_node_subset(self, subset_tag):
        if subset_tag == "all":
            #target_nodes = th.arange(self.prefix_sum[self.rank],
            #                         self.prefix_sum[self.rank + 1],
            #                         dtype=th.int64)
            #return target_nodes
            return th.concat((self.train_nid, self.valid_nid, self.test_nid))
        elif subset_tag == "train":
            return self.train_nid
        elif subset_tag == "valid":
            return self.valid_nid
        elif subset_tag == "test":
            return self.test_nid
        else:
            assert False, f"{subset_tag} is unknown"
        return None

    #
    # xb_ functions
    #
    def xb_dataloader(self,
                      args,
                      force_even,
                      full_neighbor=False,
                      subset_tag="all"):
        fos = args.fan_out.split(",")
        fanouts = [-1 if full_neighbor else fo for fo in fos]
        if self.use_dgl_dsg_sampler:
            from t10n.dgl_dsg.sampler import XBorderSampler as DDXBorderSampler
            self.infer_sampler = DDXBorderSampler(fanouts, self.rank,
                                                  self.num_part,
                                                  self.prefix_sum, self.xb,
                                                  self.device, self.da)

            target_nodes = self.get_node_subset(subset_tag)
            self.infer_loader = dgl.dataloading.DataLoader(
                self.cur_g,
                target_nodes,
                self.infer_sampler,
                batch_size=args.batch_size_eval,
                shuffle=True,  # False for debug
                drop_last=False,
            )
            return self.infer_loader
        else:
            assert False, "not tested"
            self.infer_sampler = None
            self.infer_loader = None
            return self.infer_loader

    def xb_dataloader_per_layer(self, args, subset_tag="all"):
        target_nodes = self.get_node_subset(subset_tag)

        batch_num = (target_nodes.size()[0] + int(args.batch_size_eval) -
                     1) // int(args.batch_size_eval)
        max_batch = th.tensor([batch_num, batch_num], device=self.da.comm_dev())
        th.distributed.all_reduce(max_batch, op=th.distributed.ReduceOp.MAX)
        max_batch_num = max_batch[0].item()
        assert max_batch_num == batch_num, f"{max_batch_num} == {batch_num}, == is required for BSP"

        if self.use_dgl_dsg_sampler:
            from t10n.dgl_dsg.sampler import XBorderSampler as DDXBorderSampler
            self.infer_sampler = DDXBorderSampler([-1], self.rank,
                                                  self.num_part,
                                                  self.prefix_sum, self.xb,
                                                  self.device, self.da)

            self.infer_loader = dgl.dataloading.DataLoader(
                self.cur_g,
                target_nodes,
                self.infer_sampler,
                batch_size=args.batch_size_eval,
                shuffle=True,  # False for debug
                drop_last=False,
            )
            return self.infer_loader
        else:
            self.infer_sampler = None
            self.infer_loader = None

            self.infer_loader = HostXBBatchSampler(
                self.cur_g,
                target_nodes, [np.iinfo(np.int64).max],
                repeated=False,
                batch_size=args.batch_size_eval,
                shuffle=True,
                xb=self.xb,
                rank=self.rank,
                num_part=self.num_part,
                device_allocator=self.da)
            return self.infer_loader

    @timing
    def xb_get_batch_input(self, batch_input, target_nodes):
        return xb_gather(self.xb, self.da.comm_dev(), batch_input, target_nodes,
                         self.feat_n[self.rank], self.rank, self.num_part,
                         self.prefix_sum)

    def xb_get_batch_emb(self, batch_input, target_nodes, emb_self):
        return xb_gather(self.xb, self.da.comm_dev(), batch_input, target_nodes,
                         emb_self, self.rank, self.num_part, self.prefix_sum)

    def infer_train_mask_of(self, seeds):
        seeds = seeds - self.prefix_sum[self.rank]
        mask = self.train_mask[seeds]
        return mask

    def infer_valid_mask_of(self, seeds):
        seeds = seeds - self.prefix_sum[self.rank]
        mask = self.valid_mask[seeds]
        return mask

    def infer_test_mask_of(self, seeds):
        seeds = seeds - self.prefix_sum[self.rank]
        mask = self.test_mask[seeds]
        return mask
