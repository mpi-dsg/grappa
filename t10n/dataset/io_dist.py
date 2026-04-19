#  Copyright (c) 2024-2026 by MPI-SWS, Germany
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

from .meta import name_to_meta

import os
import json
import numpy as np

import torch as th


def per_rank_stride(world_size: int, rank: int, total_num: int):
    stride = (total_num + world_size - 1) // world_size
    return stride


def per_rank_local_num(world_size: int, rank: int, total_num: int):
    stride = per_rank_stride(world_size, rank, total_num)
    if rank == (world_size - 1):
        mod = total_num % stride
        return stride if mod == 0 else mod
    else:
        return stride


def per_rank_padding(world_size: int, rank: int, total_num: int):
    mod = total_num % world_size
    if mod == 0:
        return total_num
    else:
        return total_num - mod + world_size


class SplitConfig:

    def __init__(self, world_size, rank, nid2pid, nid2lid, num_part, ds_meta,
                 nodes_per_part):
        assert len(nodes_per_part) == num_part
        assert num_part % world_size == 0

        self.world_size = world_size
        self.rank = rank
        self.nid2pid = nid2pid
        self.nid2lid = nid2lid
        self.num_part = num_part
        self.ds_meta = ds_meta
        self.nodes_per_part = nodes_per_part


class DatasetIO:

    def __init__(self, src_path, dest_path):
        t10n_cfg_f = os.path.join(src_path, "t10n.json")
        self.ds_cfg = None
        self.dest_path = dest_path
        with open(t10n_cfg_f) as cfg_f:
            self.ds_cfg = json.load(cfg_f)

        self.allowed_keys = [
            'node_label_file', 'node_feats_file_bin', 'train_mask_file_bin',
            'val_mask_file_bin', 'test_mask_file_bin', 'edge_file_bin', 'ideg'
        ]
        # edge_file_bin

    def read(self, key, oid_start, oid_end):
        pass

    def zero(self, key, num_node):
        pass

    def write(self, key, pidx, data_tensor):
        pass

    # [ off_start, off_end)
    def read_edge(self, off_start, off_end):
        pass

    def write_primary(self, pidx, p_u, p_v):
        pass

    def write_bridge(self, pidx, b_u, pjdx, b_v):
        pass

    def check_npy_dtype(self, key, npy_arr):
        assert npy_arr.dtype == self.get_npy_dtype(key)

    def get_npy_dtype(self, key):
        assert key in self.allowed_keys
        if key == 'node_label_file':
            return np.float32
        elif key == 'edge_file_bin':
            return np.int64
        elif key == 'node_feats_file_bin':
            return np.float32
        elif key == 'train_mask_file_bin':
            return np.int8
        elif key == 'val_mask_file_bin':
            return np.int8
        elif key == 'test_mask_file_bin':
            return np.int8
        elif key == 'ideg':
            return np.int64
        else:
            assert False

    def get_th_dtype(self, key):
        assert key in self.allowed_keys
        if key == 'node_label_file':
            return th.float32
        elif key == 'edge_file_bin':
            return th.int64
        elif key == 'node_feats_file_bin':
            return th.float32
        elif key == 'train_mask_file_bin':
            return th.int8
        elif key == 'val_mask_file_bin':
            return th.int8
        elif key == 'test_mask_file_bin':
            return th.int8
        elif key == 'ideg':
            return th.int64
        else:
            assert False

    # the shape of dataset on disk
    def get_npy_shape(self, key):
        n_n = self.ds_cfg['num_nodes']
        n_e = self.ds_cfg['num_edges']
        n_dim = self.ds_cfg['feat_dim']
        if key == 'node_feats_file_bin':
            return (n_n, n_dim)
        if key == 'node_label_file':
            return (n_n,)
        elif key == 'edge_file_bin':
            assert False, "edges dont use this"
        elif key == 'train_mask_file_bin':
            return (n_n,)
        elif key == 'val_mask_file_bin':
            return (n_n,)
        elif key == 'test_mask_file_bin':
            return (n_n,)
        elif key == 'ideg':
            return (n_n,)
        else:
            assert False

    def get_node_data_fname(self, key, pidx):
        assert key in self.allowed_keys
        if key == 'node_label_file':
            return f"p{pidx}.label.bin"
        elif key == 'edge_file_bin':
            assert False, "edges dont use this"
        elif key == 'node_feats_file_bin':
            return f"p{pidx}.nfeat.bin"
        elif key == 'train_mask_file_bin':
            return f"p{pidx}.train.bin"
        elif key == 'val_mask_file_bin':
            return f"p{pidx}.valid.bin"
        elif key == 'test_mask_file_bin':
            return f"p{pidx}.test.bin"
        elif key == 'ideg':
            return f"p{pidx}.ideg.bin"
        else:
            assert False


class OgbprIO(DatasetIO):

    def __init__(self, src_path, dest_path):
        super().__init__(src_path, dest_path)

    def read(self, key, oid_start, oid_end):
        assert key in self.allowed_keys
        assert key != 'edge_file_bin'
        file_path = self.ds_cfg[key]
        mmaped_npy = None
        if file_path[-8:] == ".bin.npy":
            mmaped_npy = np.load(file_path, mmap_mode='r')
        elif file_path[-8:] == "mmap.npy":
            mmaped_npy = np.memmap(file_path,
                                   dtype=self.get_npy_dtype(key),
                                   mode='r',
                                   shape=self.get_npy_shape(key))
        else:
            assert False
        assert mmaped_npy.shape == self.get_npy_shape(key)
        self.check_npy_dtype(key, mmaped_npy)
        read_in = mmaped_npy[oid_start:oid_end]
        return th.from_numpy(read_in)

    def zero(self, key, num_node):
        assert key in self.allowed_keys
        if key == 'node_feats_file_bin':
            return th.zeros((num_node, self.ds_cfg['feat_dim']),
                            dtype=self.get_th_dtype(key))
        else:
            return th.zeros((num_node,), dtype=self.get_th_dtype(key))

    def write(self, key, pidx, data_tensor):
        assert key in self.allowed_keys
        assert th.is_tensor(data_tensor)
        self.check_npy_dtype(key, data_tensor.numpy())
        output_fname = self.get_node_data_fname(key, pidx)
        # t10n dataloader will load from binary by np.fromfile
        file_name = self.dest_path + "/" + output_fname
        data_tensor.numpy().tofile(file_name)

    # [ off_start, off_end)
    def read_edge(self, off_start, off_end):
        key = 'edge_file_bin'
        assert key in self.allowed_keys
        file_path = self.ds_cfg[key]
        assert file_path[-8:] == ".bin.npy"
        mmaped_npy = np.load(file_path, mmap_mode='r')
        self.check_npy_dtype(key, mmaped_npy)
        assert mmaped_npy.shape[0] % 2 == 0
        length = mmaped_npy.shape[0] // 2
        assert length == self.ds_cfg['num_edges']
        u = mmaped_npy[off_start:off_end]
        v = mmaped_npy[length + off_start:length + off_end]
        return th.tensor(u), th.tensor(v)

    def write_primary(self, pidx, p_u, p_v):
        assert th.is_tensor(p_u)
        assert th.is_tensor(p_v)
        assert p_u.dtype == th.int64
        assert p_v.dtype == th.int64
        file_name = self.dest_path + "/" + f"p{pidx}.u.bin"
        # t10n dataloader will load from binary by np.fromfile
        p_u.numpy().tofile(file_name)

        file_name = self.dest_path + "/" + f"p{pidx}.v.bin"
        # t10n dataloader will load from binary by np.fromfile
        p_v.numpy().tofile(file_name)

    def write_bridge(self, pidx, b_u, pjdx, b_v):
        assert th.is_tensor(b_u)
        assert th.is_tensor(b_v)
        assert b_u.dtype == th.int64
        assert b_v.dtype == th.int64
        file_name = self.dest_path + "/" + f"b{pidx}_{pjdx}.u.bin"
        # t10n dataloader will load from binary by np.fromfile
        b_u.numpy().tofile(file_name)

        file_name = self.dest_path + "/" + f"b{pidx}_{pjdx}.v.bin"
        # t10n dataloader will load from binary by np.fromfile
        b_v.numpy().tofile(file_name)


class Igb260mIO(OgbprIO):

    def __init__(self, src_path, dest_path):
        super().__init__(src_path, dest_path)

    # [ off_start, off_end)
    def read_edge(self, off_start, off_end):
        key = 'edge_file_bin'
        assert key in self.allowed_keys
        file_path = self.ds_cfg[key]
        assert file_path[-8:] == ".bin.npy"
        mmaped_npy = np.load(file_path, mmap_mode='r')
        self.check_npy_dtype(key, mmaped_npy)
        assert mmaped_npy.shape[1] == 2
        length = mmaped_npy.shape[0]
        cfg_n_e = int(self.ds_cfg['num_edges'])
        assert length == cfg_n_e, f"len={length} != {cfg_n_e}"
        proption = mmaped_npy[off_start:off_end, :]
        u = th.from_numpy(proption[:, 0])
        v = th.from_numpy(proption[:, 1])
        return u, v


def get_dataset_io(src_path, dest_path, graph_name):
    ds_meta = name_to_meta(graph_name)
    if graph_name == "ogbpr":
        reader = OgbprIO(src_path, dest_path)
        return reader
    if graph_name == "ogbpa":
        # same as ogbpr
        reader = OgbprIO(src_path, dest_path)
        return reader
    if graph_name == "igb260m":
        reader = Igb260mIO(src_path, dest_path)
        return reader
    assert False, f"{graph_name} is not supported"
