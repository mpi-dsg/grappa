import os
import argparse

import torch as th
import json
import numpy as np
import pandas as pd

import dgl
from dgl.data import CoraGraphDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
#from igb.dataloader import IGBHeteroDGLDataset

from .meta import name_to_meta
from ..util import timing


@timing
def load_as_dgl_g(dest_path, dataset_name):
    meta_o = name_to_meta(dataset_name)
    print(f"load_as_dgl_g: start {dataset_name}")
    if 'ogb' in dataset_name:
        data = DglNodePropPredDataset(name=meta_o.origin_name, root=dest_path)
        print(f"load_as_dgl_g: finish {dataset_name}")
        splitted_idx = data.get_idx_split()
        graph, labels = data[0]
        labels = labels[:, 0]

        graph.ndata["label"] = labels

        # Find the node IDs in the training, validation, and test set.
        train_nid, val_nid, test_nid = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
        train_mask[train_nid] = True
        val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
        val_mask[val_nid] = True
        test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
        test_mask[test_nid] = True
        graph.ndata["train_mask"] = train_mask
        graph.ndata["val_mask"] = val_mask
        graph.ndata["test_mask"] = test_mask

        return graph

    if dataset_name == 'cora':
        dataset = CoraGraphDataset(raw_dir=dest_path)
        dgl_g = dataset[0]
        print(f"load_as_dgl_g: finish {dataset_name}")
        return dgl_g
    if dataset_name == 'reddit':
        dataset = RedditDataset(raw_dir=dest_path)
        dgl_g = dataset[0]
        print(f"load_as_dgl_g: finish {dataset_name}")
        return dgl_g
    if dataset_name == 'mag':
        dataset, b = dgl.load_graphs(dest_path+"mag240m/mag")
        print(b)
        dgl_g = dataset[0]
        print(dgl_g.ndata)
        dgl_g = dgl_g.formats(['coo', 'csc'])
        dgl_g.ndata["label"] = th.from_numpy(np.load(dest_path+"mag240m/full_label.npy"))
        dgl_g.ndata["train_mask"] = th.from_numpy(np.load(dest_path+"mag240m/full_train_mask.npy"))
        dgl_g.ndata["val_mask"] = th.from_numpy(np.load(dest_path+"mag240m/full_valid_mask.npy"))
        dgl_g.ndata["test_mask"] = th.from_numpy(np.load(dest_path+"mag240m/full_test_mask.npy"))
        print(f"load_as_dgl_g: finish {dataset_name}")
        full_feat = np.memmap(dest_path+"mag240m/full/output", dtype='float16', mode='r', shape=(244160499, 768))
        dgl_g.ndata['feat'] = th.from_numpy(full_feat)
        return dgl_g
    if dataset_name == 'igb260m':
        from igb.dataloader import IGB260MDGLDataset

        class igb260m_args:

            def __init__(self, path):
                self.path = path
                self.dataset_size = "full"  #260m
                self.num_classes = 19
                self.in_memory = 0  # mmap
                self.synthetic = 0  # nlp emb

        args = igb260m_args(dest_path)
        dataset = IGB260MDGLDataset(args)
        dgl_g = dataset[0]
        print(f"load_as_dgl_g: finish {dataset_name}")
        return dgl_g

    assert False, f"{dataset_name} is not handled"


@timing
def augmentation_dgl_g(dgl_g):
    dgl_g = dgl.remove_self_loop(dgl_g)
    dgl_g = dgl.add_self_loop(dgl_g)
    return dgl_g


def get_num_labels(labels):
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    return num_labels


@timing
def to_t10n_from_dgl_g(dest_path, dataset_name, dgl_g):
    meta_o = name_to_meta(dataset_name)

    if dataset_name == 'igb260m':
        assert False, "igb260m should not use this funciton, because it's too slow"

    output_dir = os.path.join(dest_path, meta_o.name)
    os.makedirs(output_dir, exist_ok=True)
    t10n_cfg_f = os.path.join(output_dir, "t10n.json")
    if os.path.exists(t10n_cfg_f):
        print("already preprocessed for t10n")
        return t10n_cfg_f

    edge_file_bin = os.path.join(output_dir, "edge_index.bin.npy")
    node_feat_file_bin = os.path.join(output_dir, "node_feat.bin.npy")

    node_label_file_bin = os.path.join(output_dir, "node_label.bin.npy")
    train_mask_file_bin = os.path.join(output_dir, "train_mask.bin.npy")
    val_mask_file_bin = os.path.join(output_dir, "valid_mask.bin.npy")
    test_mask_file_bin = os.path.join(output_dir, "test_mask.bin.npy")

    # all data are saved by:
    #   np.save(file_path, array)
    # load by:
    #   np.load(file_path)
    #
    # edges       numpy array: np.int64
    # labels      numpy array: np.float32
    # node_feat   numpy array: np.float32
    # train_mask  numpy array: np.int8
    # val_mask    numpy array: np.int8
    # test_mask   numpy array: np.int8

    src, dst = dgl_g.edges()
    edge = th.concat((src, dst)).numpy()
    edge = edge.astype(np.int64)
    np.save(edge_file_bin, edge)

    nl = dgl_g.ndata['label'].numpy()
    nl = nl.astype(np.float32).flatten()
    np.save(node_label_file_bin, nl)

    node_feat = dgl_g.ndata['feat'].numpy()
    node_feat = node_feat.astype(np.float32)

    if 'int' in str(node_feat.dtype):
        print("Warning: node_feat was converted to np.float32 from int")
    np.save(node_feat_file_bin, node_feat)

    train_mask = dgl_g.ndata['train_mask'].numpy()
    train_mask = train_mask.astype(np.int8)
    np.save(train_mask_file_bin, train_mask)

    val_mask = dgl_g.ndata['val_mask'].numpy()
    val_mask = val_mask.astype(np.int8)
    np.save(val_mask_file_bin, val_mask)

    test_mask = dgl_g.ndata['test_mask'].numpy()
    test_mask = test_mask.astype(np.int8)
    np.save(test_mask_file_bin, test_mask)

    dgl_g_feat_dim = node_feat.shape[1]
    assert dgl_g_feat_dim == meta_o.n_dim
    num_train_nodes = np.count_nonzero(train_mask)
    num_labels = get_num_labels(dgl_g.ndata['label'])

    cfg_json = {}
    cfg_json['node_label_file'] = node_label_file_bin
    cfg_json['edge_file_bin'] = edge_file_bin
    cfg_json['node_feats_file_bin'] = node_feat_file_bin
    cfg_json['train_mask_file_bin'] = train_mask_file_bin
    cfg_json['val_mask_file_bin'] = val_mask_file_bin
    cfg_json['test_mask_file_bin'] = test_mask_file_bin
    cfg_json['num_train_nodes'] = num_train_nodes
    cfg_json['num_nodes'] = dgl_g.num_nodes()
    cfg_json['num_edges'] = dgl_g.num_edges()
    cfg_json['feat_dim'] = dgl_g_feat_dim
    cfg_json['num_labels'] = num_labels
    # if add reverse or not IN t10n
    cfg_json['add_reverse_edge'] = False

    with open(t10n_cfg_f, 'w+') as f:
        json.dump(cfg_json, f, indent=4)
    return t10n_cfg_f


def _igb_dataset_to_t10n(dest_path, src_path):
    meta_o = name_to_meta('igb260m')
    input_path = os.path.join(src_path, meta_o.origin_path)
    output_dir = os.path.join(dest_path, meta_o.name)
    os.makedirs(output_dir, exist_ok=True)
    t10n_cfg_f = os.path.join(output_dir, "t10n.json")
    if os.path.exists(t10n_cfg_f):
        print("already preprocessed for t10n")
        return t10n_cfg_f

    #####################################
    # large dataset
    # use .mmap.npy format, read by np.memmap
    # use .bin.npy format, read by np.load,
    # igb edge_idx format: [[src, dst], [src, dst]...]
    #####################################
    edge_file_bin = os.path.join(output_dir, "edge_index.bin.npy")
    node_feat_file_bin = os.path.join(output_dir, "node_feat.mmap.npy")

    node_label_file_bin = os.path.join(output_dir, "node_label.mmap.npy")
    # no predefined
    train_mask_file_bin = os.path.join(output_dir, "train_mask.bin.npy")
    val_mask_file_bin = os.path.join(output_dir, "valid_mask.bin.npy")
    test_mask_file_bin = os.path.join(output_dir, "test_mask.bin.npy")

    train_mask = th.zeros(meta_o.n_n, dtype=th.int8)
    val_mask = th.zeros(meta_o.n_n, dtype=th.int8)
    test_mask = th.zeros(meta_o.n_n, dtype=th.int8)
    n_train = int(meta_o.n_n * 0.6)
    n_val = int(meta_o.n_n * 0.2)
    train_mask[:n_train] = 1
    val_mask[n_train:n_train + n_val] = 1
    test_mask[n_train + n_val:-1] = 1
    np.save(train_mask_file_bin, train_mask.numpy())
    np.save(val_mask_file_bin, val_mask.numpy())
    np.save(test_mask_file_bin, test_mask.numpy())

    #-------------------------------------------
    src_label_file_bin = os.path.join(
        input_path, "paper/node_label_19.npy")  # np.memmap(np.float32)
    label_19 = np.memmap(src_label_file_bin, dtype=np.float32, mode='r')
    assert label_19.dtype == np.float32
    assert label_19.shape[0] == meta_o.n_n
    try:
        symbolic_link = node_label_file_bin
        os.symlink(src_label_file_bin, symbolic_link)
    except OSError as e:
        print(f"Failed to create symbolic link: {e}")

    #-------------------------------------------
    src_node_feat_file_bin = os.path.join(
        input_path, "paper/node_feat.npy")  # np.memmap(np.float32)
    node_feat = np.memmap(src_node_feat_file_bin,
                          dtype=np.float32,
                          mode='r',
                          shape=(meta_o.n_n, meta_o.n_dim))
    assert node_feat.shape == (meta_o.n_n, meta_o.n_dim)
    try:
        symbolic_link = node_feat_file_bin
        os.symlink(src_node_feat_file_bin, symbolic_link)
    except OSError as e:
        print(f"Failed to create symbolic link: {e}")

    #-------------------------------------------
    src_edge_file = os.path.join(input_path,
                                 "paper__cites__paper/edge_index.npy")
    # src_edge_file is not added self-loop
    edge_file = np.load(src_edge_file, mmap_mode='r')
    assert edge_file.shape == (meta_o.origin_n_e, 2)
    assert edge_file.dtype == np.int64
    try:
        symbolic_link = edge_file_bin
        os.symlink(src_edge_file, symbolic_link)
    except OSError as e:
        print(f"Failed to create symbolic link: {e}")

    num_train_nodes = n_train
    num_nodes = meta_o.n_n
    num_edges = meta_o.origin_n_e  # not augmentation of this graph
    feat_dim = meta_o.n_dim
    num_labels = 19  # the label file of 19 class was selected

    cfg_json = {}
    cfg_json['node_label_file'] = node_label_file_bin
    cfg_json['edge_file_bin'] = edge_file_bin
    cfg_json['node_feats_file_bin'] = node_feat_file_bin
    cfg_json['train_mask_file_bin'] = train_mask_file_bin
    cfg_json['val_mask_file_bin'] = val_mask_file_bin
    cfg_json['test_mask_file_bin'] = test_mask_file_bin
    cfg_json['num_train_nodes'] = num_train_nodes
    cfg_json['num_nodes'] = num_nodes
    cfg_json['num_edges'] = num_edges
    cfg_json['feat_dim'] = feat_dim
    cfg_json['num_labels'] = num_labels
    # if add reverse or not IN t10n
    cfg_json['add_reverse_edge'] = False

    with open(t10n_cfg_f, 'w+') as f:
        json.dump(cfg_json, f, indent=4)
    return t10n_cfg_f


def _ogbpa_dataset_to_t10n(dest_path, src_path):
    meta_o = name_to_meta('ogbpa')
    input_path = os.path.join(src_path, meta_o.origin_path)
    mask_path = os.path.join(src_path, meta_o.origin_name, "split/time")
    output_dir = os.path.join(dest_path, meta_o.name)
    os.makedirs(output_dir, exist_ok=True)
    t10n_cfg_f = os.path.join(output_dir, "t10n.json")
    if os.path.exists(t10n_cfg_f):
        print("already preprocessed for t10n")
        return t10n_cfg_f

    # ogbpa t10n format should be the same as to_t10n_from_dgl_g()
    edge_file_bin = os.path.join(output_dir, "edge_index.bin.npy")
    node_feat_file_bin = os.path.join(output_dir, "node_feat.bin.npy")

    node_label_file_bin = os.path.join(output_dir, "node_label.bin.npy")
    train_mask_file_bin = os.path.join(output_dir, "train_mask.bin.npy")
    val_mask_file_bin = os.path.join(output_dir, "valid_mask.bin.npy")
    test_mask_file_bin = os.path.join(output_dir, "test_mask.bin.npy")

    # all data are saved by:
    #   np.save(file_path, array)
    # load by:
    #   np.load(file_path)
    #
    # edges       numpy array: np.int64
    # labels      numpy array: np.float32
    # node_feat   numpy array: np.float32
    # train_mask  numpy array: np.int8
    # val_mask    numpy array: np.int8
    # test_mask   numpy array: np.int8

    label_file = os.path.join(input_path, "node-label.npz")
    data_file = os.path.join(input_path, "data.npz")
    train_mask_file = os.path.join(mask_path, "train.csv.gz")
    val_mask_file = os.path.join(mask_path, "valid.csv.gz")
    test_mask_file = os.path.join(mask_path, "test.csv.gz")

    # node label : convert to t10n
    node_label = np.load(label_file, mmap_mode='r')
    lst = node_label.files
    assert len(lst) == 1
    nl = node_label[lst[0]].astype(np.float32).flatten()
    np.save(node_label_file_bin, nl)
    num_labels = get_num_labels(nl)

    # data ditc : convert to t10n
    data_dict = np.load(data_file, mmap_mode='r')
    num_nodes_list = data_dict['num_nodes_list']
    num_edges_list = data_dict['num_edges_list']
    assert len(num_nodes_list) == 1, "ogbpa is homo"
    assert len(num_edges_list) == 1, "ogbpa is homo"
    num_nodes = num_nodes_list[0]
    num_edges = num_edges_list[0]

    assert 'edge_idx' in data_dict.keys()
    assert data_dict['edge_idx'].dtype == np.int64
    assert data_dict['edge_idx'].shape == (meta_o.origin_n_e, 2)
    src = data_dict['edge_idx'][:, 0]
    dst = data_dict['edge_idx'][:, 1]
    dgl_g = dgl.graph((src, dst))
    dgl_g = augmentation_dgl_g(dgl_g)
    src, dst = dgl_g.edges()
    edge = th.concat((src, dst)).numpy()
    np.save(edge_file_bin, edge)
    src, dst, edge = None, None, None

    assert 'node_feat' in data_dict.keys()
    if 'int' in str(data_dict['node_feat'].dtype):
        assert False, "node_feat of type int, refer function to_t10n_from_dgl_g"
    assert data_dict['node_feat'].dtype == np.float32
    assert data_dict['node_feat'].shape == (meta_o.n_n, meta_o.n_dim)
    np.save(node_feat_file_bin, data_dict['node_feat'])

    def _ogbpa_load_mask_file(mask_file: str):
        mask_idx = th.as_tensor(
            pd.read_csv(mask_file,
                        compression='gzip', header=None).values.T[0]).to(
                            th.long)  # (num_graph, ) python list
        mask = th.zeros((num_nodes,), dtype=th.bool)
        mask[mask_idx] = True
        num_nodes = mask_idx.shape[0]
        return mask, num_nodes

    train_mask, num_train_nodes = _ogbpa_load_mask_file(train_mask_file)
    train_mask = train_mask.astype(np.int8)
    np.save(train_mask_file_bin, train_mask)

    val_mask, _ = _ogbpa_load_mask_file(val_mask_file)
    val_mask = val_mask.astype(np.int8)
    np.save(val_mask_file_bin, val_mask)

    test_mask, _ = _ogbpa_load_mask_file(test_mask_file)
    test_mask = test_mask.astype(np.int8)
    np.save(test_mask_file, test_mask)

    cfg_json = {}
    cfg_json['node_label_file'] = node_label_file_bin
    cfg_json['edge_file_bin'] = edge_file_bin
    cfg_json['node_feats_file_bin'] = node_feat_file_bin
    cfg_json['train_mask_file_bin'] = train_mask_file_bin
    cfg_json['val_mask_file_bin'] = val_mask_file_bin
    cfg_json['test_mask_file_bin'] = test_mask_file_bin
    cfg_json['num_train_nodes'] = num_train_nodes
    cfg_json['num_nodes'] = num_nodes
    cfg_json['num_edges'] = num_edges
    cfg_json['feat_dim'] = meta_o.n_dim
    cfg_json['num_labels'] = num_labels
    # if add reverse or not IN t10n
    cfg_json['add_reverse_edge'] = False

    with open(t10n_cfg_f, 'w+') as f:
        json.dump(cfg_json, f, indent=4)
    return t10n_cfg_f


@timing
def to_t10n_from_downloaded_ds(dest_path, dataset_name, src_path):
    supported_dataset = ['ogbpa', 'igb260m']
    assert dataset_name in supported_dataset, f"{dataset_name} is not supported to preprocess from downloaed dataset"
    if dataset_name == 'ogbpa':
        _ogbpa_dataset_to_t10n(dest_path, src_path)
    if dataset_name == 'igb260m':
        _igb_dataset_to_t10n(dest_path, src_path)
