import os, pathlib
import time
import json

s = time.time()
from t10n._C import c_seq
from .meta import name_to_meta, t10n_chunk_path
from ..util import timing

import numpy as np
import torch as th

import dgl
import dgl.backend as F
from dgl.partition import get_peak_mem
from dgl.partition import metis_partition_assignment as metis_assignment
from dgl.random import choice as random_choice

print(
    f"python import: {time.time()- s:.2f} s, peak_mem: {get_peak_mem():.3f} GB")

#from dgl.partition import partition_graph_vertex_cut_with_halo as custom_vc_assignment
#def custom_vc_assignment():
#    pass


@timing
def get_u_v_nid2pid(cfg_json, part_method, n_parts):
    edge_file = cfg_json['edge_file_bin']
    edges = np.load(edge_file)
    num_nodes = int(cfg_json['num_nodes'])
    num_edges = int(cfg_json['num_edges'])
    assert len(edges) // 2 == num_edges
    u, v = edges[:num_edges], edges[num_edges:]
    edges = None
    uu = F.zerocopy_from_numpy(u)
    vv = F.zerocopy_from_numpy(v)

    # create DGL graph and do metis assignment
    if n_parts == 1:
        node_parts = th.zeros((num_nodes,), dtype=th.int64)
    elif part_method == 'metis':
        g = dgl.graph((uu, vv))
        uu, vv = None, None
        train_mask = np.load(cfg_json['train_mask_file_bin'])
        balance_ntypes = th.tensor(train_mask)
        node_parts = metis_assignment(g,
                                      n_parts,
                                      mode='k-way',
                                      balance_ntypes=balance_ntypes)
    elif part_method == 'random':
        node_parts = random_choice(n_parts, num_nodes)


#   elif part_method == 'vcdeg':
#        edge_file_bin = cfg_json['edge_file_bin']
#        num_nodes = cfg_json['num_nodes']
#        num_edges = cfg_json['num_edges']
#        train_mask_bin = cfg_json['train_mask_file_bin']
#        num_train_nodes = cfg_json['num_train_nodes']
#        add_rev_edge = cfg_json['add_reverse_edge']
#
#        vc_maps, parts, _, _ = custom_vc_assignment(
#            edge_file_bin,
#            num_nodes,
#            num_edges,
#            n_parts,
#            'vcdeg',
#            0,
#            reshuffle=False,  # 0 is num_hops
#            num_train_nodes=num_train_nodes,
#            train_mask_file=train_mask_bin,
#            add_rev_edge=add_rev_edge)
#
#        VCR_MPID_MASK = 0xFFFF
#        vc_maps[0] = vc_maps[0] & VCR_MPID_MASK
#        node_parts = vc_maps[0]
    else:
        assert False, f"not supported method: {part_method}"
    nid2pid = F.zerocopy_to_numpy(node_parts)
    assert nid2pid.dtype == np.int64
    return u, v, nid2pid


@timing
def _build_check(output_dir, n_n, n_e, n_dim, n_part, o_u, o_v, nid2pid):
    assert o_u.dtype == np.int64
    assert o_v.dtype == np.int64
    assert nid2pid.dtype == np.int64
    c_seq.build(output_dir, n_n, n_e, n_dim, n_part, o_u, o_v, nid2pid)


@timing
def _split_node_feat_check(output_dir, n_n, n_dim, n_part, node_feat):
    assert node_feat.dtype == np.float32
    c_seq.split_node_feat(output_dir, n_n, n_dim, n_part, node_feat)


@timing
def _split_node_data_check(output_dir, n_n, n_part, node_label, train_mask,
                           val_mask, test_mask):
    assert node_label.dtype == np.float32
    assert train_mask.dtype == np.int8
    assert val_mask.dtype == np.int8
    assert test_mask.dtype == np.int8
    c_seq.split_node_data(output_dir, n_n, n_part, node_label, train_mask,
                          val_mask, test_mask)


@timing
def build_chunks_seq(dest_path: str, dataset_name: str, src_path: str,
                     part_method: str, n_part: str):
    SUPPORTED_METHOD = ['random', 'metis']
    assert part_method in SUPPORTED_METHOD
    meta_o = name_to_meta(dataset_name)
    if dataset_name == 'igb260m':
        assert False

    output_dir = os.path.join(
        dest_path, t10n_chunk_path(dataset_name, part_method, n_part))
    if os.path.exists(output_dir):
        assert False, f"please delete {output_dir} manually"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    t10n_cfg_f = os.path.join(src_path, "t10n.json")
    cfg = None
    with open(t10n_cfg_f) as cfg_f:
        cfg = json.load(cfg_f)
    # get partition result
    o_u, o_v, nid2pid = get_u_v_nid2pid(cfg, part_method, n_part)
    n_n, n_e, n_dim = int(cfg['num_nodes']), int(cfg['num_edges']), int(
        cfg['feat_dim'])
    # build chunks
    _build_check(output_dir, n_n, n_e, n_dim, n_part, o_u, o_v, nid2pid)

    # split node feat
    node_feat = np.load(cfg['node_feats_file_bin'])
    _split_node_feat_check(output_dir, n_n, n_dim, n_part, node_feat)

    # split node data
    node_label = cfg['node_label_file']
    train_mask = cfg['train_mask_file_bin']
    val_mask = cfg['val_mask_file_bin']
    test_mask = cfg['test_mask_file_bin']
    node_label = np.load(node_label)
    train_mask = np.load(train_mask)
    val_mask = np.load(val_mask)
    test_mask = np.load(test_mask)
    _split_node_data_check(output_dir, n_n, n_part, node_label, train_mask,
                           val_mask, test_mask)
