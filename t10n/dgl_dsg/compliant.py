from t10n._C.graph import Batch
from t10n.util import timing

import torch
import dgl
from dgl.base import NID #, EID
from dgl.transforms import to_block
import copy

def to_dgl_batch(output_nodes, b: Batch):
    assert torch.is_tensor(output_nodes), "only handle tensor input"

    blocks = []
    dst_nodes = output_nodes

    all_layers = b.get_layers()
    #all_inputs = b.get_input_nodes()
    n_layer = len(all_layers) // 2

    for i in range(n_layer):
        u = torch.as_tensor(all_layers[2*i])
        v = torch.as_tensor(all_layers[2*i+1])
        sub_g = dgl.graph((u, v))
        block = to_block(sub_g, dst_nodes)
        dst_nodes = block.srcdata[NID]
        #dst_nodes = torch.as_tensor(all_inputs[i])
        blocks.insert(0, block)

    return dst_nodes, output_nodes, blocks

#@timing
def to_dgl_batch(output_nodes, all_layers, all_inputs=None):
    assert torch.is_tensor(output_nodes), "only handle tensor input"

    blocks = []
    dst_nodes = output_nodes

    n_layer = len(all_layers) // 2

    for i in range(n_layer):
        u = torch.as_tensor(all_layers[2*i])
        v = torch.as_tensor(all_layers[2*i+1])
        sub_g = dgl.graph((u, v))
        block = to_block(sub_g, dst_nodes)
        #dst_nodes = torch.as_tensor(all_inputs[i])
        dst_nodes = block.srcdata[NID]
        blocks.insert(0, block)

    return dst_nodes, output_nodes, blocks

