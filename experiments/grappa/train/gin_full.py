import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn.pytorch as dglnn

# from dgl.nn.pytorch.conv import GINConv
# from dgl.nn.pytorch.glob import SumPooling

from contextlib import contextmanager

def sum_pool(block, feat):
    pass
#    src_ids, dst_ids = block.edges(order='eid')
#    converted_graph = dgl.graph((src_ids, dst_ids), num_nodes=block.num_src_nodes())
#    dglnn.glob.SumPooling(converted_graph, feat)


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = self.linears[0](h)
        h = self.batch_norm(h)
        h = F.relu(h)
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.n_hidden = hidden_dim
        num_layers = num_layers
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.layers.append(
                dglnn.conv.GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        mlp = MLP(hidden_dim, hidden_dim, output_dim)
        self.layers.append( dglnn.conv.GINConv(mlp, learn_eps=False) )
        self.batch_norms.append(nn.BatchNorm1d(output_dim))

        """
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            dglnn.glob.SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
        """

    def forward(self, graph, h):
        # list of hidden representation at each layer (including the input layer)
        # print(blocks[0])
        # print(blocks[0].ndata)
        # print(blocks[0].ntypes)
        # print("--", h)
        hidden_rep = [h]
        # print("len of self.layers: ", len(self.layers))
        for i, layer in enumerate(self.layers):
            # print(f"{i}- h.shape={h.shape}")
            h = layer(graph, h)
            # print(f"{i}+ h.shape={h.shape}")
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
            # check hidden h
            # print(f"block_{i}: ", blocks[i])
            # print("#dst_nodes: ", blocks[i].num_dst_nodes())
            # print("#src_nodes: ", blocks[i].num_src_nodes())

            # print(h.shape)

        logits = hidden_rep[-1]
        # print(f"logits shape: {logits.shape}")
        # print(f"logits type : {logits.dtype}")
        return logits

        """
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            src_ids, dst_ids = blocks[i].edges()
            converted_graph = dgl.graph((src_ids, dst_ids), num_nodes=blocks[i].num_src_nodes())
            pooled_h = self.pool(converted_graph, h)
            print("pooled_h shape: ",  pooled_h.shape)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        print("score_over_layer shape: ",  score_over_layer.shape)
        _, predicted = torch.max(score_over_layer, 1)

        return predicted.float()
        """

    def inference(self, g, x, batch_size, device):
        pass

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield
