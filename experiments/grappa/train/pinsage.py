# ported from dgl/examples, gnnlab/example/dgl

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from contextlib import contextmanager


class WeightedSAGEConv(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims, dropout, act):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0,
                                 torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


class PinSAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        self.layers.append(
            WeightedSAGEConv(in_feats, n_hidden, n_hidden, dropout, activation))
        for _ in range(1, n_layers - 1):
            self.layers.append(
                WeightedSAGEConv(n_hidden, n_hidden, n_hidden, dropout,
                                 activation))
        self.layers.append(
            WeightedSAGEConv(n_hidden, n_hidden, n_classes, dropout,
                             activation))

    def forward(self, graph, x):
        h = x
        for layer in self.layers:
            h_dst = h[:graph.number_of_nodes('DST/' + graph.ntypes[0])]
            h = layer(graph, (h, h_dst), graph.edata['weights'])
        return h

    def inference(self, g, x, batch_size, device):
        pass

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield
