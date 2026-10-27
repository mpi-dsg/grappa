from t10n.xborder import XBorder
from t10n.xborder import xb_group_by_partition as xb_group_by_partition_opt

from t10n.util import timing

import torch

n_part = 16
#n_per_part = 1000000
n_per_part = 5000
prefix_sum = [n_per_part * i for i in range(n_part + 1)]

nodes_1 = torch.randint(n_per_part, 16 * n_per_part, (1 * n_per_part,))
nodes_9 = torch.randint(0, n_per_part, (9 * n_per_part,))
seed_nodes = torch.concat((nodes_1, nodes_9))


@timing
def split_to_partitions(seed_nodes, prefix_sum):
    seed_nodes_list = []
    for i in range(len(prefix_sum) - 1):
        mask = seed_nodes >= prefix_sum[i]
        mask = mask & (seed_nodes < prefix_sum[i + 1])
        nodes = seed_nodes[mask]
        seed_nodes_list.append(nodes)
    return seed_nodes_list


xb = XBorder(0, n_part, prefix_sum)
xb.set_nbr_idx(2)


@timing
def xb_group_by_partition(seed_nodes, prefix_sum):
    xb_group_by_partition_opt(xb, seed_nodes, n_part)


@timing
def dumm(val):
    return val


for i in range(10):
    dumm(i)

for i in range(10):
    ret = split_to_partitions(seed_nodes, prefix_sum)
    # 0.067429 seconds, use 16 threads, n_per_part=1000k
    # 0.002390 seconds, use 16 threads, n_per_part=5k

for i in range(10):
    ret = xb_group_by_partition(seed_nodes, prefix_sum)
    # 0.232397 seconds, use 1 threads
    # 0.084172 seconds, use 8 threads
    # 0.021680 seconds, use 16 threads, n_per_part=1000000
    # 0.000089 seconds, use 16 threads, n_per_part=5k
