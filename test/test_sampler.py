import numpy as np

from t10n._C.graph import Batch
from t10n._C.graph import HostGraph
from t10n._C.graph import GpuGraph

from t10n.sampler import HostBatchSampler
from t10n.dgl_dsg.compliant import to_dgl_batch

import torch as th

u = np.array([1, 2, 3, 4, 5, 100, 101, 101], dtype=np.int64)
v = np.array([10, 5, 200, 24, 200, 120, 200, 300], dtype=np.int64)
seeds = np.array([10, 200], dtype=np.int64)
bs = 1
bn = (seeds.shape[0] + bs - 1) // bs

hg = HostGraph(u, v)
gg = GpuGraph(hg)

hbs = HostBatchSampler(hg,
                       th.tensor(seeds), [2, 5],
                       repeated=False,
                       batch_size=bs,
                       shuffle=True)

exit(0)
REPEAT = 10000
for idx in range(REPEAT):
    print(idx, '-' * 50)
    for i, (batch_input, batch_output, blocks) in enumerate(hbs):
        print(f"{i} = {batch_input}")
        print(f"{i} = {batch_output}")
        print(f"{i} = {blocks}")

for idx in range(REPEAT):
    hbs._reinit_from(hbs.target_nodes, 1)
    for i in range(2):
        all_layers, all_inputs = hbs._next_batch()
        print(all_layers)
        print(all_inputs)
        output_nodes = hbs._cur_batch_opt_nodes()
        hbs.batch_idx += 1
        to_dgl_batch(output_nodes, all_layers, all_inputs)
