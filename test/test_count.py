import torch as th

import dgl
import dgl.backend as FF

u = th.tensor([2, 3, 4, 200, 201], dtype=th.int64)
v = th.tensor([1, 1, 1, 103, 104], dtype=th.int64)
g = dgl.graph((u, v))

seed_nodes = th.tensor([1], dtype=th.int64)
fanout = 3

gideg = FF.zerocopy_to_dgl_ndarray(
    5 + th.zeros(seed_nodes.shape[0], dtype=th.int64))
#print(f"rank:{self.rank}, idx:{idx}, gideg:{gideg}")
lideg = FF.zerocopy_to_dgl_ndarray(th.zeros(seed_nodes.shape[0],
                                            dtype=th.int64))
#print(f"rank:{self.rank}, idx:{idx}, lideg:{lideg}")
resample_count = FF.zerocopy_to_dgl_ndarray(th.zeros(1, dtype=th.int64))
#print(f"rank:{self.rank}, idx:{idx}, resample_count:{resample_count}")

gideg = None
lideg = None
resample_count = None

frontier = g.sample_neighbors(
    seed_nodes,
    fanout,
    edge_dir='in',
    prob=None,
    replace=True,  #self.replace,
    output_device=th.device("cuda:0"),
    exclude_edges=None,
    degs=(gideg, lideg),
    resample_count=resample_count)

if resample_count != None:
    tmp_tensor = FF.zerocopy_from_dgl_ndarray(resample_count)
    precise_count = tmp_tensor.item()
    print(f"precise_count={precise_count}")
