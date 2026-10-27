import numpy as np

from t10n._C.xborder import XBorder

import torch as th
import dgl.backend as FF

ids = np.array([i + 200 * (i % 2) for i in range(10)], dtype=np.int64)
prefix_sum = np.array([0, 200, 203, 207, 1000], dtype=np.int64)

xb = XBorder(0, 4, prefix_sum)
mask = th.zeros([4, ids.shape[0]], dtype=th.int8)
print("xb", xb)
xb.set_nbr_idx(2)

print("ids", ids)
ret = xb.group_by_partition(ids, mask.numpy())
print("type(ret)", type(ret))
print("ret", ret)
for e in ret:
    print("\ttype(e)", type(e))
    print("\te", e)

xb.ad_hoc_clear()

ids = np.array([300 + i for i in range(10)], dtype=np.int64)
xb.ad_hoc_build_id_mapping(3, ids)

batch_size = 4
feat_dim = 5

batch_feat = th.zeros(batch_size, feat_dim, dtype=th.float32)

feat_n = [  # required for every partition
    th.zeros(1, feat_dim, dtype=th.float32),  # part 0, use 1 for easier print 
    th.ones(1, feat_dim, dtype=th.float32),  # part 1
    th.zeros(1, feat_dim, dtype=th.float32) + 2.0,  # part 2 
    th.zeros(10, feat_dim, dtype=th.float32) + 3.0,  # part 3 
]

ids = np.array([0, 200, 301, 302], dtype=np.int64)
ids = FF.zerocopy_from_numpy(ids)

feat_n_np = [ele.numpy() for ele in feat_n]
print("batch_feat", batch_feat)
xb.ad_hoc_fill_batch_feat(batch_feat.numpy(), feat_n_np, ids)
print("batch_feat", batch_feat)
