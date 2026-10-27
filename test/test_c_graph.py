import numpy as np

from t10n._C.graph import Batch
from t10n._C.graph import HostGraph
from t10n._C.graph import GpuGraph
from t10n._C.graph import HostSampler
from t10n._C.graph import HostNeighborSampler
from t10n._C.graph import GpuSampler
from t10n._C.graph import py_test_batch


def pb(b: Batch):
    print(b.get_layers())
    print(b.get_input_nodes())


b: Batch = py_test_batch()
pb(b)

u = np.array([1, 2, 3, 4, 5, 100, 101, 101], dtype=np.int64)
v = np.array([10, 5, 200, 24, 200, 120, 200, 300], dtype=np.int64)
seeds = np.array([10, 200], dtype=np.int64)
bs = 1
bn = (seeds.shape[0] + bs - 1) // bs
print(f"bn={bn}")

hg = HostGraph(u, v)
gg = GpuGraph(hg)

hs = HostSampler(hg, [2, 5], True)
REPEAT = 5
for idx in range(REPEAT):
    print(idx, '-' * 50)
    hs.c_reinit_from(seeds, bs)
    for i in range(bn):
        b: Batch = hs.c_next_batch()
        pb(b)

for idx in range(REPEAT):
    print(idx, '+' * 50)
    hs.c_reinit_from(seeds, bs)
    layers = hs.c_next_batch_py()
    print(layers)
    layers = hs.c_next_batch_py()
    print(layers)

gs = GpuSampler([20, 15])
gs.c_reinit_from()
gs.c_next_batch()

hbs = HostNeighborSampler(hg, False)
for idx in range(REPEAT):
    for f in reversed([np.iinfo(np.int64).max] * 2):
        layers = hbs.c_next_batch_py(seeds, f)
        print(layers)
