# source
downloaded from https://github.com/google-research/google-research/tree/master/cluster_gcn
commit : 1d49f2c

follow version of requirements
except install: pip install tensorflow==2.13.*

dataset:
https://snap.stanford.edu/graphsage/#datasets


# set up
```bash
#  metis build issue
#  https://github.com/KarypisLab/METIS/issues/83
# 
git clone https://github.com/KarypisLab/GKlib.git
cd GKlib/
make config CONFIG_FLAGS='-D BUILD_SHARED_LIBS=ON -D CMAKE_INSTALL_PREFIX=/DS/dsg-graphs/work/ds4gnn/WS/custom_install/'
make
make install

cd ../

git clone https://github.com/KarypisLab/METIS.git
cd METIS
sed -i '/^CONFIG_FLAGS \?= / s,$, -DCMAKE_BUILD_RPATH=${HOME}/.local/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON,' Makefile
make config shared=1 cc=gcc prefix=/home/cxu/ws/gnn/WS/custom_install/  gklib_path=/home/cxu/ws/gnn/WS/custom_install/
make
make install
```
## code fix if use networkx 1.11
```python
change networkx import fractions ==> import math
```

```python
metis.py

def add_star(G, nodes):
    center = nodes[0]
        edges = [(center, node) for node in nodes[1:]]
	    G.add_edges_from(edges)

def add_path(G, nodes):
    # Create edges between consecutive nodes in the list
        edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes) - 1)]
	    G.add_edges_from(edges)
```

## code fix if use networkx 3.4
