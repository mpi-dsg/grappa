cd /workspace
rm -r dgl_dsg
git config --global --add safe.directory '*'

# downloading
cd /workspace/
git clone -b v1.0.1/ds4gnn --recurse-submodules https://github.com/mpi-dsg/dgl_dsg.git
cd dgl_dsg
git submodule update --init --recursive

cd /workspace/dgl_dsg && mkdir -p build && rm -r build
mkdir build && cd build
cmake -DUSE_CUDA=ON ..
#
## apt update && apt install -y gcc-9 g++-9
#cmake -DUSE_CUDA=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CPP_COMPILER=/usr/bin/g++-9 ..
#
make -j8

# install or build wheels
#cd /workspace/dgl_dsg/python
#python3 setup.py install
#pip wheel . -w wheels
#python setup.py sdist bdist_wheel
#cp wheels/dgl-1.0.1-cp*-cp*-linux_x86_64.whl /data/tmp/


#for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no t10n_cu118_w$i nvidia-smi; done
#for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no t10n_cu118_w$i pip install dgl==1.0.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html; done
#for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no t10n_cu118_w$i "cd /workspace/t10n/; pip install ."; done
#for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no t10n_cu118_w$i "cd /workspace/dgl_dsg/python && python3 setup.py install"; done
