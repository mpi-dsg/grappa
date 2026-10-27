pip wheel . -w wheels/
pip install pyyaml
pip uninstall -y t10n; pip install /workspace/t10n/wheels/t10n-0.0.0-cp310-cp310-linux_x86_64.whl
pip uninstall -y dgl; pip install /workspace/dgl_dsg/python/wheels/dgl-1.0.1-cp310-cp310-linux_x86_64.whl

for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no  t10n_cu118_w$i "pip install pyyaml"; done
for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no  t10n_cu118_w$i "pip uninstall -y t10n ; pip install /workspace/t10n/wheels/t10n-0.0.0-cp310-cp310-linux_x86_64.whl"; done
for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no  t10n_cu118_w$i "pip uninstall -y dgl ; pip install /workspace/dgl_dsg/python/wheels/dgl-1.0.1-cp310-cp310-linux_x86_64.whl"; done
