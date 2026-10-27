for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no  t10n_cu118_w$i "apt update && apt install -y libunwind8 && apt install -y libunwind-dev"; done

pip uninstall -y t10n ; pip uninstall -y dgl ; pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html ; pip install numa ; pip install /workspace/t10n/wheels/t10n-0.0.0-cp310-cp310-linux_x86_64.whl

for i in `seq 0 15`; do ssh -o StrictHostKeyChecking=no  t10n_cu118_w$i "pip uninstall -y t10n ; pip uninstall -y dgl ; pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html ; pip install numa ; pip install /workspace/t10n/wheels/t10n-0.0.0-cp310-cp310-linux_x86_64.whl"; done
