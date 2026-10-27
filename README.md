# Grappa

Grappa is a distributed training system for graph neural networks that enables efficient **gradient-only communication** for scalable cross-partition training.

This code supports the paper *Grappa: Gradient-Only Communication for Scalable Graph Neural Network Training* ([arXiv:2602.01872](https://arxiv.org/abs/2602.01872)).

Status: preprint on arXiv. Repository: https://github.com/mpi-dsg/grappa.

## Note on Authorship

This repository reflects the state used for internal development and submission.
The underlying implementation of the Grappa system originates from [Chongyang Xu](https://github.com/chongyang-xu).
Subsequent commits were performed to meet submission and reproducibility requirements.

The actively maintained version is available at: https://github.com/chongyang-xu/grappa

## Overview

Grappa addresses the scalability challenge of training GNNs on large-scale graphs that do not fit in a single machine's memory. The key innovation is **gradient-only communication**: during each iteration, partitions train in isolation and exchange only gradients for the global update. To recover accuracy lost to isolation, Grappa (i) periodically **repartitions** to expose new neighborhoods and (ii) applies lightweight **coverage-corrected gradient aggregation**.

### Key Features

- **Gradient-only communication**: Partitions train in isolation, exchange only gradients (no feature/activation traffic)
- **Periodic repartitioning**: Exposes new neighborhoods to recover accuracy without full communication
- **Coverage-corrected gradients**: Asymptotically unbiased gradient estimator for isolated training
- **Model-agnostic**: Works with any GNN architecture (GCN, GraphSAGE, GAT, GIN, PinSAGE)
- **Full-graph and mini-batch**: Supports both training modes
- **Commodity hardware**: No high-bandwidth interconnects required

## System Components

```
grappa/
├── csrc/           # C++ core (sampling engine, cross-partition communication)
├── t10n/           # Python module (dataloader, dataset handling)
├── scripts/        # Installation and utility scripts
├── test/           # Unit tests
└── experiments/
    ├── grappa/     # Grappa training scripts
    └── baselines/  # Baseline systems (DGL, BNS-GCN, Cluster-GCN, etc.)
```

## Installation

### Prerequisites

- CUDA 11.8+
- PyTorch 1.12+
- DGL 1.1+
- CMake 3.18+
- NCCL (for multi-GPU)

### Build from source

```bash
# Install Python dependencies
pip install torch dgl

# Build Grappa extension
python setup.py install
```

See `scripts/t10n_install.sh` for detailed installation instructions.

## Running Experiments

### Quick Start

```bash
# Single-node, 4-GPU training on OGBN-Products
cd experiments/grappa
./run.sh

# Multi-node training (see hosts_*, slurm_*.sh templates)
```

### Baselines Comparison

The following baselines are included for reproducibility:

| System | Location | Description |
|--------|----------|-------------|
| **Grappa** | `experiments/grappa/` | Our system (cross-partition sampling) |
| DGL | `experiments/baselines/dgl/` | DGL distributed training |
| BNS-GCN | `experiments/baselines/bns_gcn/` | Neighbor sampling with caching |
| Cluster-GCN | `experiments/baselines/cluster_gcn/` | Cluster-based sampling |
| Neutron Star | `experiments/baselines/neutron_star/` | Hybrid partitioning |
| ROC | `experiments/baselines/roc/` | Rule-based partitioning |

### Experiment Configuration

Each system has its own run script:

```bash
# Grappa (our system)
cd experiments/grappa && python t10nrun.py [args]

# DGL baseline
cd experiments/baselines/dgl && python run.py [args]

# BNS-GCN baseline
cd experiments/baselines/bns_gcn/t10n_baseline && python run.py [args]
```

## Reproducing the paper

The experiments ran on GPU clusters (an 8-machine V100 cluster, A100 and H100 nodes) with the datasets under `/data/t10n/`. They cannot run on a laptop.

### How a run is launched

1. Build and install the `t10n` extension on every node (`python setup.py install`; `scripts/t10n_install.sh` shows the cluster install).
2. Partition the dataset: `python3 t10nrun.py --host_file <hosts> --tag_id <id> --task partition`, run from `experiments/grappa/`. This expands `partition.sh.template`, which calls `build_chunks_dist.py`. That script is not in this repository (see "Known gaps").
3. Train: set the sweep in `start_train()` in `experiments/grappa/t10nrun.py` (dataset list, model list, layer and fan-out pairs, `NUM_PART`, `N_EPOCH`, repeat count), then run `python3 t10nrun.py --host_file <hosts> --tag_id <id> --task train --train_num_proc <gpus per node>`. The launcher expands `train.sh.template` into a `torch.distributed.run` call of `train/t10n_train.py` on each host and writes one log per host to `experiments/grappa/log/<tag_id>_<host index>.txt`.

`t10nrun.py` accepts only `--host_file`, `--tag_id`, `--task`, `--train_script`, `--train_num_proc`, `--partition_script`, `--partition_num_proc`, and `--inline_cmd`. Per-run settings are edited in the file. Host files (`hosts*`) list one node per line with its GPU count.

### Paper figures and tables

| Paper item | Code | Notes |
|---|---|---|
| `fig:training_pipeline`, `fig:training-example`, `fig:sweep_chunk` | none | Conceptual diagrams. |
| `tab:overhead` (per-partition neighbor traffic) | `experiments/baselines/dgl/comm_volume.py` | Measures cross-partition traffic under DGL. |
| `tab:datasets` | `t10n/dataset/meta.py` | Dataset statistics. |
| `tab:overall-comparison`, `tab:accuracy` | Grappa: `experiments/grappa/t10nrun.py` with `train/t10n_train.py`. DGL: `experiments/baselines/dgl/run.py` with `experiments/baselines/dgl/train/` | Epoch time and test accuracy are read from the run logs. |
| `fig:scalability-partitions` | same as above, varying `NUM_PART` and the host file | Plot script not in this repository. |
| `fig:training-convergence` | same as above, `N_EPOCH = 500` | Plot script not in this repository. |
| `tab:minimally-biased`, `tab:ablation` | `experiments/grappa/t10nrun.py`; estimator and counters in `t10n/grad.py` and `t10n/dgl_dsg/sampler.py`, repartitioning interval `--repart_every` in `train/t10n_train.py` | The ablation variants are not exposed as flags; they were produced by changing the code. |
| `tab:cluster-gcn` | `experiments/baselines/cluster_gcn/t10n_baseline/run_*.sh` | |
| `tab:mgg` | Grappa full-graph mode: `train.sh.template.fullg`, `train/t10n_fullgraph_train.py`. MGG: `experiments/baselines/mgg/setup_cu118.sh` | Only the MGG setup script is included. |
| `tab:overheads` (partitioning and switching time) | `experiments/grappa/t10nrun.py` (partition and train tasks); DGL METIS partitioning in `experiments/baselines/dgl/` | |
| `tab:partition-storage` | `experiments/grappa/t10nrun_partition_size.py`, `experiments/grappa/run_partition_size.sh` | |

Other baselines: BNS-GCN (`experiments/baselines/bns_gcn/`), NeutronStar (`experiments/baselines/neutron_star/`), and ROC (`experiments/baselines/roc/`) are applied as patches to the upstream projects plus run scripts under `t10n_baseline/`.

### Known gaps

- `build_chunks_dist.py`, called by `partition.sh.template`, is not in this repository.
- There are no log-parsing or plotting scripts. Figures and tables were produced from the run logs outside this repository.
- The experiment logs (over 22 GB) are not included.

## Datasets

Grappa supports OGB datasets and custom large-scale graphs:

| Dataset | Nodes | Edges | Description |
|---------|-------|-------|-------------|
| OGBN-Products | 2.4M | 61M | Product recommendation |
| OGBN-arXiv | 170K | 1.1M | Citation network |
| Reddit | 232K | 11M | Discussion threads |
| Papers100M | 111M | 1.6B | Academic citations |

See `t10n/dataset/meta.py` for the complete list and dataset processing scripts.

## Project Structure

### Core Implementation (`csrc/`)

- `base/`: Utilities (chunking, GPU allocator, logging)
- `graph/`: Graph data structures and sampling kernels
- `python/`: Python bindings via pybind11

### Python Module (`t10n/`)

- `dataloader.py`: Main distributed dataloader
- `xborder.py`: Cross-partition border handling (key innovation)
- `sampler.py`: Sampling interface
- `dataset/`: Dataset loading and partitioning
- `dgl_dsg/`: DGL integration layer

## License

MIT License. See [LICENSE](LICENSE). The baseline patches under `experiments/baselines/` apply to upstream projects that keep their own licenses.

## Contact

For questions about Grappa, please open an issue on GitHub or contact the authors.

## Citation

```bibtex
@misc{xu2026grappa,
  title         = {Grappa: Gradient-Only Communication for Scalable Graph Neural Network Training},
  author        = {Xu, Chongyang and Siebenbrunner, Christoph and Bindschaedler, Laurent},
  year          = {2026},
  eprint        = {2602.01872},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```
