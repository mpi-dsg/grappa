# Grappa

Grappa is a distributed training system for graph neural networks that enables efficient **gradient-only communication** for scalable cross-partition training.

## Note on Authorship

This repository reflects the state used for internal development and submission.
The underlying implementation of the Grappa system originates from [Chongyang Xu](https://github.com/chongyang-xu).
Subsequent commits were performed to meet submission and reproducibility requirements.

The actively maintained version is available at: https://github.com/chongyang-xu/grappa

## Citation

```bibtex
@inproceedings{xu2025grappa,
  title={Grappa: Gradient-Only Communication for Scalable Graph Neural Network Training},
  author={Xu, Chongyang and Siebenbrunner, Christoph and Bindschaedler, Laurent},
  booktitle={Proceedings of the VLDB Endowment},
  year={2025},
  note={Under submission}
}
```

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

## Paper Figures Reproduction

The experiments below generate the figures and tables in the paper. Replace `[DATASET]` with `ogbpr` (OGBN-Products), `ogbar` (OGBN-arXiv), `reddit`, `ogbpa` (OGBN-Papers100M), or RMAT scales.

### Figure: Sampling-based Training Pipeline (fig:training_pipeline)
Conceptual diagram showing conventional sampling-based data-parallel GNN training with cross-partition neighbor fetches. See the paper.

### Figure: Training with Halo Nodes (fig:training-example)
Conceptual diagram showing partition structure with halo nodes (dashed lines). See the paper.

### Figure: Efficient Repartitioning via Sweeping Chunks (fig:sweep_chunk)
Conceptual diagram showing Grappa's dynamic repartitioning mechanism where workers combine base chunks with variable chunks across super-epochs. See the paper.

### Figure: Scalability with Number of Partitions (fig:scalability-partitions)
Epoch time vs. number of partitions for Sage-3, comparing Grappa against DGL with METIS.
```bash
cd experiments/grappa
# Vary partition count from 1 to 64 (one partition per GPU)
for partitions in 1 2 4 8 16 32 64; do
  python t10nrun.py --dataset ogbpr --partitions $partitions --num_gpus $partitions --model sage --layers 3
done

# Also run on other datasets
for dataset in ogbar reddit ogbpa; do
  for partitions in 1 4 16 64; do
    python t10nrun.py --dataset $dataset --partitions $partitions --num_gpus $partitions
  done
done
```

### Figure: Scalability with Graph Size (fig:scalability-size)
Epoch time for Sage-3 with 16 partitions as RMAT graph size scales from RMAT-26 to RMAT-30.
```bash
cd experiments/grappa
# Scale RMAT graph size
for scale in 26 27 28 29 30; do
  python t10nrun.py --dataset rmat$scale --partitions 16 --num_gpus 16 --model sage --layers 3
done
```

### Figure: Training Convergence (fig:training-convergence)
Training accuracy over time for GCN-2 with 16 partitions on OGBN-Products.
```bash
cd experiments/grappa
# Run for full 500 epochs to observe convergence
python t10nrun.py --dataset ogbpr --partitions 16 --num_gpus 16 --model gcn --layers 2 --num_epochs 500

# Compare with DGL baseline
cd ../baselines/dgl
python run.py --dataset ogbpr --partitions 16 --num_gpus 16 --model gcn --layers 2 --num_epochs 500
```

### Table: Per-partition Neighbor Traffic (tab:overhead)
Cross-partition neighbor traffic volumes showing why conventional training becomes bottlenecked.
```bash
# Measure cross-partition traffic with DGL
cd experiments/baselines/dgl
for partitions in 2 4 8; do
  python run.py --dataset ogbpr --partitions $partitions --measure_traffic
done
```

### Table: Datasets (tab:datasets)
Dataset statistics including vertices, edges, and feature dimensions. See `t10n/dataset/meta.py`.

### Table: Overall Performance Comparison (tab:overall-comparison)
Epoch time comparison between Grappa and DGL across different models and datasets.
```bash
cd experiments/grappa
# Run Grappa on all datasets with different models
for dataset in ogbar reddit ogbpr; do
  for model in gcn sage gat; do
    for layers in 2 3 4; do
      python t10nrun.py --dataset $dataset --model $model --layers $layers --partitions 16 --num_gpus 16
    done
  done
done

# Compare with DGL baselines
cd ../baselines/dgl
for dataset in ogbar reddit ogbpr; do
  for model in gcn sage gat; do
    for layers in 2 3 4; do
      python run.py --dataset $dataset --model $model --layers $layers --partitions 16
    done
  done
done
```

### Table: Accuracy Results (tab:accuracy)
Average test accuracy across model depths (2, 3, 4 layers).
```bash
# Results are obtained from the overall comparison experiments above
# Extract test accuracy from experiment logs
```

### Table: Cluster-GCN Comparison (tab:cluster-gcn)
Epoch time and test accuracy comparison with Cluster-GCN on GCN-2 for OGBN-Products.
```bash
# Cluster-GCN baseline
cd experiments/baselines/cluster_gcn
./run.sh --dataset ogbpr --model gcn --layers 2

# Compare with Grappa
cd ../../grappa
python t10nrun.py --dataset ogbpr --model gcn --layers 2 --partitions 16 --num_gpus 16
```

### Table: MGG Single-Node Performance (tab:mgg)
Single-node full-graph training comparison with MGG on NVLink-connected H100 GPUs.
```bash
# MGG baseline - requires single H100 node with NVLink
cd experiments/baselines/mgg
./run.sh --dataset ogbpr

# Compare with Grappa on same hardware
cd ../../grappa
python t10nrun.py --dataset ogbpr --partitions 8 --num_gpus 8 --machine single_node
```

### Table: Ablation Study (tab:ablation)
Ablation study comparing Grappa against variants: UW (no weighted aggregation), FP (fixed partitions), 50S (50% remote sampling).
```bash
cd experiments/grappa
# Full Grappa
python t10nrun.py --dataset reddit --partitions 16 --num_gpus 16 --num_epochs 500

# Grappa-UW (no weighted aggregation)
python t10nrun.py --dataset reddit --partitions 16 --num_gpus 16 --no_weighted --num_epochs 500

# Grappa-FP (fixed partitions, no repartitioning)
python t10nrun.py --dataset reddit --partitions 16 --num_gpus 16 --fixed_partitions --num_epochs 500

# Grappa-50S (50% remote sampling)
python t10nrun.py --dataset reddit --partitions 16 --num_gpus 16 --remote_sample 0.5 --num_epochs 500
```

### Table: Partitioning Overhead (tab:partition-overhead)
Partitioning time comparison between Grappa (random chunking) and DGL with METIS.
```bash
# Grappa partitioning time
cd experiments/grappa
for partitions in 4 16 64; do
  python t10nrun.py --dataset ogbpa --partitions $partitions --partition_only
done

# DGL METIS partitioning time
cd ../baselines/dgl
for partitions in 4 16 64; do
  python run.py --dataset ogbpa --partitions $partitions --partitionmetis --partition_only
done
```

### Table: Repartition Overhead (tab:partition-switching)
Time spent training vs. switching partitions during repartitioning.
```bash
cd experiments/grappa
# Measure switching time overhead
python t10nrun.py --dataset ogbpr --partitions 4 --num_gpus 4 --measure_switching
python t10nrun.py --dataset ogbpr --partitions 16 --num_gpus 16 --measure_switching
python t10nrun.py --dataset ogbpr --partitions 64 --num_gpus 64 --measure_switching
```

### Table: Partition Storage Overhead (tab:partition-storage)
Size of each partition and total storage with Grappa's partitioning scheme.
```bash
cd experiments/grappa
python t10nrun.py --dataset ogbpr --partitions 4 --measure_storage
# Repeat for other datasets in tab:datasets
```

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

MIT License - see [LICENSE](LICENSE)

## Contact

For questions about Grappa, please open an issue on GitHub or contact the authors.
