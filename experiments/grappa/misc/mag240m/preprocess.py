import argparse
import os

import dgl
import dgl.function as fn

import numpy as np
import ogb
import torch
import tqdm
from ogb.lsc import MAG240MDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--rootdir",
    type=str,
    default=".",
    help="Directory to download the OGB dataset.",
)
parser.add_argument(
    "--author-output-path", type=str, help="Path to store the author features."
)
parser.add_argument(
    "--inst-output-path",
    type=str,
    help="Path to store the institution features.",
)
parser.add_argument(
    "--graph-output-path", type=str, help="Path to store the graph."
)
parser.add_argument(
    "--graph-format",
    type=str,
    default="csc",
    help="Graph format (coo, csr or csc).",
)
parser.add_argument(
    "--graph-as-homogeneous",
    action="store_true",
    help="Store the graph as DGL homogeneous graph.",
)
parser.add_argument(
    "--full-output-path",
    type=str,
    help="Path to store features of all nodes.  Effective only when graph is homogeneous.",
)
args = parser.parse_args()

print("Building graph")
dataset = MAG240MDataset(root=args.rootdir)

ei_writes = dataset.edge_index("author", "writes", "paper")
ei_cites = dataset.edge_index("paper", "paper")
ei_affiliated = dataset.edge_index("author", "institution")

print(len(ei_writes), len(ei_writes[0]))
print(len(ei_cites), len(ei_cites[0]))
print(len(ei_affiliated), len(ei_affiliated[0]))

exit(0)

# We sort the nodes starting with the papers, then the authors, then the institutions.
author_offset = 0
inst_offset = author_offset + dataset.num_authors
paper_offset = inst_offset + dataset.num_institutions
print(f"paper_offset={paper_offset}")
full_node_n = dataset.num_authors + dataset.num_institutions + dataset.num_papers
print("full_node_n: ", full_node_n)
print("num_paper_feat: ", dataset.num_paper_features)
print("paper_label.shape: ", dataset.paper_label.shape)
print("paper_label.dtype: ", dataset.paper_label.dtype)
exit(0)

full_label = torch.full((full_node_n,), 0.0, dtype=torch.float32)
full_label[paper_offset:] = torch.from_numpy(dataset.paper_label).to(torch.float32)

dt = full_label.numpy().astype(np.float32)
np.save("$DATA_PATH/dgl/mag240m/full_label", dt)

train_idx = dataset.get_idx_split('train')
valid_idx = dataset.get_idx_split("valid")
test_idx = dataset.get_idx_split("test-dev")
train_idx = torch.from_numpy(train_idx) + paper_offset
valid_idx = torch.from_numpy(valid_idx) + paper_offset
test_idx = torch.from_numpy(test_idx) + paper_offset

print(train_idx.shape)
print(train_idx.dtype)
print(valid_idx.shape)
print(test_idx.shape)

full_train_mask = torch.full((full_node_n,), False)
full_valid_mask = torch.full((full_node_n,), False)
full_test_mask = torch.full((full_node_n,), False)
full_train_mask[train_idx] = True
full_valid_mask[valid_idx] = True
full_test_mask[test_idx] = True

dt = full_train_mask.numpy().astype(np.int8)
np.save("$DATA_PATH/dgl/mag240m/full_train_mask", dt)

dt = full_valid_mask.numpy().astype(np.int8)
np.save("$DATA_PATH/dgl/mag240m/full_valid_mask", dt)

dt = full_test_mask.numpy().astype(np.int8)
np.save("$DATA_PATH/dgl/mag240m/full_test_mask", dt)

print(full_train_mask)
exit(0)

g = dgl.heterograph(
    {
        ("author", "write", "paper"): (ei_writes[0], ei_writes[1]),
        ("paper", "write-by", "author"): (ei_writes[1], ei_writes[0]),
        ("author", "affiliate-with", "institution"): (
            ei_affiliated[0],
            ei_affiliated[1],
        ),
        ("institution", "affiliate", "author"): (
            ei_affiliated[1],
            ei_affiliated[0],
        ),
        ("paper", "cite", "paper"): (
            np.concatenate([ei_cites[0], ei_cites[1]]),
            np.concatenate([ei_cites[1], ei_cites[0]]),
        ),
    }
)

paper_feat = dataset.paper_feat
author_feat = np.memmap(
    args.author_output_path,
    mode="w+",
    dtype="float16",
    shape=(dataset.num_authors, dataset.num_paper_features),
)
inst_feat = np.memmap(
    args.inst_output_path,
    mode="w+",
    dtype="float16",
    shape=(dataset.num_institutions, dataset.num_paper_features),
)

# Iteratively process author features along the feature dimension.
BLOCK_COLS = 16
with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
    for start in tq:
        tq.set_postfix_str("Reading paper features...")
        g.nodes["paper"].data["x"] = torch.FloatTensor(
            paper_feat[:, start : start + BLOCK_COLS].astype("float32")
        )
        # Compute author features...
        tq.set_postfix_str("Computing author features...")
        g.update_all(fn.copy_u("x", "m"), fn.mean("m", "x"), etype="write-by")
        # Then institution features...
        tq.set_postfix_str("Computing institution features...")
        g.update_all(
            fn.copy_u("x", "m"), fn.mean("m", "x"), etype="affiliate-with"
        )
        tq.set_postfix_str("Writing author features...")
        author_feat[:, start : start + BLOCK_COLS] = (
            g.nodes["author"].data["x"].numpy().astype("float16")
        )
        tq.set_postfix_str("Writing institution features...")
        inst_feat[:, start : start + BLOCK_COLS] = (
            g.nodes["institution"].data["x"].numpy().astype("float16")
        )
        del g.nodes["paper"].data["x"]
        del g.nodes["author"].data["x"]
        del g.nodes["institution"].data["x"]
author_feat.flush()
inst_feat.flush()

# Convert to homogeneous if needed.  (The RGAT baseline needs homogeneous graph)
if args.graph_as_homogeneous:
    # Process graph
    g = dgl.to_homogeneous(g)
    # DGL ensures that nodes with the same type are put together with the order preserved.
    # DGL also ensures that the node types are sorted in ascending order.
    assert torch.equal(
        g.ndata[dgl.NTYPE],
        torch.cat(
            [
                torch.full((dataset.num_authors,), 0),
                torch.full((dataset.num_institutions,), 1),
                torch.full((dataset.num_papers,), 2),
            ]
        ),
    )
    assert torch.equal(
        g.ndata[dgl.NID],
        torch.cat(
            [
                torch.arange(dataset.num_authors),
                torch.arange(dataset.num_institutions),
                torch.arange(dataset.num_papers),
            ]
        ),
    )
    g.edata["etype"] = g.edata[dgl.ETYPE].byte()
    del g.edata[dgl.ETYPE]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]

    # Process feature
    full_feat = np.memmap(
        args.full_output_path,
        mode="w+",
        dtype="float16",
        shape=(
            dataset.num_authors + dataset.num_institutions + dataset.num_papers,
            dataset.num_paper_features,
        ),
    )
    BLOCK_ROWS = 100000
    for start in tqdm.trange(0, dataset.num_authors, BLOCK_ROWS):
        end = min(dataset.num_authors, start + BLOCK_ROWS)
        full_feat[author_offset + start : author_offset + end] = author_feat[
            start:end
        ]
    for start in tqdm.trange(0, dataset.num_institutions, BLOCK_ROWS):
        end = min(dataset.num_institutions, start + BLOCK_ROWS)
        full_feat[inst_offset + start : inst_offset + end] = inst_feat[
            start:end
        ]
    for start in tqdm.trange(0, dataset.num_papers, BLOCK_ROWS):
        end = min(dataset.num_papers, start + BLOCK_ROWS)
        full_feat[paper_offset + start : paper_offset + end] = paper_feat[
            start:end
        ]

# Convert the graph to the given format and save.  (The RGAT baseline needs CSC graph)
g = g.formats(args.graph_format)
dgl.save_graphs(args.graph_output_path, g)
