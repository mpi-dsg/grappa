import numpy as np

import dgl
import torch as th
import torch.nn.functional as F

import tqdm


def inference_for_default_sampling(model, g, x, batch_size, device):
    """
    Inference with the GraphSAGE model on full neighbors (i.e. without
    neighbor sampling).

    g : the entire graph.
    x : the input of entire node set.

    Distributed layer-wise inference.
    """
    # During inference with sampling, multi-layer blocks are very
    # inefficient because lots of computations in the first few layers
    # are repeated. Therefore, we compute the representation of all nodes
    # layer by layer.  The nodes on each layer are of course splitted in
    # batches.
    # TODO: can we standardize this?
    force_even_flg = True
    nodes = dgl.distributed.node_split(
        np.arange(g.num_nodes()),
        g.get_partition_book(),
        force_even=force_even_flg,
    )
    if model.name == 'gat':
        infer_hidden_dim = model.n_hidden * model.n_heads
    else:
        infer_hidden_dim = model.n_hidden

    policy = None
    y = dgl.distributed.DistTensor(
        (g.num_nodes(), infer_hidden_dim),
        th.float32,
        "h",
        persistent=True,
        part_policy=policy,
    )
    for i, layer in enumerate(model.layers):
        if i == len(model.layers) - 1:
            y = dgl.distributed.DistTensor(
                (g.num_nodes(), model.n_classes),
                th.float32,
                "h_last",
                persistent=True,
                part_policy=policy,
            )
        print(
            f"g.rank()={g.rank()}, |V|={g.num_nodes()}, eval batch size: {batch_size}"
        )

        sampler = dgl.dataloading.NeighborSampler([-1],)
        dataloader = dgl.dataloading.DistNodeDataLoader(
            g,
            nodes,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        if model.name == 'gat':
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(model.layers) - 1:
                    h = h.flatten(1)
                    h = model.activation(h)
                    h = model.dropout(h)
                else:
                    h = h.mean(1)

                y[output_nodes] = h.cpu()
                h = None
                h_dst = None
        elif model.name == 'gin':
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(model.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.cpu()
                h = None
        else:
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(model.layers) - 1:
                    h = model.activation(h)
                    h = model.dropout(h)

                y[output_nodes] = h.cpu()
                h = None

        x = y
        g.barrier()
    x = None
    return y


# x feature of g
def inference(model, g, x, batch_size, device):
    return inference_for_default_sampling(model, g, x, batch_size, device)
