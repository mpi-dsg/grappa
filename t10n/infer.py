import gc

from t10n.dataloader import IsolatedDataloader
from t10n.util import pprint
#from t10n.util import pprint, pyml_handle

import torch as th


def ifdbg(any, *args, **kwargs):
    pprint(0, f"rank{any.rank:02d}:", *args, **kwargs)


def infer_sync_count(device, train_yes_n, valid_yes_n, test_yes_n, train_n,
                     valid_n, test_n):
    count_reduce = th.Tensor([
        float(train_yes_n),
        float(valid_yes_n),
        float(test_yes_n),
        float(train_n),
        float(valid_n),
        float(test_n)
    ])

    count_reduce = count_reduce.to(device=device)
    th.distributed.all_reduce(count_reduce, async_op=False)
    return count_reduce


def get_inc_count(batch_pred, batch_labels, mask):
    mask_idx = th.flatten(mask.nonzero())
    mask_pred = batch_pred[mask_idx]
    mask_label = batch_labels[mask_idx]
    n_correct = (th.argmax(mask_pred, dim=1) == mask_label).float().sum()
    return mask_idx.shape[0], n_correct


def run_infer_xborder(model, isloader: IsolatedDataloader, device, args,
                      subset_tag):
    model.eval()

    infer_loader = isloader.xb_dataloader(args,
                                          force_even=True,
                                          full_neighbor=False,
                                          subset_tag=subset_tag)
    train_n = 0
    valid_n = 0
    test_n = 0
    train_yes_n = 0
    valid_yes_n = 0
    test_yes_n = 0
    with th.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(infer_loader):
            batch_labels = isloader.get_batch_labels(seeds)
            batch_inputs = th.zeros([input_nodes.shape[0], isloader.in_feats],
                                    dtype=th.float32,
                                    device=isloader.da.comm_dev())
            isloader.xb_get_batch_input(batch_inputs, input_nodes)
            batch_labels = batch_labels.to(device)
            batch_inputs = batch_inputs.to(device)

            blocks = [block.to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs)

            if subset_tag == "all" or subset_tag == "train":
                mask = isloader.infer_train_mask_of(seeds)
                inc_n, inc_yes_n = get_inc_count(batch_pred, batch_labels, mask)
                train_n += inc_n
                train_yes_n += inc_yes_n

            if subset_tag == "all" or subset_tag == "valid":
                mask = isloader.infer_valid_mask_of(seeds)
                inc_n, inc_yes_n = get_inc_count(batch_pred, batch_labels, mask)
                valid_n += inc_n
                valid_yes_n += inc_yes_n

            if subset_tag == "all" or subset_tag == "test":
                mask = isloader.infer_test_mask_of(seeds)
                inc_n, inc_yes_n = get_inc_count(batch_pred, batch_labels, mask)
                test_n += inc_n
                test_yes_n += inc_yes_n

    model.train()
    return infer_sync_count(isloader.get_comm_dev(), train_yes_n, valid_yes_n,
                            test_yes_n, train_n, valid_n, test_n)
    #return train_yes_n, valid_yes_n, test_yes_n, train_n, valid_n, test_n


def run_infer_xborder_by_layer(model, isloader: IsolatedDataloader, device,
                               args, subset_tag):

    model.eval()
    with th.no_grad():
        infer_hidden_dim = model.n_hidden * (model.n_heads
                                             if model.name == 'gat' else 1)
        y = th.zeros([isloader.get_num_node(), infer_hidden_dim],
                     dtype=th.float32,
                     device=isloader.get_comm_dev())
        y_label = th.zeros([isloader.get_num_node(), isloader.n_classes],
                           dtype=th.float32,
                           device=isloader.get_comm_dev())

        for layer_i, layer in enumerate(model.layers):
            gc.collect()
            th.cuda.empty_cache()
            ifdbg(isloader, f"infer:layer_{layer_i}")
            # pyml_handle.report_memory(isloader.rank, "infer_xborder")

            infer_loader = isloader.xb_dataloader_per_layer(
                args, subset_tag=subset_tag)
            for step, (input_nodes, output_nodes,
                       blocks) in enumerate(infer_loader):
                #ifdbg(isloader, f"infer:layer_{layer_i}:step_{step}:{max_batch_num}")
                block = blocks[0].to(device)
                if layer_i == 0:
                    # input_nodes are local, see xb_dataloader_per_layer
                    batch_input = th.zeros(
                        [input_nodes.shape[0], isloader.in_feats],
                        dtype=th.float32,
                        device=isloader.da.part_feat_dev())
                    isloader.get_batch_input(batch_input, input_nodes)
                    h = batch_input.to(device)
                    del batch_input
                else:
                    #ifdbg(isloader, f"infer:layer_{layer_i}:{step}: batch_emb:({input_nodes.shape[0]}, {y.shape[1]}) ")
                    batch_emb = th.zeros([input_nodes.shape[0], y.shape[1]],
                                         dtype=th.float32,
                                         device=y.device)
                    isloader.xb_get_batch_emb(batch_emb, input_nodes, y)
                    h = batch_emb
                    del batch_emb
                h_dst = h[:block.number_of_dst_nodes()]
                if model.name == 'gat':  # handle activations
                    h = layer(block, (h, h_dst))  # apply a layer
                    if layer_i != len(model.layers) - 1:
                        h = h.flatten(1)
                        h = model.activation(h)
                        h = model.dropout(h)
                    else:
                        h = h.mean(1)
                elif model.name in ['sage', 'gcn']:
                    h = layer(block, (h, h_dst))  # apply a layer
                    if layer_i != len(model.layers) - 1:
                        h = model.activation(h)
                        h = model.dropout(h)
                elif model.name == "gin":
                    h = layer(block, (h, h_dst))  # apply a layer
                    if layer_i != len(model.layers) - 1:
                        h = F.relu(h)
                elif model.name == 'pinsage':
                    eid = block.edata['_ID']
                    ew = th.ones((eid.shape[0],), device=eid.device)
                    h = layer(block, (h, h_dst), ew)  # apply a layer
                else:
                    assert False, f"{model.name} is not tested"

                index_nodes = output_nodes - isloader.get_nid_offset()
                if layer_i == len(model.layers) - 1:
                    y_label[index_nodes] = h
                else:
                    y[index_nodes] = h
                    # output_nodes is local
                    # otherwise use isloader.xb_scatter(h, output_nodes, y)
                index_nodes = None
                output_nodes = None
                block = None
                h = None
                h_dst = None
            # end for

    model.train()

    ground_truth = isloader.node_label.to(isloader.get_comm_dev())
    train_mask = isloader.train_mask.to(isloader.get_comm_dev())
    valid_mask = isloader.valid_mask.to(isloader.get_comm_dev())
    test_mask = isloader.test_mask.to(isloader.get_comm_dev())

    train_n = 0
    valid_n = 0
    test_n = 0
    train_yes_n = 0
    valid_yes_n = 0
    test_yes_n = 0

    if subset_tag == "all" or subset_tag == "train":
        train_n, train_yes_n = get_inc_count(y_label, ground_truth, train_mask)

    if subset_tag == "all" or subset_tag == "valid":
        valid_n, valid_yes_n = get_inc_count(y_label, ground_truth, valid_mask)

    if subset_tag == "all" or subset_tag == "test":
        test_n, test_yes_n = get_inc_count(y_label, ground_truth, test_mask)

    y = None
    y_label = None

    return infer_sync_count(isloader.get_comm_dev(), train_yes_n, valid_yes_n,
                            test_yes_n, train_n, valid_n, test_n)
