import argparse
import socket
import time
import os
import datetime
import numpy as np
from copy import deepcopy

from t10n.dataloader import IsolatedDataloader
from t10n.dataset.meta import name_to_meta

from t10n.util import pyml_handle
from t10n.grad import weighted_grad_hook

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
import dgl.backend as FF

from gcn_full import GCN
from graphsage_full import SAGE as GraphSAGE
from gat import GAT
from gin_full import GIN
from pinsage import PinSAGE


def pprint(*args, **kwargs):
    rank = th.distributed.get_rank()
    if rank % 2 == 0:
        print(f"rank{rank}", *args, **kwargs)


def compute_acc(pred, labels, device):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    n_correct = (th.argmax(pred, dim=1) == labels).float().sum()
    n_correct_total = th.Tensor([n_correct, float(len(pred))])
    n_correct_total = n_correct_total.to(device)
    th.distributed.all_reduce(n_correct_total, async_op=False)
    return n_correct_total


def get_model(args, n_feat_dim: int, n_class: int):
    if args.model == 'sage':
        model = GraphSAGE(
            n_feat_dim,
            args.num_hidden,
            n_class,
            args.num_layers,
            F.relu,
            args.dropout,
        )
        model.name = 'sage'
    elif args.model == 'gat':
        n_heads = 4
        model = GAT(
            n_feat_dim,
            args.num_hidden,
            n_class,
            n_heads,
            args.num_layers,
            F.relu,
            args.dropout,
        )
        model.name = 'gat'
    elif args.model == 'gcn':
        assert args.model == 'gcn'
        model = GCN(
            n_feat_dim,
            args.num_hidden,
            n_class,
            args.num_layers,
            F.relu,
            args.dropout,
        )
        model.name = 'gcn'
    elif args.model == 'gin':
        model = GIN(n_feat_dim, args.num_hidden, n_class)
        # model = GIN(n_feat_dim, args.num_hidden, n_class, num_layers=args.num_layers)
        model.name = 'gin'
    elif args.model == 'pinsage':
        model = PinSAGE(n_feat_dim,
                        args.num_hidden,
                        n_class,
                        args.num_layers,
                        F.relu,
                        args.dropout)
        model.name = 'pinsage'
    else:
        assert False

    return model

def warp_model_for_ddp(model, device, args):
    if args.standalone:
        return model

    if args.num_gpus == -1:
        model = th.nn.parallel.DistributedDataParallel(model)
    else:
        model = th.nn.parallel.DistributedDataParallel(model,
                                                       device_ids=[device],
                                                       output_device=device)
    return model


def resum_from_ckpt(model, optimizer, resume_path):
    # resume from checkpoint if resume_path is not None
    ckpt_epoch = -1
    if args.resume_path is None or args.resume_path.strip() == '':
        return ckpt_epoch
    else:
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        # ckpt = th.load(checkpoint_path, map_location=map_location))
        ckpt = th.load(args.resume_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        ckpt_epoch = ckpt['epoch']
        return ckpt_epoch


def save_to_ckpt(args, epoch, rank, model, optimizer, loss):
    flag = args.checkpoint_path is not None
    flag = flag and args.checkpoint_path.strip() != ''
    if flag and args.checkpoint_every > 0:
        # epoch starts from 0
        if (epoch + 1) % args.checkpoint_every == 0 and rank == 0:
            ckpt_name = f"{args.checkpoint_path}/epoch_{epoch:03d}.pt"
            th.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, ckpt_name)

def run(args, device):
    meta_o = name_to_meta(args.graph_name)
    model = get_model(args, meta_o.n_dim, meta_o.n_label)

    model = model.to(device)
    model = warp_model_for_ddp(model, device, args)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    batch_state = {'coarse_count': 0.0}
    if not args.standalone and args.n_part > 1:
        model.register_comm_hook(state=batch_state, hook=weighted_grad_hook)

    ckpt_epoch = resum_from_ckpt(model, optimizer, args.resume_path)

    # loader
    loader = IsolatedDataloader(device, args)
    rank = loader.get_rank()

    best = {'val_acc': 0.0, 'state_dict': None}

    dataloader = loader.get_cur_dataloader(args, 0, overide_exp_mode=1)
    cur_g = loader.cur_g.to(device)
    feat = loader.cur_node_feat.to(device)
    part_feat  = feat.to(device)
    ground_truth = loader.node_label

    # Training loop
    iter_tput = []
    for epoch in range(0, args.num_epochs):
        tic = time.time()

        sample_time = 0
        f_fetch_time = 0  # graph struct copy time
        pcie_time = 0  # feature copy time
        forward_time = 0
        backward_time = 0
        update_time = 0
        account_time = 0  # calculating time for log
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        step_time = []


        with model.join():
            bbbbbe = time.time()


            pcie_end = time.time()
            pcie_time += pcie_end - bbbbbe

            part_pred = model(cur_g, part_feat)
            part_label = ground_truth.to(device).long()
            num_label = part_label.shape[0]
            loss = loss_fcn(part_pred[:num_label], part_label)
            forward_end = time.time()
            optimizer.zero_grad()
            ###################
            #part_opt.zero_grad() # part_bias
            ###################
            loss.backward()
            compute_end = time.time()
            forward_time += forward_end - pcie_end
            backward_time += compute_end - forward_end

            optimizer.step()
            ###################
            #part_opt.step() # part_bias
            ###################
            update_end = time.time()
            update_time += update_end - compute_end

            step_t = update_end - bbbbbe
            step_time.append(step_t)

            #print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step} flag 3")

            if epoch % args.log_every == 0:
                acc_t = compute_acc(part_pred[:part_label.shape[0]], part_label,
                                    loader.get_comm_dev())
                gpu_mem_alloc = (th.cuda.max_memory_allocated() /
                                    1000000 if th.cuda.is_available() else 0)
                pprint(
                    "epoch|{:04d}|part|{:04d}|loss|{:.4f}|"
                    "train_acc|{:.4f}|gpu_mb|{:.1f}|step_time|{:.2f}|train:{:.1f},{:.1f}"
                    .format(epoch, rank, loss.item(),
                            acc_t[0].item() / acc_t[1].item(),
                            gpu_mem_alloc,
                            np.sum(step_time[-args.log_every:]),
                            acc_t[0].item(), acc_t[1].item()))

            account_end = time.time()
            account_time += account_end - update_end
            start = account_end

            del part_pred
            del part_label

        toc = time.time()

        pprint(
            "epoch_|epoch|{:04d}|part|{:04d}|epoch_seconds|{:.4f}|pcie|{:.4f}|"
            "forward|{:.4f}|backward|{:.4f}|update|{:.4f}|account|{:.4f}"
            .format(
                epoch,
                rank,
                toc - tic,
                pcie_time,
                forward_time,
                backward_time,
                update_time,
                account_time
            ))



def main(args):
    print(socket.gethostname(), "Initializing Triskelion")

    if not args.standalone:
        print(socket.gethostname(), "Initializing Triskelion process group")
        t_b = time.time()
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        WORLD_RANK = int(os.environ['RANK'])
        os.environ['OMP_PROC_BIND'] = 'true'
        if args.backend == "gloo":
            th.distributed.init_process_group(
                backend=args.backend,
                timeout=datetime.timedelta(seconds=3600 * 5))
        elif args.backend == "nccl":
            master_ip = os.environ['MASTER_ADDR']
            master_port = os.environ['MASTER_PORT']

            os.environ['NCCL_SOCKET_IFNAME'] = args.socket_ifname
            os.environ['NCCL_DEBUG'] = 'WARN'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

            #os.environ['CUDA_LAUNCH_BLOCKING']='1'
            dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
                master_ip=master_ip, master_port=master_port)
            th.distributed.init_process_group(
                backend=args.backend,
                init_method=dist_init_method,
                world_size=WORLD_SIZE,
                rank=WORLD_RANK,
                timeout=datetime.timedelta(seconds=3600 * 5))
        print("local_rank_{}, init_pg: {:.4f} sec".format(
            LOCAL_RANK,
            time.time() - t_b))

    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        if args.standalone:
            dev_id = 0
        else:
            dev_id = th.distributed.get_rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    run(args, device)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t10n_train")
    ######################
    # dataset parameters
    ######################
    parser.add_argument("--data_path",
                        type=str,
                        help="path where you put dataset for t10n")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--part_method",
                        type=str,
                        help="name of the partition method")
    parser.add_argument("--n_part", type=int, help="number of partitions")

    ######################
    # model parameters
    ######################
    parser.add_argument("--model",
                        type=str,
                        default="sage",
                        required=True,
                        help="gnn model in(sage, gcn, gat)")
    parser.add_argument("--num_hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fan_out", type=str, default="15,10,5")

    ######################
    # hyper parameters
    ######################
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--stop_at_border",
                        action="store_true",
                        default=False,
                        help="sampler will stop at border")
    parser.add_argument(
        "--disable_backup_server",
        action="store_true",
        default=False,
        help=
        "when enabled, 1 graph server will serve 1 partition, no backup servers"
    )
    parser.add_argument("--repart_every", type=int, default=50)

    ######################
    # runtime parameters
    ######################
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="#GPU per machine. Use -1 for CPU training",
    )
    parser.add_argument("--standalone",
                        action="store_true",
                        help="run in the standalone mode")
    parser.add_argument("--backend",
                        type=str,
                        default="gloo",
                        help="pytorch distributed backend")
    parser.add_argument("--socket_ifname",
                        type=str,
                        default="eth0",
                        help="pytorch distributed backend")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)

    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="resume from a path of a checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="a path to store all checkpoints",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=-1,
        help="save a checkpoint erver N EPOCHS",
    )

    parser.add_argument("--tag_id",
                        type=str,
                        required=True,
                        help="name for this job")
    parser.add_argument("--close_dd",
                        action="store_true",
                        help="dont use dgl dsg")

    args = parser.parse_args()

    print(args)
    main(args)
