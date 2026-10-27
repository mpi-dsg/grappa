import argparse
import socket
import time
import os
import datetime
import numpy as np
from copy import deepcopy

from t10n.dataloader import IsolatedDataloader
from t10n.dataset.meta import name_to_meta

from t10n.infer import run_infer_xborder_by_layer as run_infer

from t10n.util import pyml_handle
from t10n.grad import weighted_grad_hook

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
import dgl.backend as FF

from gcn import GCN
from graphsage import SAGE as GraphSAGE
from gat import GAT
from gin import GIN
from pinsage import PinSAGE

import faulthandler
faulthandler.enable()

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
        model = GIN(n_feat_dim, args.num_hidden, n_class, args.num_layers)
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

    # model = th.nn.parallel.DistributedDataParallel(model)
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

    # Training loop
    iter_tput = []
    for epoch in range(ckpt_epoch + 1, args.num_epochs):
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
        dataloader = loader.get_cur_dataloader(args, epoch)
        max_step = loader.get_max_step()

        with model.join():
            bbbbbe = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step}: {time.time() - bbbbbe: .2f} sec")
                #bbbbbe = time.time()
                #if rank == 14:
                #    print("with model.join():")

                #pprint(f"epoch@{epoch}:step@{step}")
                tic_step = time.time()
                sample_time += tic_step - start
                # fetch features/labels
                batch_inputs = th.zeros([input_nodes.shape[0], loader.in_feats],
                                        dtype=th.float32,
                                        device=loader.da.part_feat_dev())
                loader.get_batch_input(batch_inputs, input_nodes)
                batch_state['coarse_count'] = loader.get_coarse_count()
                print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step} flag 1")

                #pyml_handle.report_memory(loader.rank,
                #                          f"train@epoch{epoch}@step{step}")
                batch_labels = loader.get_batch_labels(seeds)
                #batch_inputs = batch_inputs + part_bias
                f_fetch_end = time.time()
                f_fetch_time += f_fetch_end - tic_step
                batch_labels = batch_labels.long()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                # move to target device
                print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step} flag 1.2")
                for block in blocks:
                    print(f"{block.device}")
                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                pcie_end = time.time()
                pcie_time += pcie_end - f_fetch_end
                # Compute loss and prediction
                #start = time.time()
                print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step} flag 1.3")
                batch_pred = model(blocks, batch_inputs)
                print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step} flag 2")
                #batch_labels.zero_()
                #batch_pred.zero_()
                loss = loss_fcn(batch_pred, batch_labels)
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

                step_t = update_end - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)

                print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step} flag 3")

                if step % args.log_every == 0:
                    acc_t = compute_acc(batch_pred, batch_labels,
                                        loader.get_comm_dev())
                    gpu_mem_alloc = (th.cuda.max_memory_allocated() /
                                     1000000 if th.cuda.is_available() else 0)
                    pprint(
                        "step_|epoch|{:04d}|step|{:04d}|part|{:04d}|loss|{:.4f}|"
                        "train_acc|{:.4f}|sample_p_s|{:.2f}|gpu_mb|{:.1f}|step_time|{:.2f}|train:{:.1f},{:.1f}"
                        .format(epoch, step, rank, loss.item(),
                                acc_t[0].item() / acc_t[1].item(),
                                np.mean(iter_tput[3:]), gpu_mem_alloc,
                                np.sum(step_time[-args.log_every:]),
                                acc_t[0].item(), acc_t[1].item()))

                account_end = time.time()
                account_time += account_end - update_end
                start = account_end
                if step >= max_step:
                    break
                del batch_inputs
                del batch_labels
                print(f"STEP : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {step} flag 4")

        #with model.join() ends

        toc = time.time()

        pprint(
            "epoch_|epoch|{:04d}|part|{:04d}|epoch_seconds|{:.4f}|sampling|{:.4f}|f_fetch|{:.4f}|pcie|{:.4f}|"
            "forward|{:.4f}|backward|{:.4f}|update|{:.4f}|account|{:.4f}|n_seed|{:012d}|n_input|{:012d}"
            .format(
                epoch,
                rank,
                toc - tic,
                sample_time,
                f_fetch_time,
                pcie_time,
                forward_time,
                backward_time,
                update_time,
                account_time,
                num_seeds,
                num_inputs,
            ))

        # keep model of best valid accuracy
        if (epoch != 0) and (epoch + 1) % args.eval_every == 0:
            reduced_count = run_infer(
                model if args.standalone else model.module,
                loader,
                device,
                args,
                subset_tag="valid")

            valid_t = float(reduced_count[1])
            valid_a = float(reduced_count[4])

            val_acc = valid_t / valid_a
            pprint(f"epoch={epoch}, val_acc={val_acc}")
            if (val_acc > best["val_acc"]):
                best["val_acc"] = val_acc
                m_handle = model if args.standalone else model.module
                best["state_dict"] = deepcopy(m_handle.state_dict())
    # end for

    if best["state_dict"] is None:
        exit(0)
    print(f"I'm rank : {rank} !")

    start = time.time()
    m_handle = model if args.standalone else model.module
    m_handle.load_state_dict(best["state_dict"])
    reduced_count = run_infer(m_handle, loader, device, args, subset_tag="all")

    valid_t = float(reduced_count[1])
    valid_a = float(reduced_count[4])

    test_t = float(reduced_count[2])
    test_a = float(reduced_count[5])

    train_t = float(reduced_count[0])
    train_a = float(reduced_count[3])

    pprint(
        "infer_|epoch|{:04d}|part|{:04d}|val_acc|{:.4f}|test_acc|{:.4f}|train_acc|{:.4f}|time_sec|{:.4f}|val:{:.1f}|test:{:.1f}|train:{:.1f}"
        .format(epoch, rank, valid_t / valid_a, test_t / test_a,
                train_t / train_a,
                time.time() - start, valid_a, test_a, train_a))

    save_to_ckpt(args, epoch, rank, m_handle, optimizer, loss)


def main(args):
    print(socket.gethostname(), "Initializing Triskelion")

    if not args.standalone:
        print(socket.gethostname(), "Initializing Triskelion process group")
        t_b = time.time()
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        WORLD_RANK = int(os.environ['RANK'])
        SLURM_NODEID = int(os.environ.get('SLURM_NODEID', 0))
        #LOCAL_RANK = int(os.environ.get('SLURM_LOCALID', os.environ.get('LOCAL_RANK', 0)))
        #WORLD_SIZE = int(os.environ.get('SLURM_NTASKS', os.environ.get('WORLD_SIZE', 1)))
        #WORLD_RANK = int(os.environ.get('SLURM_PROCID', os.environ.get('RANK', 0)))

        print(f"LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE}, WORLD_RANK: {WORLD_RANK}, SLURM_NODEID: {SLURM_NODEID}", flush=True)

        #os.environ['OMP_PROC_BIND'] = 'true'
        if args.backend == "gloo":
            th.distributed.init_process_group(
                backend=args.backend,
                timeout=datetime.timedelta(seconds=3600 * 24))
        elif args.backend == "nccl":
            master_ip = os.environ['MASTER_ADDR']
            master_port = os.environ['MASTER_PORT']

            os.environ['NCCL_SOCKET_IFNAME'] = args.socket_ifname
            ### os.environ['NCCL_DEBUG'] = 'WARN'
            ### os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

            #os.environ['CUDA_LAUNCH_BLOCKING']='1'
            dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
                master_ip=master_ip, master_port=master_port)
            th.distributed.init_process_group(
                backend=args.backend,
                init_method=dist_init_method,
                world_size=WORLD_SIZE,
                rank=WORLD_RANK,
                timeout=datetime.timedelta(seconds=3600 * 24))
        print("local_rank_{}, init_pg: {:.4f} sec".format(
            LOCAL_RANK,
            time.time() - t_b))

    #if True:
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
