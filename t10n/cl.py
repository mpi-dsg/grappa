from t10n.util import timing, pprint, backtrace, pyml_handle

import torch as th


def cldbg(rank, *args, **kwargs):
    pass
    #pprint(rank, f"rank{rank:02d}:", *args, **kwargs)


def total_mb(list_of_count_th, ele_size):
    sum_num = 0
    for len_th in list_of_count_th:
        mb = float(len_th[0].item() * ele_size) / 1024 / 1024
        sum_num += mb
    return sum_num


def t10n_gather_gloo(in_tensor, rank, world_size, dst_rank):
    assert th.is_tensor(in_tensor)
    assert len(in_tensor.shape) == 1 or len(in_tensor.shape) == 2

    def alloc_zero(first_dim, refer_tensor):
        if len(refer_tensor.shape) == 1:
            return th.zeros((first_dim,), dtype=refer_tensor.dtype)
        elif len(refer_tensor.shape) == 2:
            return th.zeros((first_dim, refer_tensor.shape[1]),
                            dtype=refer_tensor.dtype)
        else:
            assert False

    max_of_all = th.tensor(in_tensor.shape[0], dtype=th.int64)
    th.distributed.all_reduce(max_of_all, op=th.distributed.ReduceOp.MAX)
    len_per_rank = th.tensor(in_tensor.shape[0], dtype=th.int64)
    len_all_rank = [th.tensor(0, dtype=th.int64) for i in range(world_size)
                   ] if dst_rank == rank else None
    th.distributed.gather(len_per_rank, gather_list=len_all_rank, dst=dst_rank)

    gather_list = [
        alloc_zero(max_of_all.item(), in_tensor) for i in range(world_size)
    ] if dst_rank == rank else None
    padded_input = alloc_zero(max_of_all.item(), in_tensor)

    padded_input[:in_tensor.shape[0]] = in_tensor
    th.distributed.gather(padded_input, gather_list=gather_list, dst=dst_rank)

    final_gathered = None
    if dst_rank == rank:
        total_length = sum([e.item() for e in len_all_rank])
        final_gathered = alloc_zero(total_length, in_tensor)
        copy_idx = 0
        for ridx in range(world_size):
            length = len_all_rank[ridx].item()
            final_gathered[copy_idx:copy_idx +
                           length] = gather_list[ridx][:length]
            copy_idx += length

    th.distributed.barrier()
    return final_gathered


def t10n_all_to_all_1d_int64_nccl(list_of_th_tensor, rank, world_size, device):
    assert len(list_of_th_tensor[0].shape) == 1
    count_list = [
        th.tensor([arr.shape[0]], dtype=th.int64, device=device)
        for arr in list_of_th_tensor
    ]

    gather_count_list = [
        th.tensor([0], dtype=th.int64, device=device) for _ in range(world_size)
    ]
    cldbg(rank, "1d_int64 b count_list={count_list}")
    cldbg(rank, "1d_int64 b gather_count_list={gather_count_list}")
    th.distributed.all_to_all(gather_count_list, count_list)
    cldbg(rank, "1d_int64 e count_list={count_list}")
    cldbg(rank, "1d_int64 e gather_count_list={gather_count_list}")

    gather_arr_list = [
        th.full((length[0].item(),), 0, dtype=th.int64, device=device)
        for length in gather_count_list
    ]

    list_of_th_tensor_in = [te.to(device) for te in list_of_th_tensor]
    cldbg(rank, "1d_int64 b gather_arr_list={gather_arr_list}")
    th.distributed.all_to_all(gather_arr_list, list_of_th_tensor_in)
    cldbg(rank, "1d_int64 e gather_arr_list={gather_arr_list}")

    return gather_arr_list


def t10n_all_to_all_2d_float32_nccl(list_of_th_tensor, rank, world_size, device,
                                    dim_2nd):
    # some of the tensor could be empty
    #assert len(list_of_th_tensor[0].shape) == 2
    #assert list_of_th_tensor[0].shape[1] == dim_2nd

    count_list = [
        th.tensor([arr.shape[0]], dtype=th.int64, device=device)
        for arr in list_of_th_tensor
    ]
    cldbg(rank, "2d_float32 b gather_count_list={gather_count_list}")
    gather_count_list = [
        th.tensor([0], dtype=th.int64, device=device) for _ in range(world_size)
    ]
    th.distributed.all_to_all(gather_count_list, count_list)
    cldbg(rank, "2d_float32 e gather_count_list={gather_count_list}")

    #tmb = total_mb(gather_count_list, dim_2nd*4)
    #cldbg(rank, "2d_float32 alloc total_mb: {tmb}")
    gather_arr_list = [
        th.full((length, dim_2nd), 0, dtype=th.float32, device=device)
        for length in gather_count_list
    ]

    list_of_th_tensor_in = [te.to(device) for te in list_of_th_tensor]
    th.distributed.all_to_all(gather_arr_list, list_of_th_tensor_in)
    return gather_arr_list


def t10n_all_to_all_1d_int64_gloo(list_of_th_tensor, world_size, device):
    pass


def t10n_all_to_all_2d_float32_gloo(list_of_th_tensor, world_size, device,
                                    dim_2nd):
    pass
