import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from typing import List

# We will have 1024 tensors in total, each machine initially holds 256 tensors.
WORD_SIZE = 4
TOTAL_TENSORS = 1024
NUM_TENSORS_PER_MACHINE = TOTAL_TENSORS // WORD_SIZE
TENSOR_SIZE = 1024 * 1024 // 4  # This makes a 1MB tensor of float32 (4 bytes per element)

def setup(rank, world_size):
    # Initializes the distributed backend and process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    # Cleans up the distributed process group
    dist.destroy_process_group()

def distributed_shuffle(rank, world_size):
    # Setup distributed environment
    setup(rank, world_size)

    # Each process creates 256 random tensors (each tensor is 1MB)
    tensors: List[torch.Tensor] = [torch.randn(TENSOR_SIZE, dtype=torch.float32) for _ in range(NUM_TENSORS_PER_MACHINE)]

    print(f"Process {rank} - Initial tensors, first tensor data: {tensors[0][:5]}...")  # Display first 5 elements of the first tensor

    # Gather tensors from all processes into a list
    gathered_tensors : List[torch.Tensor] = []
    for _ in range(world_size):
        gathered_tensors.append([torch.zeros_like(tensors[0]) for _ in range(NUM_TENSORS_PER_MACHINE)])

    start = time.time()
    # Use all_gather to collect all tensors from each machine
    for i in range(NUM_TENSORS_PER_MACHINE):
        gathered_tensor: List[torch.Tensor] = [torch.zeros_like(tensors[i]) for _ in range(world_size)]
        dist.all_gather(gathered_tensor, tensors[i])
        for j in range(world_size):
            gathered_tensors[j][i] = gathered_tensor[j]

    print(f"all_gather: {time.time() - start: .2f} seconds")
    # Flatten the list of lists to get all 1024 tensors into a single list
    all_tensors: List[torch.Tensor] = [t for sublist in gathered_tensors for t in sublist]

    # Shuffle the tensors
    perm = torch.randperm(TOTAL_TENSORS)
    shuffled_tensors: List[torch.Tensor] = [all_tensors[i] for i in perm]

    # Split shuffled tensors back into chunks of 256 per machine
    split_tensors: List[torch.Tensor] = [shuffled_tensors[i * world_size:(i + 1) * world_size] for i in range(NUM_TENSORS_PER_MACHINE)]

    # Scatter the shuffled data back to all processes
    for i in range(NUM_TENSORS_PER_MACHINE):
        scatter_list_input = split_tensors[i] if rank == 0 else None
        #print(type(scatter_list_input))
        #print(type(scatter_list_input[0]))
        dist.scatter(tensors[i], scatter_list=scatter_list_input)

    end = time.time()
    print(f"Process {rank} - Shuffled tensors, first tensor data: {tensors[0][:5]}...")  # Display first 5 elements of the first tensor
    print(f"{rank}: time: {end - start : .2f}")
    # Clean up the distributed environment
    cleanup()

def run_distributed(world_size):
    # This function spawns distributed processes for each machine
    # mp.spawn(distributed_shuffle, args=(world_size,), nprocs=world_size, join=True)
    pass

if __name__ == "__main__":
    world_size = WORD_SIZE  # Number of processes (4 machines)
    WORLD_RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])

    print(WORLD_RANK, LOCAL_RANK)
    distributed_shuffle(WORLD_RANK, world_size)
