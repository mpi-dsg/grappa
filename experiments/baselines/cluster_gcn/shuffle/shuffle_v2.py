import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# We will have 1024 tensors in total, each machine initially holds 256 tensors.
WORD_SIZE = 4
TOTAL_TENSORS = 1024
NUM_TENSORS_PER_MACHINE = TOTAL_TENSORS / WORD_SIZE
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
    tensors = [torch.randn(TENSOR_SIZE, dtype=torch.float32) for _ in range(NUM_TENSORS_PER_MACHINE)]
    
    print(f"Process {rank} - Initial tensors, first tensor data: {tensors[0][:5]}...")  # Display first 5 elements of the first tensor

    # Step 1: Master node (rank 0) maintains the global index of tensor locations
    if rank == 0:
        # Create a global map of tensor indices to machine (0-255 -> machine 0, 256-511 -> machine 1, etc.)
        all_tensor_indices = torch.arange(TOTAL_TENSORS)
        
        # Shuffle the global tensor indices
        perm = torch.randperm(TOTAL_TENSORS)
        shuffled_indices = all_tensor_indices[perm]
        
        # Broadcast the shuffled indices to all machines
        dist.broadcast(shuffled_indices, src=0)
        print(f"Master node (rank 0) - Shuffled indices: {shuffled_indices[:10]}...")
    else:
        # Receive the shuffled indices on all non-master nodes
        shuffled_indices = torch.zeros(TOTAL_TENSORS, dtype=torch.long)
        dist.broadcast(shuffled_indices, src=0)
        print(f"Process {rank} - Received shuffled indices: {shuffled_indices[:10]}...")

    # Step 2: Every machine computes which tensors it needs to gather based on shuffled indices
    new_tensor_indices = shuffled_indices[rank * NUM_TENSORS_PER_MACHINE:(rank + 1) * NUM_TENSORS_PER_MACHINE]
    
    # Step 3: Gather the required tensors from the respective machines
    recv_tensors = [torch.zeros(TENSOR_SIZE, dtype=torch.float32) for _ in range(NUM_TENSORS_PER_MACHINE)]
    send_requests = []
    recv_requests = []
    
    for i, idx in enumerate(new_tensor_indices):
        source_machine = idx // NUM_TENSORS_PER_MACHINE  # Determine which machine the tensor is currently on
        
        if source_machine == rank:
            # If the tensor is already on this machine, no need to transfer
            recv_tensors[i] = tensors[idx % NUM_TENSORS_PER_MACHINE]
        else:
            # Receive the tensor from the corresponding machine
            recv_requests.append(dist.irecv(recv_tensors[i], src=source_machine))
    
    # Send required tensors to other machines
    for i, idx in enumerate(torch.arange(rank * NUM_TENSORS_PER_MACHINE, (rank + 1) * NUM_TENSORS_PER_MACHINE)):
        destination_machine = torch.where(shuffled_indices == idx)[0].item() // NUM_TENSORS_PER_MACHINE
        if destination_machine != rank:
            send_requests.append(dist.isend(tensors[i], dst=destination_machine))
    
    # Wait for all sends and receives to complete
    for req in send_requests:
        req.wait()
    for req in recv_requests:
        req.wait()
    
    print(f"Process {rank} - Shuffled tensors, first received tensor data: {recv_tensors[0][:5]}...")  # Display first 5 elements of the first received tensor

    # Clean up the distributed environment
    cleanup()

def run_distributed(world_size):
    # This function spawns distributed processes for each machine
    mp.spawn(distributed_shuffle, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    world_size = WORD_SIZE  # Number of processes (4 machines)
    run_distributed(world_size)
