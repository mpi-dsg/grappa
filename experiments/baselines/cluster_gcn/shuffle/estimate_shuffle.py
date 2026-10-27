import random
import numpy as np
import copy
from collections import Counter

def simulate_shuffle(D, C, N, B):
    """
    Simulate a real shuffle of data chunks across N nodes and calculate
    the actual amount of data each node needs to send to every other node.

    Parameters:
    D (float): Total size of the data blob in MB.
    C (int): Number of chunks the data is partitioned into.
    N (int): Number of machine nodes.
    B (float): Network bandwidth of each node in MB/s.

    Returns:
    dict: A nested dictionary containing the amount of data each node sends to every other node.
    """
    chunk_size = D / C  # Size of each chunk in MB
    print(chunk_size)
    # Step 1: Initial assignment of chunks to nodes (even distribution)
    initial_assignment = []  # chunk_id -> node_id
    chunks_per_node = C // N
    extra_chunks = C % N  # Handle uneven distribution
    chunk_id = 0
    for node_id in range(N):
        num_chunks = chunks_per_node + (1 if node_id < extra_chunks else 0)
        for _ in range(num_chunks):
            initial_assignment.append(node_id)

    # Step 2: Randomly shuffle chunks to new nodes
    new_assignment = copy.deepcopy(initial_assignment)
    np.random.shuffle(new_assignment)
    frequency = Counter(new_assignment)
    print(f"shuffled assignment: {frequency}")

    # Step 3: Calculate data each node needs to send to every other node
    data_to_send = {i: {j: 0 for j in range(N)} for i in range(N)}
    for chunk_id in range(C):
        current_node = initial_assignment[chunk_id]
        new_node = new_assignment[chunk_id]
        if current_node != new_node:
            data_to_send[current_node][new_node] += chunk_size

    # Step 4: Summarize total data sent and received per node
    total_sent = {node: sum(data_to_send[node].values()) for node in range(N)}
    total_received = {node: sum(data_to_send[i][node] for i in range(N)) for node in range(N)}

    return data_to_send, total_sent, total_received

def calculate_transfer_times(data_to_send, N, B, alpha=1.5):
    """
    Calculate the transfer times between nodes considering TCP network congestion.

    Parameters:
    data_to_send (dict): Nested dictionary with data to send from node i to node j.
    N (int): Number of machine nodes.
    B (float): Network bandwidth of each node in MB/s.
    alpha (float): Congestion exponent to model TCP congestion effects.

    Returns:
    tuple: transfer_times, total_times, overall_shuffle_time
    """
    # Step 1: Summarize total data sent and received per node
    total_sent = {node: sum(data_to_send[node].values()) for node in range(N)}
    total_received = {node: sum(data_to_send[i][node] for i in range(N)) for node in range(N)}

    # Step 2: Calculate the number of outgoing and incoming connections
    outgoing_connections = {
        node: sum(1 for dest in data_to_send[node] if data_to_send[node][dest] > 0 and dest != node)
        for node in range(N)
    }
    incoming_connections = {
        node: sum(1 for src in range(N) if data_to_send[src][node] > 0 and src != node)
        for node in range(N)
    }

    # Step 3: Adjust effective bandwidths considering TCP congestion
    effective_out_bandwidth = {
        node: B / max(1, outgoing_connections[node] ** alpha)
        for node in range(N)
    }
    effective_in_bandwidth = {
        node: B / max(1, incoming_connections[node] ** alpha)
        for node in range(N)
    }

    # Step 4: Compute time for each data transfer
    transfer_times = {i: {j: 0 for j in range(N)} for i in range(N)}
    for i in range(N):
        for j in range(N):
            if i != j and data_to_send[i][j] > 0:
                # Effective bandwidth is the minimum of sender's out and receiver's in bandwidth
                eff_bandwidth = min(effective_out_bandwidth[i], effective_in_bandwidth[j])
                transfer_times[i][j] = data_to_send[i][j] / eff_bandwidth

    # Step 5: Compute total time per node
    send_times = {
        node: max([transfer_times[node][j] for j in range(N) if j != node and transfer_times[node][j] > 0] + [0])
        for node in range(N)
    }
    receive_times = {
        node: max([transfer_times[i][node] for i in range(N) if i != node and transfer_times[i][node] > 0] + [0])
        for node in range(N)
    }
    total_times = {node: max(send_times[node], receive_times[node]) for node in range(N)}

    # Step 6: Compute overall shuffle time
    overall_shuffle_time = max(total_times.values())

    return transfer_times, total_times, overall_shuffle_time

def calculate_transfer_times_full(data_to_send, N, B):

    time_to_send = {node: total_sent[node] / B for node in range(N)}
    time_to_receive = {node: total_received[node] / B for node in range(N)}

    # Print the results
    print("Data each node needs to send to every other node (in MB):\n")
    for i in range(N):
        for j in range(N):
            if i != j:
                print(f"Node {i} sends {data_to_send[i][j]:.2f} MB to Node {j}")
        print(f"Total sent by Node {i}: {total_sent[i]:.2f} MB")
        print(f"Total received by Node {i}: {total_received[i]:.2f} MB")
        print(f"Time to send data for Node {i}: {time_to_send[i]:.2f} seconds")
        print(f"Time to receive data for Node {i}: {time_to_receive[i]:.2f} seconds")
        print("-" * 50)

# Example usage:
D = 1416  # Total data size in MB
C = 1600     # Number of chunks
N = 2      # Number of nodes
B = 600.0   # Bandwidth in MB/s

data_to_send, total_sent, total_received = simulate_shuffle(D, C, N, B)
calculate_transfer_times_full(data_to_send, N, B)
transfer_times, total_times, overall_shuffle_time = calculate_transfer_times(data_to_send, N, B)

# Print the results
print("Data each node needs to send to every other node (in MB):\n")
for i in range(N):
    for j in range(N):
        if i != j and data_to_send[i][j] > 0:
            print(f"Node {i} sends {data_to_send[i][j]:.2f} MB to Node {j}")
    print(f"Total sent by Node {i}: {total_sent[i]:.2f} MB")
    print(f"Total received by Node {i}: {total_received[i]:.2f} MB")
    print("-" * 50)

print("\nTransfer times between nodes (in seconds):\n")
for i in range(N):
    for j in range(N):
        if i != j and transfer_times[i][j] > 0:
            print(f"Time for Node {i} to send data to Node {j}: {transfer_times[i][j]:.2f} seconds")
    print(f"Total time for Node {i}: {total_times[i]:.2f} seconds")
    print("-" * 50)

print(f"Overall shuffle time: {overall_shuffle_time:.2f} seconds")
