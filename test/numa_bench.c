#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numa.h>
#include <omp.h>
#include <sched.h>
#include <sys/time.h>

double get_time_in_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

void memory_benchmark(double *array, size_t size, const char *description) {
    size_t i;
    double start_time, end_time, duration;

    // Write benchmark
    start_time = get_time_in_seconds();
    #pragma omp parallel
    {
        #pragma omp for
        for (i = 0; i < size; ++i) {
            array[i] = i * 1.0;
        }
    }
    end_time = get_time_in_seconds();
    duration = end_time - start_time;
    printf("%s Write Bandwidth: %f GB/s\n", description, (size * sizeof(double)) / duration / 1e9);

    // Read benchmark
    start_time = get_time_in_seconds();
    double sum = 0.0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:sum)
        for (i = 0; i < size; ++i) {
            sum += array[i];
        }
    }
    end_time = get_time_in_seconds();
    duration = end_time - start_time;
    printf("%s Read Bandwidth: %f GB/s\n", description, (size * sizeof(double)) / duration / 1e9);
}

int main() {
    if (numa_available() < 0) {
        printf("NUMA is not available on this system.\n");
        return 1;
    }

    // Get the number of NUMA nodes
    int num_nodes = numa_max_node() + 1;
    printf("Number of NUMA nodes: %d\n", num_nodes);

    // Define array size (256 MB)
    size_t size = (256 * 1024 * 1024) / sizeof(double);

    // Allocate memory on each NUMA node
    double *local_memory = (double *)numa_alloc_onnode(size * sizeof(double), 0);
    double *remote_memory = (double *)numa_alloc_onnode(size * sizeof(double), num_nodes > 1 ? 1 : 0);

    if (local_memory == NULL || remote_memory == NULL) {
        printf("Failed to allocate memory on NUMA nodes.\n");
        return 1;
    }


    // Run benchmarks
    printf("Running local memory benchmark (NUMA node 0)...\n");
    memory_benchmark(local_memory, size, "Local");

    if (num_nodes > 1) {
        printf("Running remote memory benchmark (NUMA node 1)...\n");
        memory_benchmark(remote_memory, size, "Remote");
    } else {
        printf("Only one NUMA node available; skipping remote memory benchmark.\n");
    }

    // Free the allocated memory
    numa_free(local_memory, size * sizeof(double));
    numa_free(remote_memory, size * sizeof(double));
    printf("Memory freed.\n");

    return 0;
}
