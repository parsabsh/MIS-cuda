#include "helpers.cuh"
#include <stdio.h>
#include <time.h>

int main() {
    int n = 1000000;
    Graph* graph = new_graph_host(n, n);
    int E = 900000;
    add_random_edges(graph, E);

    //* uncomment to print the initial graph
    // print_graph(*graph);

    Graph *graph_dev;
    deep_copy(graph, &graph_dev);

    int* Flags = (int*) malloc(graph->n * sizeof(int));
    memset(Flags, 0, graph->n * sizeof(int));
    int* Flags_dev;
    cudaMalloc(&Flags_dev, graph->n * sizeof(int));
    cudaMemcpy(Flags_dev, Flags, graph->n * sizeof(int), cudaMemcpyHostToDevice);

    int* locks = (int*) malloc(graph->n * sizeof(int));
    memset(locks, 0, graph->n * sizeof(int));
    int* locks_dev;
    cudaMalloc(&locks_dev, graph->n * sizeof(int));
    cudaMemcpy(locks_dev, locks, graph->n * sizeof(int), cudaMemcpyHostToDevice);

    size_t threadsPerBlock = 1;
    dim3 blocksPerGrid = calculateGridDim(graph->n);

    clock_t begin = clock();

    maximalIndependentSet<<<blocksPerGrid, threadsPerBlock>>>(graph_dev, Flags_dev, locks_dev);
    cudaDeviceSynchronize();

    clock_t end = clock();
    double time_spent_parallel = (double)(end - begin) / CLOCKS_PER_SEC;

    cudaMemcpy(Flags, Flags_dev, graph->n * sizeof(int), cudaMemcpyDeviceToHost);

    //* uncomment to print the result mis
    // print_mis(Flags, graph->n);

    begin = clock();

    int *Flags_serial = maximalIndependentSetSerial(graph);

    end = clock();
    double time_spent_serial = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("CUDA implementation:\n");
    printf("execution time: %.6f sec\n", time_spent_parallel);
    checkMIS(graph, Flags);

    printf("---------------------------\n");
    printf("Serial implementation:\n");
    printf("execution time: %.6f sec\n", time_spent_serial);
    checkMIS(graph, Flags_serial);

    printf("---------------------------\n");
    printf("speedup: %.8fx\n", time_spent_serial / time_spent_parallel);
}