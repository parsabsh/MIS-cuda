#include "helpers.cuh"
#include <stdio.h>

int main() {
    int n = 5;
    Graph* graph = new_graph_host(n, n);
    add_undirected_edge(graph, 0, 1);
    add_undirected_edge(graph, 4, 3);
    add_undirected_edge(graph, 3, 2);
    add_undirected_edge(graph, 1, 3);

    Graph *graph_dev;
    deep_copy_to_device(graph, &graph_dev);

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

    maximalIndependentSet<<<blocksPerGrid, threadsPerBlock>>>(graph_dev, Flags_dev, locks_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(Flags, Flags_dev, graph->n * sizeof(int), cudaMemcpyDeviceToHost);

    print_mis(Flags, graph->n);
}