#include "helpers.cu"
#include <stdio.h>

int main() {
    // showoff();
    int n = 5;
    Graph* graph = new_graph_host(n, n);
    add_undirected_edge(graph, 0, 1);
    add_undirected_edge(graph, 4, 3);
    add_undirected_edge(graph, 3, 2);
    add_undirected_edge(graph, 1, 3);
    // Graph* g_dev = deep_copy_to_device(g);

    // ----------------Deep Copy----------------------

    Graph *dev_graph;
    cudaMalloc(&dev_graph, sizeof(Graph));
    cudaMemcpy(dev_graph, graph, sizeof(Graph), cudaMemcpyHostToDevice);

    Vertex *v;
    cudaMalloc(&v, graph->n * sizeof(Vertex));

    cudaMemcpy(&(dev_graph->V), &v, sizeof(Vertex *), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < graph->n; i++) {
        cudaMemcpy(&(v[i]), &(graph->V[i]), sizeof(Vertex), cudaMemcpyHostToDevice);
        int *neighbors;
        cudaMalloc(&neighbors, graph->max_degree * sizeof(int));
        cudaMemcpy(neighbors, graph->V[i].Neighbors, graph->max_degree * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&(v[i].Neighbors), &neighbors, sizeof(int *), cudaMemcpyHostToDevice);
    }
    // --------------------------------------


    int* Flags = (int*) malloc(graph->n * sizeof(int));
    // for (int i = 0; i < n; i++)
    // {
    //     Flags[i] = 2;
    // }
    
    memset(Flags, 0, graph->n * sizeof(int));
    int* Flags_dev;
    cudaMalloc(&Flags_dev, graph->n * sizeof(int));
    cudaMemcpy(Flags_dev, Flags, graph->n * sizeof(int), cudaMemcpyHostToDevice);


    int* locks = (int*) malloc(graph->n * sizeof(int));
    memset(locks, 0, graph->n * sizeof(int));
    int* locks_dev;
    cudaMalloc(&locks_dev, graph->n * sizeof(int));
    cudaMemcpy(locks_dev, locks, graph->n * sizeof(int), cudaMemcpyHostToDevice);

    int* lock = (int *) malloc(sizeof(int));
    *lock = 0;
    int *lock_dev;
    cudaMalloc(&lock_dev, sizeof(int));
    cudaMemcpy(locks_dev, locks, sizeof(int), cudaMemcpyHostToDevice);


    // dim3 thread_per_block(1024);
    // dim3 block_per_grid((graph->n + thread_per_block.x - 1) / thread_per_block.x);

    // printf("%d %d\n", thread_per_block, block_per_grid);

    // Lock lock;

    // print_graph_dev<<<block_per_grid, thread_per_block>>>(lock, dev_graph);

    // cudaDeviceSynchronize();
    // print_graph(graph);

    size_t blockSize = 256;
    size_t gridSize = (n + blockSize - 1) / blockSize;

    gridSize = n;
    blockSize = 1;

    // maximalIndependentSetKernel<<<gridSize, blockSize>>>(dev_graph, Flags_dev, locks_dev);
    maximalIndependentSet<<<gridSize, blockSize>>>(dev_graph, lock_dev, Flags_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(Flags, Flags_dev, graph->n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d: %d\n", i, Flags[i]);
    }
}