#include <stdio.h>
// #include "helpers.h"

// struct Lock {

//     int *d_state;

//     // --- Constructor
//     Lock(void) {
//         int h_state = 0;                                        // --- Host side lock state initializer
//         cudaMalloc((void **)&d_state, sizeof(int));  // --- Allocate device side lock state
//         cudaMemcpy(d_state, &h_state, sizeof(int), cudaMemcpyHostToDevice); // --- Initialize device side lock state
//     }

//     // --- Destructor
//     __host__ __device__ ~Lock(void) { 
// #if !defined(__CUDACC__)
//         cudaFree(d_state); 
// #else

// #endif  
//     }

//     // --- Lock function
//     __device__ void lock(void) { while (atomicCAS(d_state, 0, 1) != 0); }

//     // --- Unlock function
//     __device__ void unlock(void) { atomicExch(d_state, 0); }
// };

typedef struct Vertex {
    int degree;
    int *Neighbors;
} Vertex;

typedef struct Graph {
    int n;
    int max_degree;
    Vertex *V;
} Graph;

Graph* deep_copy_to_device(Graph *graph) {
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
        printf("hello %d\n", i);
    }
    return dev_graph;
}


Graph* new_graph_host(int n, int max_degree) {
    Graph *graph = (Graph *) malloc(sizeof(Graph));
    graph->n = n;
    graph->max_degree = max_degree;
    graph->V = (Vertex *) malloc(n * sizeof(Vertex));
    for (int i = 0; i < n; i++) {
        graph->V[i].degree = 0;
        graph->V[i].Neighbors = (int *) malloc(max_degree * sizeof(int));
    }
    return graph;
}

void add_undirected_edge(Graph *graph, int i, int j) {
    graph->V[i].Neighbors[graph->V[i].degree] = j;
    graph->V[i].degree = graph->V[i].degree + 1;
    // graph->V[i].Neighbors = (int *) realloc(graph->V[j].Neighbors, graph->V[j].degree * sizeof(int));
    graph->V[j].Neighbors[graph->V[j].degree] = i;
    graph->V[j].degree = graph->V[j].degree + 1;
    // graph->V[j].Neighbors = (int *) realloc(graph->V[j].Neighbors, graph->V[j].degree * sizeof(int));
}

__global__ void print_graph_dev(Graph* graph) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < graph->n) {
        Vertex v = graph->V[tid];
        printf("Vertex %d with degree %d: [ ", tid, v.degree);
        for (int j = 0; j < v.degree; j++) {
            printf("%d ", v.Neighbors[j]);
        }
        printf("]\n");
    }
}

// __global__ void print_graph_dev(Lock lock, Graph* graph) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < graph->n) {
//         Vertex v = graph->V[tid];
//         lock.lock();
//         printf("Vertex %d with degree %d: [ ", tid, v.degree);
//         for (int j = 0; j < v.degree; j++) {
//             printf("%d ", v.Neighbors[j]);
//         }
//         printf("]\n");
//         lock.unlock();
//     }
// }

__global__ void print_graph_dev(Graph* graph, unsigned int* locks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < graph->n) {
        Vertex v = graph->V[tid];
        bool leaveLoop = false;
        while (!leaveLoop) {
            if (atomicExch(&(locks[tid]), 1u) == 0u) {
                printf("Vertex %d with degree %d: [ ", tid, v.degree);
                for (int j = 0; j < v.degree; j++) {
                    printf("%d ", v.Neighbors[j]);
                }
                printf("]\n");
                leaveLoop = true;
                atomicExch(&(locks[tid]),0u);
            }
        } 
        
    }
}

__global__ void k_testLocking(unsigned int* locks, int n) {
    int id = threadIdx.x % n;
    bool leaveLoop = false;
    while (!leaveLoop) {
        if (atomicExch(&(locks[id]), 1u) == 0u) {
            //critical section
            leaveLoop = true;
            atomicExch(&(locks[id]),0u);
        }
    } 
}

void print_graph(Graph graph) {
    for (int i = 0; i < graph.n; i++) {
        Vertex v = graph.V[i];
        printf("Vertex %d with degree %d: [ ", i, v.degree);
        for (int j = 0; j < v.degree; j++) {
            printf("%d ", v.Neighbors[j]);
        }
        printf("]\n");
    }
}

// CUDA kernel for the maximalIndependentSet algorithm
__global__ void maximalIndependentSetKernel(const Graph* G, int* Flags, int* V) {
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;

    while (v < G->n) {
        if (Flags[v]) break;

        if (atomicCAS(&V[v], 0, 1) == 0) {
            size_t k = 0;
            for (size_t j = 0; j < G->V[v].degree; j++) {
                int ngh = G->V[v].Neighbors[j];
                if (Flags[ngh] == 2 || atomicCAS(&V[ngh], 0, 1) == 0) {
                    k++;
                } else {
                    break;
                }
            }
            if (k == G->V[v].degree) {
                printf("hi %d\n",v);
                // Win on self and neighbors, fill flags
                Flags[v] = 1;
                for (size_t j = 0; j < G->V[v].degree; j++) {
                    int ngh = G->V[v].Neighbors[j];
                    if (Flags[ngh] != 2) {
                        Flags[ngh] = 2;
                    }
                }
            } else {
                // Lose, reset V values up to the point where it lost
                V[v] = 0;
                for (size_t j = 0; j < k; j++) {
                    int ngh = G->V[v].Neighbors[j];
                    if (Flags[ngh] != 2) {
                        V[ngh] = 0;
                    }
                }
            }
        } 
        // v += gridDim.x * blockDim.x;
        // printf("%d : %d\n",v, Flags[v]);
    }
}


__global__ void maximalIndependentSet(const Graph* G, int* lock, int* Flags) {
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;

    while (atomicCAS(lock, 0, 1) != 0);
    printf("hi %d\n", v);
    bool f = true;
    for (size_t i = 0; i < G->V[v].degree; i++)
    {
        int ngh = G->V->Neighbors[i];
        if (Flags[ngh] == 1)
            f = false; 
    }
    if (!f)
        Flags[v] = 0;
    
    atomicExch(lock, 0);
    // *lock = 0;
}

__global__ void maximalIndependentSet(const Graph* G, int* Flags) {
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;

    // while (atomicCAS(lock, 0, 1) != 0);
    // printf("hi %d\n", v);
    bool f = true;
    for (size_t i = 0; i < G->V[v].degree; i++)
    {
        int ngh = G->V->Neighbors[i];
        if (Flags[ngh] == 1)
            f = false; 
    }
    if (!f)
        Flags[v] = 1;
    
    // atomicExch(lock, 0);
    // *lock = 0;
}

// __global__ void find_indepenedent_set(const Graph* G, int* Flags, )

