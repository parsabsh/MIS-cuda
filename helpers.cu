#include <stdio.h>
#include "helpers.cuh"

dim3 calculateGridDim(int n) {
    if (n < 65535)
        return dim3(n, 1); 
    return dim3(65535, (n + 65535 - 1) / 65535);
}

void deep_copy_to_device(Graph *graph, Graph **graph_dev) {
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
    *graph_dev = dev_graph;
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
    graph->V[j].Neighbors[graph->V[j].degree] = i;
    graph->V[j].degree = graph->V[j].degree + 1;
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

void print_mis(int* Flags, int n) {
    printf("MIS set: [");
    for (int i = 0; i < n; i++) {
        if (Flags[i] == 1)
            printf(" %d", i);
    }
    printf(" ]\n");
}


void checkMIS(Graph* G, int* Flags) {
    int result = 1;
    for(int i = 0; i < G->n; ++i) {
        if(Flags[i] == 2) {
            int has_ngh = 0;
            for(int j = 0; j < G->V[i].degree; ++j) {
                int ngh = G->V[i].Neighbors[j];
                if (Flags[ngh] == 1) {
                    has_ngh = 1;
                    break;
                }
            }
            if(has_ngh == 0) {
                result = 0;
                break;
            }
        }
    }
    if(result) printf("Correct Maximal Independent Set!\n");
    else printf("Incorrect Maximal Independent Set!\n");
}

int* maximalIndependentSetSerial(Graph* G, int *Flags) {
    Flags = (int*) malloc(G->n * sizeof(int));
    memset(Flags, 0, G->n * sizeof(int));

    for(int i = 0; i < G->n; ++i) {
        int has_ngh_in_mis = 0;
        for(int j = 0; j < G->V[i].degree; ++j) {
            int ngh = G->V[i].Neighbors[j];
            if(Flags[ngh] == 1) {
                has_ngh_in_mis = 1;
                break;
            }
        }
        if(!has_ngh_in_mis) {
            Flags[i] = 1;
        }
        else {
            Flags[i] = 2;
        }
    }
}

__global__ void maximalIndependentSet(const Graph* G, int* Flags, int* V) {
    size_t v = blockIdx.x * gridDim.y + blockIdx.y;

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
    }
}
