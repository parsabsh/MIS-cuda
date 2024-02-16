typedef struct Vertex {
    int degree;
    int *Neighbors;
} Vertex;

typedef struct Graph {
    int n;
    int max_degree;
    Vertex *V;
} Graph;

// add E random undirected edges to the graph
void add_random_edges(Graph *graph, int E);

// construct a new graph with n vertices and max_degree
// NOTE: max_degree is by default eqaul to n. But in sparse graphs,
// knowing the max_degree will increase space efficiency.
Graph* new_graph_host(int n, int max_degree);

// calculate the grid dimensions based on n and the maximum number of blocks in each grid dimension
dim3 calculateGridDim(int n);

// perform a deep copy of the data structure Graph, from host to device and set graph_dev pointer to it
void deep_copy(Graph *graph, Graph **graph_dev);

// add an undirected edge between i and j to the graph
void add_undirected_edge(Graph *graph, int i, int j);

// print a graph
void print_graph(Graph graph);

// check for the correctness of MIS
void checkMIS(Graph* G, int* Flags);

// serial implementation of MIS algorithm
int* maximalIndependentSetSerial(Graph* G);

// print a Maximal Independent Set
void print_mis(int* Flags, int n);

// CUDA kernel for the maximalIndependentSet algorithm
__global__ void maximalIndependentSet(const Graph* G, int* Flags, int* V);

// CUDA kernel for printing a graph on the device
__global__ void print_graph_dev(Graph* graph);
