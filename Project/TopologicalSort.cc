#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>

#define CMD_LINE_ARGS_NO 3

struct Node {
    int starting;
    int no_of_edges;
    int id;
};


FILE *fp;

void Usage(int argc, char** argv) {
    fprintf(stderr, "Usage: %s <num_threads> <input_file>\n", argv[0]);
}

void Read_Input(int argc, char** argv, int** graph_edges, struct Node** graph_nodes, int* no_of_nodes, int* edge_list_size, int* source, int* num_omp_threads) {
    char* input_f;
    
    if (argc < CMD_LINE_ARGS_NO) {
        Usage(argc, argv);
        exit(0);
    }

    *num_omp_threads = atoi(argv[1]);
    input_f = argv[2];

    printf("Reading File\n");
    // Read in Graph from a file
    fp = fopen(input_f, "r");

    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    fscanf(fp, "%d", no_of_nodes);
    // Allocate memory for graph_nodes
    *graph_nodes = (struct Node*)malloc(*no_of_nodes * sizeof(struct Node));


    int start, edge_no;
    // Initialize the memory
    for (unsigned int i = 0; i < *no_of_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edge_no);
        (*graph_nodes)[i].starting = start;
        (*graph_nodes)[i].no_of_edges = edge_no;
        (*graph_nodes)[i].id = i;
    }

    // Read the source node from the file
    fscanf(fp, "%d", source);
    // Read size of edge list
    fscanf(fp, "%d", edge_list_size);

    // Allocate memory for graph_edges
    *graph_edges = (int*)malloc(*edge_list_size * sizeof(int));

    int id, cost;
    for (int i = 0; i < *edge_list_size; i++) {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        (*graph_edges)[i] = id;
    }

    if (fp)
        fclose(fp);
}


void TopologicalSort(struct Node* graph_nodes, int* graph_edges, int n, int source) {
    int* groups = (int*)malloc(n * sizeof(int));
    std::vector<Node> frontier;
    std::vector<Node> next_frontier;
    int group_id = 0;
    int changed = 1;
    memset(groups, -1, n * sizeof(int));
    frontier.push_back(graph_nodes[source]);

    double start_time, end_time;  
    start_time = omp_get_wtime();  // Record the start time

    while (changed) {
        changed = 0;

        #pragma omp parallel shared(frontier, next_frontier, groups, changed)
        {
            std::vector<Node> private_next_frontier;
            #pragma omp for 
            for (int i = 0; i < frontier.size(); i++) {
                const auto& node = frontier[i];
                if (groups[node.id] == -1) {
                    groups[node.id] = group_id;
                }
                for (int j = node.starting; j < (node.starting + node.no_of_edges); j++) {
                    int x = graph_edges[j];
                    Node dest = graph_nodes[x];
                    if (groups[dest.id] == -1) {
                        private_next_frontier.push_back(graph_nodes[dest.id]);
                        changed = 1;
                    }
                }
            }
            #pragma omp critical
            {
                next_frontier.insert(next_frontier.end(), private_next_frontier.begin(), private_next_frontier.end());
            }
        }

        if (!next_frontier.empty()) {
            group_id += 1;
            frontier = std::move(next_frontier);
            next_frontier.clear();
        }
    } //end of while loop

    end_time = omp_get_wtime();  // Record the end time

    double execution_time = (end_time - start_time);
    printf("Execution time: %f \n", execution_time);
    
    FILE *outp1;
    outp1 = fopen("Results/testingVersion3/graph_1Mnodes.txt", "w");
    for (int i = 0; i < n; ++i) {
          fprintf(outp1, "%d ", groups[i]);
 }
free(groups);
}

int main(int argc, char** argv) {
    int num_threads, no_of_nodes, edge_list_size, source;
    int* graph_edges;
    struct Node* graph_nodes;

    Read_Input(argc, argv, &graph_edges, &graph_nodes, &no_of_nodes, &edge_list_size, &source, &num_threads);

    // int max_threads = omp_get_max_threads();
    //printf("Maximum number of threads: %d\n", max_threads);

    omp_set_num_threads(num_threads);

    TopologicalSort(graph_nodes,graph_edges,no_of_nodes,source);

    // Free allocated memory
    free(graph_edges);
    free(graph_nodes);

    return 0;
}