#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

//change to reflect the number of inputs in your program.
#define CMD_LINE_ARGS_NO 3

FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}

void Read_Input(int argc, char** argv, int* graph_edges, Node *graph_nodes, int &no_of_nodes, int &edge_list_size, int &source, int &num_omp_threads) {

    char *input_f;
	
	if(argc<CMD_LINE_ARGS_NO){
	Usage(argc, argv);
	exit(0);
	}
    
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	//int source = 0;

	fscanf(fp,"%d",&no_of_nodes);
   
	// allocate host memory

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		graph_nodes[i].starting = start;
		graph_nodes[i].no_of_edges = edgeno;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);

	//read size of edge list
	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

}
