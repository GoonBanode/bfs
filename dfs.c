// Depth First Search using OpenMP (Parallel Version - Lab Safe)

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX 100

int graph[MAX][MAX];
int visited[MAX];
int n;

// Parallel DFS function
void parallelDFS(int node) {

    int i;

    // Critical section to safely visit node
    #pragma omp critical
    {
        if (visited[node])
            return;
        visited[node] = 1;
        printf("%d ", node);
    }

    // Parallel loop
    #pragma omp parallel for
    for (i = 0; i < n; i++) {

        int shouldVisit = 0;

        // Check safely
        #pragma omp critical
        {
            if (graph[node][i] && !visited[i]) {
                shouldVisit = 1;
            }
        }

        if (shouldVisit) {
            parallelDFS(i);
        }
    }
}

int main() {
    int i, j, e;

    printf("Enter number of vertices: ");
    scanf("%d", &n);

    // Initialize graph and visited
    for(i = 0; i < n; i++) {
        visited[i] = 0;
        for(j = 0; j < n; j++) {
            graph[i][j] = 0;
        }
    }

    printf("Enter number of edges: ");
    scanf("%d", &e);

    printf("Enter edges (u v):\n");
    for(i = 0; i < e; i++) {
        int u, v;
        scanf("%d %d", &u, &v);

        graph[u][v] = 1;
        graph[v][u] = 1; // Undirected graph
    }

    int start;
    printf("Enter starting vertex: ");
    scanf("%d", &start);

    printf("\nParallel DFS Traversal:\n");

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelDFS(start);
        }
    }

    return 0;
}
