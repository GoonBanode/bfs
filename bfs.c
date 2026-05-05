// Parallel BFS using OpenMP (C90 Compatible)

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX 100

int graph[MAX][MAX];
int visited[MAX];
int queue[MAX];

int main() {
    int n, e;
    int i, j;   // Declare variables at top (C90 style)

    printf("Enter number of vertices: ");
    scanf("%d", &n);

    // Initialize graph and visited array
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
        int u, v;   // allowed (not inside for header)
        scanf("%d %d", &u, &v);
        graph[u][v] = 1;
        graph[v][u] = 1;   // undirected graph
    }

    int start;
    printf("Enter starting vertex: ");
    scanf("%d", &start);

    int front = 0, rear = 0;

    queue[rear++] = start;
    visited[start] = 1;

    printf("\nParallel BFS Traversal:\n");

    while(front < rear) {
        int current = queue[front++];
        printf("%d ", current);

        // Parallel traversal of neighbors
        #pragma omp parallel for
        for(i = 0; i < n; i++) {
            if(graph[current][i] && !visited[i]) {

                #pragma omp critical
                {
                    if(!visited[i]) {
                        visited[i] = 1;
                        queue[rear++] = i;
                    }
                }
            }
        }
    }

    return 0;
}
