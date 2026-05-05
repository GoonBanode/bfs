#include <iostream> 
#include <vector> 
#include <queue> 
#include <stack> 
#include <omp.h> 
using namespace std; 
void parallelBFS(vector<vector<int>> &graph, int start) { 
    int n = graph.size(); 
    vector<bool> visited(n, false); 
    queue<int> q; 
    visited[start] = true; 
    q.push(start); 
    cout << "\nParallel BFS Traversal: "; 
    while (!q.empty()) { 
        int size = q.size(); 
        #pragma omp parallel for 
        for (int i = 0; i < size; i++) { 
            int node; 
            #pragma omp critical 
            { 
                node = q.front(); 
                q.pop(); 
                cout << node << " "; 
            } 
            for (int neighbor : graph[node]) { 
                if (!visited[neighbor]) { 
                    #pragma omp critical 
                    { 
                        if (!visited[neighbor]) { 
                            visited[neighbor] = true; 
                            q.push(neighbor); 
                        } 
                    } 
                } 
            } 
        } 
    } 
} 
void parallelDFSUtil(vector<vector<int>> &graph, int node, vector<bool> &visited) { 
    visited[node] = true; 
 
    #pragma omp critical 
    cout << node << " "; 
 
    for (int neighbor : graph[node]) { 
        if (!visited[neighbor]) { 
            #pragma omp task 
            parallelDFSUtil(graph, neighbor, visited); 
        } 
    } 
} 
void parallelDFS(vector<vector<int>> &graph, int start) { 
    int n = graph.size(); 
    vector<bool> visited(n, false); 
    cout << "\nParallel DFS Traversal: "; 
    #pragma omp parallel 
    { 
        #pragma omp single 
        parallelDFSUtil(graph, start, visited); 
    } 
} 
int main() { 
    int vertices, edges; 
    cout << "Enter number of vertices: "; 
    cin >> vertices; 
    vector<vector<int>> graph(vertices); 
    cout << "Enter number of edges: "; 
    cin >> edges; 
    cout << "Enter edges (u v) for undirected graph:\n"; 
    for (int i = 0; i < edges; i++) { 
        int u, v; 
        cin >> u >> v; 
        graph[u].push_back(v); 
        graph[v].push_back(u); 
    } 
    int start; 
    cout << "Enter starting vertex: "; 
    cin >> start; 
    parallelBFS(graph, start); 
    parallelDFS(graph, start); 
    return 0; 
}
