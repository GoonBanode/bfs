(A) VECTOR ADDITION USING CUDA 
#include <iostream> 
#include <cuda_runtime.h> 
using namespace std; 
// Kernel function 
__global__ void addVectors(int *A, int *B, int *C, int n) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) { 
        C[i] = A[i] + B[i]; 
    } 
} 
int main() { 
    int n; 
    cout << "Enter size of vectors: "; 
    cin >> n; 
    int size = n * sizeof(int); 
    int *A, *B, *C; 
    A = (int*)malloc(size); 
    B = (int*)malloc(size); 
    C = (int*)malloc(size); 
    cout << "Enter elements of vector A:\n"; 
    for (int i = 0; i < n; i++) cin >> A[i]; 
    cout << "Enter elements of vector B:\n"; 
    for (int i = 0; i < n; i++) cin >> B[i]; 
    int *d_A, *d_B, *d_C; 
    cudaMalloc(&d_A, size); 
    cudaMalloc(&d_B, size); 
    cudaMalloc(&d_C, size); 
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); 
    int blockSize = 256; 
    int gridSize = (n + blockSize - 1) / blockSize; 
    addVectors<<<gridSize, blockSize>>>(d_A, d_B, d_C, n); 
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); 
    cout << "\nResult Vector:\n"; 
    for (int i = 0; i < n; i++) 
        cout << C[i] << " "; 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
    free(A); free(B); free(C); 
    return 0; 
}



(B) MATRIX MULTIPLICATION USING CUDA 
#include <iostream> 
#include <cuda_runtime.h> 
using namespace std; 
// Kernel 
__global__ void matrixMul(int *A, int *B, int *C, int n) { 
    int row = threadIdx.y; 
    int col = threadIdx.x; 
    int sum = 0; 
    for (int k = 0; k < n; k++) { 
        sum += A[row * n + k] * B[k * n + col]; 
    } 
    C[row * n + col] = sum; 
} 
int main() { 
    int n; 
    cout << "Enter size of matrix (n x n): "; 
    cin >> n; 
    int size = n * n * sizeof(int); 
    int A[MAX][MAX], B[MAX][MAX], C[MAX][MAX]; 
    cout << "Enter Matrix A:\n"; 
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            cin >> A[i][j]; 
    cout << "Enter Matrix B:\n"; 
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            cin >> B[i][j]; 
    int *d_A, *d_B, *d_C; 
    cudaMalloc(&d_A, size); 
    cudaMalloc(&d_B, size); 
    cudaMalloc(&d_C, size); 
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); 
    dim3 threads(n, n); 
    matrixMul<<<1, threads>>>(d_A, d_B, d_C, n); 
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); 
    cout << "\nResult Matrix:\n"; 
    for (int i = 0; i < n; i++) { 
        for (int j = 0; j < n; j++) 
            cout << C[i][j] << " "; 
        cout << endl; 
    } 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
    return 0; 
}
