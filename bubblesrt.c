// Parallel Bubble Sort using OpenMP (Odd-Even Sort)

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Swap function
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Sequential Bubble Sort
void sequentialBubbleSort(int arr[], int n) {
    int i, j;
    for(i = 0; i < n - 1; i++) {
        for(j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort (Odd-Even)
void parallelBubbleSort(int arr[], int n) {
    int i;

    for(i = 0; i < n; i++) {

        int j;

        // Even phase
        #pragma omp parallel for private(j)
        for(j = 0; j < n - 1; j += 2) {
            if(arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }

        // Odd phase
        #pragma omp parallel for private(j)
        for(j = 1; j < n - 1; j += 2) {
            if(arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

// Print array
void printArray(int arr[], int n) {
    int i;
    for(i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int n, i;

    printf("Enter number of elements: ");
    scanf("%d", &n);

    int *arr1 = (int*)malloc(n * sizeof(int));
    int *arr2 = (int*)malloc(n * sizeof(int));

    printf("Enter elements:\n");
    for(i = 0; i < n; i++) {
        scanf("%d", &arr1[i]);
        arr2[i] = arr1[i];
    }

    double start, end, seq_time, par_time;

    // Sequential
    start = omp_get_wtime();
    sequentialBubbleSort(arr1, n);
    end = omp_get_wtime();
    seq_time = end - start;

    // Parallel
    start = omp_get_wtime();
    parallelBubbleSort(arr2, n);
    end = omp_get_wtime();
    par_time = end - start;

    printf("\nSorted Array (Sequential):\n");
    printArray(arr1, n);

    printf("\nSorted Array (Parallel):\n");
    printArray(arr2, n);

    printf("\nExecution Time:\n");
    printf("Sequential Time = %f seconds\n", seq_time);
    printf("Parallel Time   = %f seconds\n", par_time);

    if (par_time > 0)
        printf("\nSpeedup = %f\n", seq_time / par_time);
    else
        printf("\nSpeedup cannot be calculated\n");

    free(arr1);
    free(arr2);

    return 0;
}
