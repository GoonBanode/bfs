// Parallel Reduction: Min, Max, Sum, Average (Lab Safe)

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

int main() {
    int n, i;

    printf("Enter number of elements: ");
    scanf("%d", &n);

    int *arr = (int*)malloc(n * sizeof(int));

    printf("Enter elements:\n");
    for(i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    // -------- Sequential --------
    int seq_min = INT_MAX, seq_max = INT_MIN;
    long long seq_sum = 0;

    double start = omp_get_wtime();

    for(i = 0; i < n; i++) {
        if(arr[i] < seq_min) seq_min = arr[i];
        if(arr[i] > seq_max) seq_max = arr[i];
        seq_sum += arr[i];
    }

    double end = omp_get_wtime();
    double seq_time = end - start;

    double seq_avg = (double)seq_sum / n;

    // -------- Parallel Reduction --------
    int par_min = INT_MAX, par_max = INT_MIN;
    long long par_sum = 0;

    start = omp_get_wtime();

    #pragma omp parallel for reduction(min:par_min) reduction(max:par_max) reduction(+:par_sum)
    for(i = 0; i < n; i++) {
        if(arr[i] < par_min) par_min = arr[i];
        if(arr[i] > par_max) par_max = arr[i];
        par_sum += arr[i];
    }

    end = omp_get_wtime();
    double par_time = end - start;

    double par_avg = (double)par_sum / n;

    // -------- Output --------
    printf("\n--- Sequential Results ---\n");
    printf("Min = %d\n", seq_min);
    printf("Max = %d\n", seq_max);
    printf("Sum = %lld\n", seq_sum);
    printf("Average = %.2f\n", seq_avg);
    printf("Time = %f seconds\n", seq_time);

    printf("\n--- Parallel Results ---\n");
    printf("Min = %d\n", par_min);
    printf("Max = %d\n", par_max);
    printf("Sum = %lld\n", par_sum);
    printf("Average = %.2f\n", par_avg);
    printf("Time = %f seconds\n", par_time);

    if (par_time > 0)
        printf("\nSpeedup = %f\n", seq_time / par_time);

    free(arr);
    return 0;
}
