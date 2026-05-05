#include <iostream> 
#include <vector> 
#include <omp.h> 
using namespace std; 
void bubbleSort(vector<int> &arr) { 
    int n = arr.size(); 
    for (int i = 0; i < n - 1; i++) { 
        for (int j = 0; j < n - i - 1; j++) { 
            if (arr[j] > arr[j + 1]) 
                swap(arr[j], arr[j + 1]); 
        } 
    } 
} 
void parallelBubbleSort(vector<int> &arr) { 
    int n = arr.size(); 
    for (int i = 0; i < n; i++) { 
        if (i % 2 == 0) { 
            #pragma omp parallel for 
            for (int j = 0; j < n - 1; j += 2) { 
                if (arr[j] > arr[j + 1]) 
                    swap(arr[j], arr[j + 1]); 
            } 
        } else { 
            #pragma omp parallel for 
            for (int j = 1; j < n - 1; j += 2) { 
                if (arr[j] > arr[j + 1]) 
                    swap(arr[j], arr[j + 1]); 
            } 
        } 
    } 
} 
void merge(vector<int> &arr, int l, int m, int r) { 
    vector<int> temp(r - l + 1); 
    int i = l, j = m + 1, k = 0; 
    while (i <= m && j <= r) { 
        if (arr[i] < arr[j]) 
            temp[k++] = arr[i++]; 
        else 
            temp[k++] = arr[j++]; 
    } 
    while (i <= m) temp[k++] = arr[i++]; 
    while (j <= r) temp[k++] = arr[j++]; 
    for (int i = l; i <= r; i++) 
        arr[i] = temp[i - l]; 
} 
void mergeSort(vector<int> &arr, int l, int r) { 
    if (l < r) { 
        int m = (l + r) / 2; 
        mergeSort(arr, l, m); 
        mergeSort(arr, m + 1, r); 
        merge(arr, l, m, r); 
    } 
} 
void parallelMergeSort(vector<int> &arr, int l, int r) { 
    if (l < r) { 
        int m = (l + r) / 2; 
        #pragma omp parallel sections 
        { 
            #pragma omp section 
            parallelMergeSort(arr, l, m); 
            #pragma omp section 
            parallelMergeSort(arr, m + 1, r); 
        } 
        merge(arr, l, m, r); 
    } 
} 
int main() { 
    int n; 
    cout << "Enter number of elements: "; 
    cin >> n; 
    vector<int> arr(n); 
    cout << "Enter elements:\n"; 
    for (int i = 0; i < n; i++) 
        cin >> arr[i]; 
    vector<int> arr1 = arr, arr2 = arr, arr3 = arr, arr4 = arr; 
    double start, end; 
    start = omp_get_wtime(); 
    bubbleSort(arr1); 
    end = omp_get_wtime(); 
    cout << "\nSequential Bubble Sort Time: " << (end - start); 
    start = omp_get_wtime(); 
    parallelBubbleSort(arr2); 
    end = omp_get_wtime(); 
    cout << "\nParallel Bubble Sort Time: " << (end - start); 
    start = omp_get_wtime(); 
    mergeSort(arr3, 0, n - 1); 
    end = omp_get_wtime(); 
    cout << "\nSequential Merge Sort Time: " << (end - start); 
    start = omp_get_wtime(); 
    parallelMergeSort(arr4, 0, n - 1); 
    end = omp_get_wtime(); 
    cout << "\nParallel Merge Sort Time: " << (end - start); 
    cout << "\n\nSorted Array:\n"; 
    for (int x : arr4) cout << x << " "; 
    return 0; 
}
