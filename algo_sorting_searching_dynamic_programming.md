# Sort, Search, Recursion, and Dynamic Programming
## 数据结构和算法的八股文通常包括以下几个方面的问题：

1. 数据结构基础：包括数组、链表、栈、队列、哈希表、树（包括二叉树、二叉搜索树、平衡树、红黑树等）、图（包括有向图、无向图、权重图等）等基础知识。

2. 基础算法：包括排序算法（冒泡排序、选择排序、插入排序、快速排序、归并排序、堆排序等）、查找算法（二分查找、深度优先搜索、广度优先搜索等）、动态规划、贪心算法、分治算法等。

3. 高级数据结构和算法：包括B树、B+树、跳表、布隆过滤器、LRU缓存、一致性哈希等。

4. 算法设计技巧：包括递归、迭代、双指针、滑动窗口、位运算等。

5. 算法复杂度分析：理解并能够计算时间复杂度和空间复杂度。

6. 数据结构和算法的应用：如何在实际问题中选择和使用合适的数据结构和算法。

7. 常见的编程题：例如LeetCode、剑指Offer等上的题目，以及它们的解题思路和代码实现。

## Question 1: Merge Sort (O(n log n))
#### 题目描述
merge sort. 归并排序（Merge sort）是一种分治算法，它是采用分治法（Divide and Conquer）的一个非常典型的应用。归并排序的思想就是先递归分解数组，再合并数组。
- Time Complexity: O(nlogn)
- Space Complexity: O(n)

- 示例：
```
输入：[5, 2, 9, 1, 5, 6]
输出：[1, 2, 5, 5, 6, 9]
```

#### 解决思路
归并排序的基本思想是：先递归分解数组，再合并数组。

- 算法步骤
    - 将数组分解成两个长度相等的子数组；
    - 递归分解子数组，直到子数组长度为1；
    - 将两个子数组合并成一个有序数组；
    - 重复步骤2~3。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

void merge(std::vector<int>& nums, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    std::vector<int> L(n1);
    std::vector<int> R(n2);
    for (int i = 0; i < n1; i++) {
        L[i] = nums[left + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = nums[mid + 1 + j];
    }
    int i = 0;
    int j = 0;
    int k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            nums[k] = L[i];
            i++;
        } else {
            nums[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        nums[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        nums[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& nums, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(nums, left, mid);
        mergeSort(nums, mid + 1, right);
        merge(nums, left, mid, right);
    }
}

int main() {
    std::vector<int> nums = {5, 2, 9, 1, 5, 6};
    mergeSort(nums, 0, nums.size() - 1);
    for (int i = 0; i < nums.size(); i++) {
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

## Question 2: Merge Sort 归并排序
#### 归并排序
归并排序是一种分治算法，它将输入数组分成两半，递归地对这两半进行排序，然后将它们合并。合并操作涉及比较两半中的元素并按顺序放置它们。归并排序的时间复杂度为O(n log n)，并且是稳定的，这意味着它保留了相等元素的相对顺序。
#### C++ Implementation
```cpp
#include <iostream>
#include <vector>

template <typename T>
void merge_sort(std::vector<T>& arr) {
    if (arr.size() > 1) {
        size_t mid = arr.size() / 2;
        std::vector<T> left_half(arr.begin(), arr.begin() + mid);
        std::vector<T> right_half(arr.begin() + mid, arr.end());

        merge_sort(left_half);
        merge_sort(right_half);

        size_t i = 0, j = 0, k = 0;

        while (i < left_half.size() && j < right_half.size()) {
            if (left_half[i] < right_half[j]) {
                arr[k] = left_half[i];
                ++i;
            } else {
                arr[k] = right_half[j];
                ++j;
            }
            ++k;
        }

        while (i < left_half.size()) {
            arr[k] = left_half[i];
            ++i;
            ++k;
        }

        while (j < right_half.size()) {
            arr[k] = right_half[j];
            ++j;
            ++k;
        }
    }
}

int main() {
    std::vector<int> arr = {38, 27, 43, 3, 9, 82, 10};
    merge_sort(arr);
    for (const auto& num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // Output: 3 9 10 27 38 43 82
    return 0;
}

```
## Question 3: Quick Sort (O(n log n))
#### 题目描述
quick sort. 快速排序（Quicksort）是对冒泡排序的一种改进。快速排序由 C. A. R. Hoare 在1960年提出。它的基本思想是：通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。
- Time Complexity: O(nlogn)
- Space Complexity: O(logn)

- 示例：
```
输入：[5, 2, 9, 1, 5, 6]
输出：[1, 2, 5, 5, 6, 9]
```

#### 解决思路
快速排序的基本思想是：通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。

- 算法步骤
    - 从数列中挑出一个元素，称为“基准”（pivot）；
    - 重新排序数列，所有比基准值小的元素摆放在基准前面，所有比基准值大的元素摆在基准的后面（相同的数可以到任一边）。在这个分区结束之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
    - 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

int partition(std::vector<int>& nums, int low, int high) {
    int pivot = nums[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (nums[j] < pivot) {
            i++;
            std::swap(nums[i], nums[j]);
        }
    }
    std::swap(nums[i + 1], nums[high]);
    return i + 1;
}

void quickSort(std::vector<int>& nums, int low, int high) {
    if (low < high) {
        int pi = partition(nums, low, high);
        quickSort(nums, low, pi - 1);
        quickSort(nums, pi + 1, high);
    }
}

int main() {
    std::vector<int> nums = {5, 2, 9, 1, 5, 6};
    quickSort(nums, 0, nums.size() - 1);
    for (int i = 0; i < nums.size(); i++) {
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```
## Question 4: Heap Sort (O(n log n))
#### 题目描述
heap sort. 堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点。

- Time Complexity: O(nlogn)
- Space Complexity: O(1)

- 示例：
```
输入：[5, 2, 9, 1, 5, 6]
输出：[1, 2, 5, 5, 6, 9]
```

#### 解决思路
堆排序的基本思想是：利用堆这种数据结构所设计的一种排序算法。

- 算法步骤
    - 创建一个堆 H[0……n-1]；
    - 把堆首（最大值）和堆尾互换；
    - 把堆的尺寸缩小 1，并调用 shift_down(0)，目的是把新的数组顶端数据调整到相应位置；
    - 重复步骤2，直到堆的尺寸为 1。

#### C++ 代码
```cpp
#include <iostream>

void heapify(int arr[], int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && arr[l] > arr[largest]) {
        largest = l;
    }
    if (r < n && arr[r] > arr[largest]) {
        largest = r;
    }
    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
    for (int i = n - 1; i > 0; i--) {
        std::swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

int main() {
    int arr[] = {5, 2, 9, 1, 5, 6};
    int n = sizeof(arr) / sizeof(arr[0]);
    heapSort(arr, n);
    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

## Question 5: Bubble Sort (O(n^2))
#### 题目描述
bubble sort. 冒泡排序是一种简单的排序算法。它重复地遍历要排序的列表，一次比较两个元素，如果它们的顺序错误就把它们交换过来。遍历列表的工作是重复地进行直到没有再需要交换，也就是说该列表已经排序完成。这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端。
- Time Complexity: O(n^2)
- Space Complexity: O(1)

- 示例：
```
输入：[5, 2, 9, 1, 5, 6]
输出：[1, 2, 5, 5, 6, 9]
```
#### 解决思路
冒泡排序的基本思想是：通过对待排序序列从前向后（从下标较小的元素开始），依次比较相邻元素的值，若发现逆序则交换，使值较大的元素逐渐从前向后移动，就像水底下的气泡一样逐渐向上冒。因此称为冒泡排序。

- 算法步骤
    - 比较相邻的元素。如果第一个比第二个大，就交换它们两个。
    - 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
    - 针对所有的元素重复以上的步骤，除了最后一个。
    - 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

void bubbleSort(std::vector<int>& nums) {
    int n = nums.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (nums[j] > nums[j + 1]) {
                std::swap(nums[j], nums[j + 1]);
            }
        }
    }
}

int main() {
    std::vector<int> nums = {5, 2, 9, 1, 5, 6};
    bubbleSort(nums);
    for (int i = 0; i < nums.size(); i++) {
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

## Question 6: Selection Sort (O(n^2))
#### 题目描述
selection sort. 选择排序（Selection sort）是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。
- Time Complexity: O(n^2)
- Space Complexity: O(1)

- 示例：
```
输入：[5, 2, 9, 1, 5, 6]
输出：[1, 2, 5, 5, 6, 9]
```
#### 解决思路
选择排序的基本思想是：首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

- 算法步骤
    - 初始状态：无序区为R[1..n]，有序区为空；
    - 第i趟排序(i=1,2,3...n-1)开始时，当前有序区和无序区分别为R[1..i-1]和R[i..n]。该趟排序从当前无序区中选出关键字最小的记录 R[k]，将它与无序区的第1个记录R[i]交换，使R[1..i]和R[i+1..n]分别变为记录个数增加1个的新有序区和记录个数减少1个的新无序区；
    - n-1趟结束，数组有序化了。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

void selectionSort(std::vector<int>& nums) {
    int n = nums.size();
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (nums[j] < nums[min_index]) {
                min_index = j;
            }
        }
        std::swap(nums[i], nums[min_index]);
    }
}

int main() {
    std::vector<int> nums = {5, 2, 9, 1, 5, 6};
    selectionSort(nums);
    for (int i = 0; i < nums.size(); i++) {
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```


## Question 7: Insertion Sort (O(n^2))
#### 题目描述
insertion sort. 插入排序（Insertion sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
- Time Complexity: O(n^2)
- Space Complexity: O(1)

- 示例：
```
输入：[5, 2, 9, 1, 5, 6]
输出：[1, 2, 5, 5, 6, 9]
```

#### 解决思路
插入排序的基本思想是：通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。

- 算法步骤
    - 从第一个元素开始，该元素可以认为已经被排序；
    - 取出下一个元素，在已经排序的元素序列中从后向前扫描；
    - 如果该元素（已排序）大于新元素，将该元素移到下一位置；
    - 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；
    - 将新元素插入到该位置后；
    - 重复步骤2~5。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

void insertionSort(std::vector<int>& nums) {
    int n = nums.size();
    for (int i = 1; i < n; i++) {
        int key = nums[i];
        int j = i - 1;
        while (j >= 0 && nums[j] > key) {
            nums[j + 1] = nums[j];
            j--;
        }
        nums[j + 1] = key;
    }
}

int main() {
    std::vector<int> nums = {5, 2, 9, 1, 5, 6};
    insertionSort(nums);
    for (int i = 0; i < nums.size(); i++) {
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```
## Question 8: Fibonacci Sequence (斐波那契数列) (递归)
#### 题目描述
斐波那契数列是一个非常经典的数列，定义如下：
\[F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)\]

#### 解决思路
1. **递归方法**：这是一种直观的方法，但由于会重复计算很多次同样的值，所以效率较低，时间复杂度为\(O(2^n)\)。
2. **动态规划或迭代方法**：通过从底向上计算并存储已经计算过的值，避免重复计算，将时间复杂度降低到\(O(n)\)。

#### C++ 代码
In the dynamic programming approach, the table can be used to store the calculated values and avoid redundant calculations and is implemented as follows:

if in C++ (consider efficency, smart pointers, STL, template, lamda function, avoid copy, move semantics, opraotr overloading, etc.):
```cpp
#include <iostream>
#include <vector>

int fibonacci(int n) {
    std::vector<int> fib = {0, 1};
    for (int i = 2; i <= n; ++i) {
        fib.push_back(fib[i - 1] + fib[i - 2]);
    }
    return fib[n];
}

int main() {
    int n = 10;
    std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    return 0;
}
```

## Question 9: Factorial 阶乘 (递归) (迭代)
#### 题目描述
factorial 阶乘，通常用于计算排列组合的数量。阶乘的定义如下：

- 0! = 1
- n! = n * (n - 1) * (n - 2) * ... * 1

#### 解决思路
阶乘的计算可以使用递归或迭代的方式。在这里，我将详细说明递归和迭代的解决思路。

- 递归法解决思路
在递归法中，我们首先定义一个递归函数 factorial，它的作用是计算 n 的阶乘。在递归函数中，我们首先判断特殊情况，如果 n 等于 0，直接返回 1。否则，我们返回 n 乘以 n - 1 的阶乘。

#### C++ 代码
```cpp
#include <iostream>

int factorial(int n) {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

int main() {
    int n = 5;
    std::cout << factorial(n) << std::endl;
    return 0;
}
```

## Question 10: Linear Search (线性搜索) (O(n))
#### 题目描述
线性搜索是一种简单的搜索算法，它顺序检查数组的每个元素，直到找到匹配的值或遍历整个数组。它的时间复杂度为O(n)，适用于小型数据集或未排序的数组。该算法将每个元素与目标值进行比较，并在找到匹配时返回索引。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

template <typename T>
int linear_search(const std::vector<T>& arr, const T& target) {
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

int main() {
    std::vector<int> arr = {3, 9, 10, 27, 38, 43, 82};
    int target = 27;
    int result = linear_search(arr, target);
    std::cout << result << std::endl;  // Output: 3
    return 0;
}
```

## Question 11: Binary Search (二分查找) (O(log n))
#### 解决思路
二分查找是一种分治算法，它通过反复将搜索区间划分为两半来在排序数组中搜索目标值。它的时间复杂度为O(log n)，对于大型数据集非常高效。该算法将目标值与数组的中间元素进行比较，并根据比较继续在适当的半边进行搜索。 如果数组未排序，则需要在应用二分查找之前对其进行排序 O(n log n)。
#### C++ 代码
```cpp
#include <iostream>
#include <vector>

template <typename T>
int binary_search(const std::vector<T>& arr, const T& target) {
    int low = 0;
    int high = arr.size() - 1;

    while (low <= high) {
        int mid = (low + high) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

int main() {
    std::vector<int> arr = {3, 9, 10, 27, 38, 43, 82};
    int target = 27;
    int result = binary_search(arr, target);
    std::cout << result << std::endl;  // Output: 3
    return 0;
}
```

## Question 12: 三数之和 (Sort + Binary Search)
#### 题目描述
三数之和。给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

- 注意：答案中不可以包含重复的三元组。

- 示例：
```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
```

#### 解决思路
- 首先对数组进行排序，这样可以方便地处理重复元素，并且可以利用有序数组的性质。
- 遍历排序后的数组，使用一个指针固定一个数，另外两个指针分别指向固定数的下一个数和数组末尾。
- 在固定数后面的子数组中使用双指针技巧寻找另外两个数，使三数之和等于 0。
- 如果找到符合条件的三元组，将其加入结果集中。
- 注意处理重复元素的情况，避免重复计算。

详细步骤

- 对数组 nums 进行排序。
- 初始化结果集 result 为空列表。
- 遍历排序后的数组 nums：
    - 初始化左指针 left 指向当前数的下一个位置，右指针 right 指向数组末尾。
    - 当 left 指针小于 right 指针时，执行以下步骤：
    - 计算当前三个数的和 sum。
    - 如果 sum == 0，将当前三个数加入结果集。
    - 如果 sum < 0，说明需要增大和，将 left 指针右移一位。
    - 如果 sum > 0，说明需要减小和，将 right 指针左移一位。
    - 在移动指针时，注意避免重复元素的情况。
    - 在当前数之后的子数组中使用双指针技巧：
- 举例来说，考虑输入数组 nums = [-1,0,1,2,-1,-4]：
    - 排序后的数组为 [-4, -1, -1, 0, 1, 2]。
    - 遍历数组：
        - 左指针指向 1，右指针指向 2，三数之和为 0 + 1 + 2 = 3，大于 0，右指针左移。
        ...
        - 左指针指向 0，右指针指向 2，三数之和为 -1 + 0 + 2 = 1，大于 0，右指针左移。
        ...
        - 左指针指向 -1，右指针指向 2，三数之和为 -4 + (-1) + 2 = -3，小于 0，左指针右移。
        - 左指针指向 -1，右指针指向 2，三数之和为 -4 + (-1) + 2 = -3，小于 0，左指针右移。
        ...
        - 当固定数为 -4 时：
        - 当固定数为 -1 时：
        - 当固定数为 0 时：
    - 遍历完成后，得到结果集 result，其中 [-1, -1, 2] 和 [-1, 0, 1] 是不重复的三元组。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
    std::vector<std::vector<int>> result;
    int n = nums.size();
    std::sort(nums.begin(), nums.end());

    for (int i = 0; i < n; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int left = i + 1;
        int right = n - 1;

        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum == 0) {
                result.push_back({nums[i], nums[left], nums[right]});
                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;
                left++;
                right--;
            } else if (sum < 0) {
                left++;
            } else {
                right--;
            }
        }
    }

    return result;
}

int main() {
    std::vector<int> nums = {-1,0,1,2,-1,-4};
    std::vector<std::vector<int>> result = threeSum(nums);

    for (const auto& triplet : result) {
        std::cout << "[";
        for (int num : triplet) {
            std::cout << num << ",";
        }
        std::cout << "]" << std::endl;
    }
    return 0;
}
```

## Question 13: Depth First Search (O(V + E))
#### 题目描述
Depth-First Search (DFS). 深度优先搜索（Depth-First Search）是一种用于遍历或搜索树或图的算法。沿着树的深度遍历树的节点，尽可能深的搜索树的分支。当节点 v 的所在边都己被探寻过，搜索将回溯到发现节点 v 的那条边的起始节点。这一过程一直进行到已发现从源节点可达的所有节点为止。如果还存在未被发现的节点，则选择其中一个作为源节点并重复以上过程。整个进程反复进行直到所有节点都被访问为止。
- Time Complexity: O(V + E): where V is number of vertices and E is number of edges, 
- Space Complexity: O(V): for storing the stack

- 示例：
```
输入：graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
输出：[0, 1, 2, 3]
```

#### 解决思路
深度优先搜索的基本思想是：沿着树的深度遍历树的节点，尽可能深的搜索树的分支。

- 算法步骤
    - 从源节点开始遍历，将源节点标记为已访问；
    - 对于源节点的所有邻接节点，如果未访问，则递归访问；
    - 重复步骤2，直到所有节点都被访问。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

void dfs(int node, std::unordered_map<int, std::vector<int>>& graph, std::unordered_set<int>& visited, std::vector<int>& result) {
    visited.insert(node);te
    result.push_back(node);
    for (int neighbor : graph[node]) {
        if (visited.find(neighbor) == visited.end()) {
            dfs(neighbor, graph, visited, result);
        }
    }
}

std::vector<int> depthFirstSearch(std::unordered_map<int, std::vector<int>>& graph) {
    std::unordered_set<int> visited;
    std::vector<int> result;
    for (auto& [node, neighbors] : graph) {
        if (visited.find(node) == visited.end()) {
            dfs(node, graph, visited, result);
        }
    }
    return result;
}

int main() {
    std::unordered_map<int, std::vector<int>> graph = {
        {0, {1, 2}},
        {1, {2}},
        {2, {0, 3}},
        {3, {3}}
    };
    std::vector<int> result = depthFirstSearch(graph);
    for (int node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

## Question 14: Breath First Search (O(V + E))
#### 题目描述
Breadth-First Search (BFS). 广度优先搜索（Breadth-First Search）是一种用于遍历或搜索树或图的算法。从根节点开始，沿着树的宽度遍历树的节点。如果所有节点都在同一层，那么就按照它们在树中出现的顺序来进行遍历。广度优先搜索的实现一般采用open-closed表。

- Time Complexity: O(V + E): where V is number of vertices and E is number of edges,
- Space Complexity: O(V): for storing the queue

- 示例：
```
输入：graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
输出：[0, 1, 2, 3]
```

#### 解决思路
广度优先搜索的基本思想是：从根节点开始，沿着树的宽度遍历树的节点。 

- 算法步骤
    - 从源节点开始遍历，将源节点标记为已访问，并加入队列；
    - 从队列中取出一个节点，对于该节点的所有邻接节点，如果未访问，则标记为已访问，并加入队列；
    - 重复步骤2，直到队列为空。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

std::vector<int> breadthFirstSearch(std::unordered_map<int, std::vector<int>>& graph) {
    std::unordered_set<int> visited;
    std::vector<int> result;
    std::queue<int> q;
    for (auto& [node, neighbors] : graph) {
        if (visited.find(node) == visited.end()) {
            q.push(node);
            visited.insert(node);
            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                result.push_back(curr);
                for (int neighbor : graph[curr]) {
                    if (visited.find(neighbor) == visited.end()) {
                        q.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }
        }
    }
    return result;
}

int main() {
    std::unordered_map<int, std::vector<int>> graph = {
        {0, {1, 2}},
        {1, {2}},
        {2, {0, 3}},
        {3, {3}}
    };
    std::vector<int> result = breadthFirstSearch(graph);
    for (int node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

## Question 15: 4 Letter Matrix Search (DFS/BFS)
### Letter Matrix Search
In the task of finding the position of the first letter of a search string in a letter matrix, you can traverse the matrix and, upon finding the first matching letter, use Depth-First Search (DFS) or Breadth-First Search (BFS) to check if it's possible to sequentially find all letters of the string starting from that point.

### Letter Matrix Search
给定一个字母矩阵和一个搜索字符串，找到这个字符串第一个字母所在的位置的题目，可以通过遍历矩阵，并在找到第一个匹配字母后，使用深度优先搜索（DFS）或广度优先搜索（BFS）检查是否可以从该点出发按顺序找到字符串中的所有字母。

#### 解决思路
- 首先，我们需要遍历矩阵，找到第一个匹配的字母。
- 然后，我们可以使用深度优先搜索（DFS）或广度优先搜索（BFS）来检查是否可以从该点出发按顺序找到字符串中的所有字母。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

bool dfs(std::vector<std::vector<char>>& board, std::string& word, int i, int j, int k) {
    if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || board[i][j] != word[k]) {
        return false;
    }
    if (k == word.size() - 1) {
        return true;
    }
    char tmp = board[i][j];
    board[i][j] = '/';
    bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i, j - 1, k + 1);
    board[i][j] = tmp;
    return res;
}

bool exist(std::vector<std::vector<char>>& board, std::string word) {
    for (int i = 0; i < board.size(); i++) {
        for (int j = 0; j < board[0].size(); j++) {
            if (dfs(board, word, i, j, 0)) {
                return true;
            }
        }
    }
    return false;
}

int main() {
    std::vector<std::vector<char>> board = {
        {'A', 'B', 'C', 'E'},
        {'S', 'F', 'C', 'S'},
        {'A', 'D', 'E', 'E'}
    };
    std::string word = "ABCCED";
    std::cout << std::boolalpha << exist(board, word) << std::endl;
    return 0;
}
```

## Question 16: Letter 深度优先搜索 Depth-First Search （DFS）
#### 题目描述
深度优先搜索（DFS）是一种用于遍历或搜索树或图数据结构的算法。它从根节点开始，沿着每个分支尽可能远地探索，然后回溯。DFS可以使用堆栈或递归来实现。它通常用于检测图中的循环，查找连通分量或解决具有多个解的谜题。

#### C++ 代码
```cpp
#include <iostream>
#include <unordered_map>
#include <unordered_set>

template <typename Node>
std::unordered_set<Node> dfs(const std::unordered_map<Node, std::unordered_set<Node>>& graph, const Node& start, std::unordered_set<Node>& visited) {
    visited.insert(start);
    for (const auto& neighbor : graph.at(start)) {
        if (visited.find(neighbor) == visited.end()) {
            dfs(graph, neighbor, visited);
        }
    }
    return visited;
}

int main() {
    std::unordered_map<char, std::unordered_set<char>> graph = {
        {'A', {'B', 'C'}},
        {'B', {'A', 'D', 'E'}},
        {'C', {'A', 'F'}},
        {'D', {'B'}},
        {'E', {'B', 'F'}},
        {'F', {'C', 'E'}}
    };
    std::unordered_set<char> visited;
    std::unordered_set<char> result = dfs(graph, 'A', visited);
    for (const auto& node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;  // Output: A B D E F C
    return 0;
}
```
```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
result = dfs(graph, 'A')
print(result)  # Output: {'A', 'B', 'C', 'D', 'E', 'F'}
```

## Question 17: Letter 广度优先搜索（BFS）
#### 题目描述
广度优先搜索（BFS）是一种用于遍历或搜索树或图数据结构的算法。它从根节点开始，探索当前深度的所有相邻节点，然后转移到下一个深度级别的节点。BFS可以使用队列来实现。它通常用于在无权图中查找最短路径，解决具有单个解的谜题或查找图的连通分量。

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    return visited
```

#### C++ 代码
```cpp
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <queue>

template <typename Node>
std::unordered_set<Node> bfs(const std::unordered_map<Node, std::unordered_set<Node>>& graph, const Node& start) {
    std::unordered_set<Node> visited;
    std::queue<Node> queue;
    queue.push(start);
    visited.insert(start);
    while (!queue.empty()) {
        Node node = queue.front();
        queue.pop();
        for (const auto& neighbor : graph.at(node)) {
            if (visited.find(neighbor) == visited.end()) {
                queue.push(neighbor);
                visited.insert(neighbor);
            }
        }
    }
    return visited;
}

int main() {
    std::unordered_map<char, std::unordered_set<char>> graph = {
        {'A', {'B', 'C'}},
        {'B', {'A', 'D', 'E'}},
        {'C', {'A', 'F'}},
        {'D', {'B'}},
        {'E', {'B', 'F'}},
        {'F', {'C', 'E'}}
    };
    std::unordered_set<char> result = bfs(graph, 'A');
    for (const auto& node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;  // Output: A B C D E F
    return 0;
}
```
```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
result = bfs(graph, 'A')
print(result)  # Output: {'A', 'B', 'C', 'D', 'E', 'F'}
```

## Question 18: Binary Tree + Sorting + Searching 
#### 题目描述
二叉树的递归遍历。
二叉树的递归遍历这一个内容，一定需要好好熟练使用，不要求理解，但要求下笔如有神！
在后面比较高阶的算法学习中，递归的思想是会不断被使用的。

具体来看二叉树的三种常见的方式：

- 前序遍历（pre-order）
- 中序遍历（in-order）
- 后序遍历（post-order）

这三种遍历方式的原理和步骤如下：

- 前序遍历（Pre-order）：
    - 遍历顺序：根节点 -> 左子树 -> 右子树
    - 步骤：
        - 访问根节点
        - 递归前序遍历左子树
        - 递归前序遍历右子树

- 中序遍历（In-order）：
    - 遍历顺序：左子树 -> 根节点 -> 右子树
    - 步骤：
        - 递归中序遍历左子树
        - 访问根节点
        - 递归中序遍历右子树

- 后序遍历（Post-order）：
    - 遍历顺序：左子树 -> 右子树 -> 根节点
    - 步骤：
        - 递归后序遍历左子树
        - 递归后序遍历右子树
        - 访问根节点

#### C/C++ 实现
```cpp
#include <iostream>

struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;
    
    TreeNode(int val) : data(val), left(nullptr), right(nullptr) {}
};

// 前序遍历
void preOrderTraversal(TreeNode* root) {
    if (root == nullptr) return;
    
    std::cout << root->data << " ";
    preOrderTraversal(root->left);
    preOrderTraversal(root->right);
}

// 中序遍历
void inOrderTraversal(TreeNode* root) {
    if (root == nullptr) return;
    
    inOrderTraversal(root->left);
    std::cout << root->data << " ";
    inOrderTraversal(root->right);
}

// 后序遍历
void postOrderTraversal(TreeNode* root) {
    if (root == nullptr) return;
    
    postOrderTraversal(root->left);
    postOrderTraversal(root->right);
    std::cout << root->data << " ";
}

int main() {
    // 在这里创建二叉树并进行测试
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);

    std::cout << "Pre-order traversal: ";
    preOrderTraversal(root);
    std::cout << std::endl;

    std::cout << "In-order traversal: ";
    inOrderTraversal(root);
    std::cout << std::endl;

    std::cout << "Post-order traversal: ";
    postOrderTraversal(root);
    std::cout << std::endl;

    return 0;
}
```

## Question 19: Binary Tree, inorder, preorder, postorder, level order traversal
A binary tree is a tree data structure in which each node has at most two children, referred to as the left child and the right child. The following are common methods for traversing a binary tree:

- **Inorder Traversal**: In this traversal, the left subtree is visited first, followed by the root node, and then the right subtree. It is commonly used to retrieve data in sorted order from a binary search tree.

- **Preorder Traversal**: In this traversal, the root node is visited first, followed by the left subtree, and then the right subtree. It is commonly used to create a copy of the tree.

- **Postorder Traversal**: In this traversal, the left subtree is visited first, followed by the right subtree, and then the root node. It is commonly used to delete a tree.

- **Level Order Traversal**: In this traversal, nodes are visited level by level, from left to right. It is commonly used to print the nodes at each level of the tree.

Here's an example of a binary tree and its traversals:

```cpp
#include <iostream>
#include <queue>
using namespace std;

struct Node {
    int data;
    Node* left;
    Node* right;
    Node(int value) : data(value), left(nullptr), right(nullptr) {}
};

void inorderTraversal(Node* root) {
    if (root == nullptr) return;
    inorderTraversal(root->left);
    cout << root->data << " ";
    inorderTraversal(root->right);
}

void preorderTraversal(Node* root) {
    if (root == nullptr) return;
    cout << root->data << " ";
    preorderTraversal(root->left);
    preorderTraversal(root->right);
}`

void postorderTraversal(Node* root) {
    if (root == nullptr) return;
    postorderTraversal(root->left);
    postorderTraversal(root->right);
    cout << root->data << " ";
}

void levelOrderTraversal(Node* root) {
    if (root == nullptr) return;
    queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        Node* current = q.front();
        cout << current->data << " ";
        if (current->left != nullptr) q.push(current->left);
        if (current->right != nullptr) q.push(current->right);
        q.pop();
    }
}

int main() {
    BinaryTree<int> tree;
    tree.insertNode(1);
    tree.insertNode(5);
    tree.insertNode(3);
    tree.insertNode(7);
    tree.insertNode(2);
    tree.insertNode(4);
    tree.insertNode(6);
    tree.insertNode(8);
    tree.insertNode(9);
    tree.insertNode(10);

    // The tree looks like:
    //     1
    //      \
    //       5
    //      / \
    //     3   7
    //    /   / \
    //   2   6   8
    //    \       \
    //     4       9
    //              \
    //               10

    
    std::cout << "Preorder Traversal: ";
    tree.preorderTraversal();
    std::cout << std::endl; // Output: 1 5 3 2 4 7 6 8 9 10
    std::cout << "Inorder Traversal: ";
    tree.inorderTraversal();
    std::cout << std::endl; // Output: 1 2 3 4 5 6 7 8 9 10
    std::cout << "Postorder Traversal: ";
    tree.postorderTraversal();
    std::cout << std::endl; // Output: 2 4 3 6 10 9 8 7 5 1
    std::cout << "Level Order Traversal: ";
    tree.levelOrderTraversal(); // Output: 1 5 3 7 2 4 6 8 9 10
    std::cout << std::endl;
    return 0;
}
```

## Question 20: Binary Tree 非递归后序遍历
#### 题目描述
题目：二叉树使用非递归实现后续遍历。

#### 解题思路
在二叉树的非递归后序遍历中，我们需要使用一个辅助栈来模拟递归的过程。具体来说，我们使用两个栈，一个栈用来模拟递归的过程，另一个栈用来存储遍历的结果。 

我们首先将根节点压入第一个栈中。然后开始循环，每次从第一个栈中弹出一个节点，然后将其压入第二个栈中。然后先将左子节点压入第一个栈中，再将右子节点压入第一个栈中。这样，我们就可以保证每次遍历的时候，都是先遍历左子树，然后遍历右子树。最后我们将第二个栈中的元素依次弹出，就可以得到后序遍历的结果。

#### 解决流程：

后续遍历访问顺序是左子树 - 右子树 - 根节点。在非递归实现后续遍历时，可以借助栈来辅助实现。

我将解决流程详细描述如下：

- 初始化： 首先，创建两个栈。一个主栈 stack 用于辅助遍历二叉树，另一个辅助栈 result_stack 用于记录后序遍历的结果。将根节点压入主栈。

- 主循环： 进入一个循环，循环条件为主栈非空。这是为了确保遍历所有节点。

- 处理当前节点： 从主栈弹出栈顶节点（当前节点），并将其值压入辅助栈 result_stack。这是为了保证后续遍历的顺序。

- 处理左子节点： 如果当前节点有左子节点，将左子节点压入主栈。这样可以确保在整个遍历过程中，左子树的节点先被处理。

- 处理右子节点： 如果当前节点有右子节点，将右子节点压入主栈。这保证了在处理完左子树后，右子树的节点会被遍历。

- 重复： 重复步骤3至步骤5直到主栈为空。这个过程确保了所有节点都被正确处理，且按照左子树 - 右子树 - 根节点的顺序。

-  得到后序遍历结果： 循环结束后，辅助栈 result_stack 中的元素顺序即为二叉树的后序遍历结果。由于是使用栈实现，最后需要将结果栈反转，使得顺序正确。

#### C++ 实现
```cpp
#include <iostream>
#include <stack>
#include <vector>

struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;
    
    TreeNode(int val) : data(val), left(nullptr), right(nullptr) {}
};

std::vector<int> postOrderTraversal(TreeNode* root) {
    std::vector<int> result;
    if (root == nullptr) return result;

    std::stack<TreeNode*> stack;
    std::stack<int> result_stack;
    stack.push(root);

    while (!stack.empty()) {
        TreeNode* current = stack.top();
        stack.pop();
        result_stack.push(current->data);

        if (current->left != nullptr) {
            stack.push(current->left);
        }
        if (current->right != nullptr) {
            stack.push(current->right);
        }
    }

    while (!result_stack.empty()) {
        result.push_back(result_stack.top());
        result_stack.pop();
    }

    return result;
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);

    std::vector<int> result = postOrderTraversal(root);
    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

## Question 21: A* Search Algorithm (A*搜索算法)
#### 题目描述
A*搜索算法是一种启发式搜索算法，它使用启发式来引导搜索过程。它通过结合从起始节点到达当前节点的成本和从当前节点到达目标节点的估计成本来评估节点。该算法使用优先级队列根据组合成本选择下一个要扩展的节点。如果启发式是可接受的和一致的，A*搜索是完备的和最优的。
#### 解决思路
- a graph, start node, and goal node as input.
- initialize open_set, came_from, g_score, and f_score data structures to keep track of nodes and their scores.
- while loop to iteratively select the next node to expand based on the combined cost f_score.
- If the goal node is reached, we reconstruct the path from the start node to the goal node and return it.
- If the goal node is not reached, we continue expanding nodes and updating their scores until the goal is reached or no more nodes are left to expand.

#### C++ 代码
```cpp
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>

template <typename Node>
std::vector<Node> astar_search(const std::unordered_map<Node, std::vector<Node>>& graph, const Node& start, const Node& goal) {
    std::unordered_set<Node> open_set;
    open_set.insert(start);
    std::unordered_map<Node, Node> came_from;
    std::unordered_map<Node, int> g_score;
    for (const auto& pair : graph) {
        g_score[pair.first] = std::numeric_limits<int>::max();
    }
    g_score[start] = 0;
    std::unordered_map<Node, int> f_score;
    for (const auto& pair : graph) {
        f_score[pair.first] = std::numeric_limits<int>::max();
    }
    f_score[start] = heuristic(start, goal);

    while (!open_set.empty()) {
        Node current;
        int min_f_score = std::numeric_limits<int>::max();
        for (const auto& node : open_set) {
            if (f_score[node] < min_f_score) {
                current = node;
                min_f_score = f_score[node];
            }
        }
        if (current == goal) {
            return reconstruct_path(came_from, current);
        }
        open_set.erase(current);
        for (const auto& neighbor : graph.at(current)) {
            int tentative_g_score = g_score[current] + dist_between(current, neighbor);
            if (tentative_g_score < g_score[neighbor]) {
                came_from[neighbor] = current;
                g_score[neighbor] = tentative_g_score;
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal);
                if (open_set.find(neighbor) == open_set.end()) {
                    open_set.insert(neighbor);
                }
            }
        }
    }
    return std::vector<Node>();
}

int main() {
    std::unordered_map<char, std::vector<char>> graph = {
        {'A', {'B', 'C'}},
        {'B', {'A', 'D', 'E'}},
        {'C', {'A', 'F'}},
        {'D', {'B'}},
        {'E', {'B', 'F'}},
        {'F', {'C', 'E'}}
    };
    std::vector<char> result = astar_search(graph, 'A', 'F');
    for (const auto& node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;  // Output: A C F
    return 0;
}
```

## Question 22: Dynamic Programming 爬樓梯
#### 题目描述
爬樓梯問題是一個非常經典的動態規劃問題，其描述如下：

假设你正在爬楼梯。需要 n 步你才能到达楼顶。每次你可以爬 1 或 2 个台阶。问有多少种不同的方法可以爬到楼顶。

- 示例: 
    - 输入：3 输出：3 解释：有三种方法可以爬到楼顶。
        - 1 步 + 1 步 + 1 步
        - 1 步 + 2 步
        - 2 步 + 1 步
当解决爬楼梯问题时，我们可以使用动态规划（Dynamic Programming）来优化问题求解。

#### 解题思路
1. 定义问题

我们要计算爬到楼梯的第 n 阶的不同方法数量。

2. 确定状态

引入一个数组 dp，其中 dp[i] 表示爬到第 i 阶楼梯的不同方法数量。

3. 初始化状态

初始化数组 dp，对于爬到第一个阶梯（dp[1]）和第二个阶梯（dp[2]）的方法数量，是已知的，因为在这两个情况下，我们只有一种方法爬到楼顶。
- dp[1] = 1
- dp[2] = 2

4. 确定状态转移方程

对于爬到第 i 阶楼梯的方法数量，可以通过前两个阶梯的方法数量之和得到：
- dp[i] = dp[i - 1] + dp[i - 2]

这是因为在爬到第 i 阶楼梯时，我们可以选择从第 i-1 阶直接跨一步到达，或者从第 i-2 阶跨两步到达。

5. 实现动态规划
#### C++ 实现
```cpp
#include <iostream>
#include <vector>

int climbStairs(int n) {
    if (n == 1) return 1;
    if (n == 2) return 2;

    std::vector<int> dp(n + 1);
    dp[1] = 1;
    dp[2] = 2;

    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
}

int climbStairs(int n){
    // Base cases
    if (n==1){
        return 1;
    }
    if (n ==2){
        return 2;
    }

    // double pointer array: dp[i] represents the number of ways to climd to the i-th step
    vector<int> dp
}

int main() {
    int n = 3;
    std::cout << climbStairs(n) << std::endl;
    return 0;
}
```

在这个函数中，我们首先检查特殊情况，如果 n 为1或2，直接返回1或2，因为在这两种情况下，爬楼梯的方法数量是已知的。然后，我们初始化一个数组 dp，并使用一个循环计算 dp[i] 直到 n。

6. 示例说明

让我们以 n = 3 为例来说明整个过程：
- 初始化 dp[1] = 1 和 dp[2] = 2。
- 对于 dp[3]，根据状态转移方程，dp[3] = dp[2] + dp[1] = 2 + 1 = 3。
- 对于 dp[4]，同样使用状态转移方程，dp[4] = dp[3] + dp[2] = 3 + 2 = 5。
- 以此类推，计算到 dp[n]。
这样，我们就得到了爬楼梯问题的动态规划解决方案。这个方法避免了重复计算，提高了算法的效率。




## Question 23: Dynamic Programming 接雨水
#### 题目描述
接雨水。给定 n 个非负整数，表示直方图的高度，每个宽度为 1 的条形块的高度可以视为一个直方图的宽度图。计算下雨之后能够装多少水。

- 示例：
```
输入：[0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
```
#### 解决思路
为了解决这个问题，我们可以使用动态规划或栈的方式。在这里，我将详细说明栈的解决思路。

- 栈的解决思路

我们可以使用栈来解决这个问题。具体来说，我们遍历直方图的高度，如果当前高度小于栈顶高度，我们将当前高度入栈。如果当前高度大于栈顶高度，我们将栈顶高度出栈，直到当前高度小于或等于栈顶高度。在出栈的过程中，我们可以计算当前高度和栈顶高度之间的水的容量。最后，我们将当前高度入栈，继续遍历直方图的高度。

- 动态规划解法

我们可以使用动态规划来解决这个问题。具体来说，我们可以预先计算每个位置左边的最大高度和右边的最大高度。然后，我们可以遍历每个位置，计算当前位置的水的容量。具体来说，我们可以使用下面的公式计算当前位置的水的容量：

- min(left_max[i], right_max[i]) - height[i]

这里，left_max[i] 表示位置 i 左边的最大高度，right_max[i] 表示位置 i 右边的最大高度，height[i] 表示位置 i 的高度。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <stack>

int trap(std::vector<int>& height) {
    int n = height.size();
    int result = 0;
    std::stack<int> stack;

    for (int i = 0; i < n; i++) {
        while (!stack.empty() && height[i] > height[stack.top()]) {
            int top = stack.top();
            stack.pop();
            if (stack.empty()) break;

            int distance = i - stack.top() - 1;
            int bounded_height = std::min(height[i], height[stack.top()]) - height[top];
            result += distance * bounded_height;
        }
        stack.push(i);
    }

    return result;
}

int main() {
    std::vector<int> height = {0,1,0,2,1,0,1,3,2,1,2,1};
    std::cout << trap(height) << std::endl;
    return 0;
}
```

## Question 24: Dynamic Programming (最大連續子數組)
#### 题目描述
Dynamic Programming. 动态规划（Dynamic Programming）是一种在数学、计算机科学和经济学中使用的，通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。动态规划常常适用于有重叠子问题和最优子结构性质的问题，动态规划方法所耗时间往往远少于朴素解法。

- 示例：
```
输入：[1, 2, 3, 4, 5]
输出：15
解释：连续子数组 [1, 2, 3, 4, 5] 的和最大，为 15。
```

#### 解决思路
动态规划的基本思想是：通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。

- 算法步骤
    - 定义子问题；
    - 实现子问题的递归；
    - 识别并求解出递归方程；
    - 将递归方程转换为迭代方程；
    - 应用迭代方程解决问题。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

int maxSubArray(std::vector<int>& nums) {
    int n = nums.size();
    int max_sum = nums[0];
    int curr_sum = nums[0];
    for (int i = 1; i < n; i++) {
        curr_sum = std::max(nums[i], curr_sum + nums[i]);
        max_sum = std::max(max_sum, curr_sum);
    }
    return max_sum;
}

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    std::cout << maxSubArray(nums) << std::endl;
    return 0;
}
```



## Question 25: Greedy Algorithm (Maximize Profit)
#### 题目描述
Greedy Algorithm. 贪心算法（Greedy Algorithm）是指在每一步选择中都采取在当前状态下最好或最优（即最有利）的选择，从而希望导致结果是全局最好或最优的算法。贪心算法在有最优子结构的问题中尤为有效。最优子结构的意思是局部最优解能决定全局最优解。

- 示例：
```
输入：[7, 1, 5, 3, 6, 4]
输出：7
解釋：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出，利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，利润 = 6-3 = 3 。

    总利润 = 4 + 3 = 7。
```

#### 解决思路
贪心算法的基本思想是：在每一步选择中都采取在当前状态下最好或最优的选择。

- 算法步骤
    - 定义问题的解；
    - 证明问题的最优子结构；
    - 设计一个递归算法；
    - 证明算法的正确性；
    - 设计一个迭代算法；
    - 证明算法的正确性。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

int maxProfit(std::vector<int>& prices) {
    int max_profit = 0;
    for (int i = 1; i < prices.size(); i++) {
        if (prices[i] > prices[i - 1]) {
            max_profit += prices[i] - prices[i - 1];
        }
    }
    return max_profit;
}

int main() {
    std::vector<int> prices = {7, 1, 5, 3, 6, 4};
    std::cout << maxProfit(prices) << std::endl;
    return 0;
}
```

## Question 26: Greedy Algorithm - Dijkstra's shortest path
#### 题目描述
Greedy Algorithm - Dijkstra's shortest path algorithm. Dijkstra's algorithm 是一种用于计算图中的最短路径的算法。它是由荷兰计算机科学家 Edsger W. Dijkstra 在1956 年提出的。该算法的目标是找到从源节点到图中所有其他节点的最短路径。

- 示例：
```
输入：graph = {
    0: {1: 4, 2: 1},
    1: {3: 1},
    2: {1: 2, 3: 5},
    3: {}
}
输出：[0, 3, 1, 2] (最短路径) 和 [0, 1, 2, 3] (最短距离) 
如何得出結果：从节点 0 到节点 3 的最短路径是 0 -> 2 -> 1 -> 3，距离为 8。

```
#### 解决思路
Dijkstra's algorithm 的基本思想是：找到从源节点到图中所有其他节点的最短路径。

- 算法步骤
    - 初始化距离数组，将源节点的距离设为 0，其他节点的距离设为无穷大；
    - 初始化优先队列，将源节点加入队列；
    - 从优先队列中取出一个节点，对于该节点的所有邻接节点，如果新的距离小于原来的距离，则更新距离，并将节点加入队列；
    - 重复步骤3，直到队列为空。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>

std::vector<int> dijkstra(std::unordered_map<int, std::unordered_map<int, int>>& graph, int source) {
    std::vector<int> dist(graph.size(), std::numeric_limits<int>::max());
    dist[source] = 0;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, source});
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        for (auto& [v, weight] : graph[u]) {
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

int main() {
    std::unordered_map<int, std::unordered_map<int, int>> graph = {
        {0, {{1, 4}, {2, 1}}},
        {1, {{3, 1}}},
        {2, {{1, 2}, {3, 5}}},
        {3, {}}
    };
    std::vector<int> dist = dijkstra(graph, 0);
    for (int d : dist) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    return 0;
}
```
## Question 27: Greedy Algorithm- Dijkstra's Algorithm V2 (Dijkstra算法)
#### 题目描述
Dijkstra算法是一种最短路径算法，它在加权图中找到从起始节点到所有其他节点的最短路径。它使用优先级队列根据从起始节点到当前节点的最短距离选择下一个要访问的节点。Dijkstra算法对于非负边权是完备和最优的，并且可以用于在道路网络、计算机网络或其他加权图中找到最短路径。
#### 解决思路
 - define a dijkstra function: take a graph and a start node as input.
 - initialize distances to keep track of the shortest distance from the start node to each node and a priority queue queue to store nodes to be visited.
 - while loop to iteratively visit nodes in the queue and update their distances if a shorter path is found.
 - return the distances after the entire graph has been traversed.
#### C++ 代码
```cpp
#include <iostream>
#include <unordered_map>
#include <queue>
#include <vector>

template <typename Node>
std::unordered_map<Node, int> dijkstra(const std::unordered_map<Node, std::unordered_map<Node, int>>& graph, const Node& start) {
    std::unordered_map<Node, int> distances;
    for (const auto& pair : graph) {
        distances[pair.first] = std::numeric_limits<int>::max();
    }
    distances[start] = 0;
    std::priority_queue<std::pair<int, Node>, std::vector<std::pair<int, Node>>, std::greater<std::pair<int, Node>>> queue;
    queue.push(std::make_pair(0, start));
    while (!queue.empty()) {
        int current_distance = queue.top().first;
        Node current_node = queue.top().second;
        queue.pop();
        if (current_distance > distances[current_node]) {
            continue;
        }
        for (const auto& pair : graph.at(current_node)) {
            Node neighbor = pair.first;
            int weight = pair.second;
            int distance = current_distance + weight;
            if (distance < distances[neighbor]) {
                distances[neighbor] = distance;
                queue.push(std::make_pair(distance, neighbor));
            }
        }
    }
    return distances;
}

int main() {
    std::unordered_map<char, std::unordered_map<char, int>> graph = {
        {'A', {{'B', 5}, {'C', 3}}},
        {'B', {{'A', 5}, {'C', 1}, {'D', 3}}},
        {'C', {{'A', 3}, {'B', 1}, {'D', 2}}},
        {'D', {{'B', 3}, {'C', 2}}}
    };
    std::unordered_map<char, int> result = dijkstra(graph, 'A');
    for (const auto& pair : result) {
        std::cout << pair.first << ": " << pair.second << " ";
    }
    std::cout << std::endl;  // Output: A: 0 B: 4 C: 3 D: 5
    return 0;
}
```
```python
graph = {
    'A': {'B': 5, 'C': 3},
    'B': {'A': 5, 'C': 1, 'D': 3},
    'C': {'A': 3, 'B': 1, 'D': 2},
    'D': {'B': 3, 'C': 2}
}
result = dijkstra(graph, 'A')
print(result)  # Output: {'A': 0, 'B': 4, 'C': 3, 'D': 5}
```

## Question 28: Greedy Algorithm - Kruskal's minimum spanning tree
#### 题目描述
Greedy Algorithm - Kruskal's minimum spanning tree algorithm. Kruskal's algorithm 是一种用于计算图中的最小生成树的算法。最小生成树是一个图的子图，它是一个树，包含图中所有的顶点，但是只包含足够的边来保持树的连通性，并且这些边的权值之和最小。

- 示例：
```
输入：graph = {
    0: {1: 4, 7: 8},
    1: {0: 4, 7: 11, 2: 8},
    2: {1: 8, 8: 2, 5: 4, 3: 7},
    3: {2: 7, 5: 14, 4: 9},
    4: {3: 9, 5: 10},
    5: {4: 10, 3: 14, 2: 4, 6: 2},
    6: {5: 2, 8: 6, 7: 1},
    7: {6: 1, 8: 7, 1: 11, 0: 8},
    8: {2: 2, 6: 6, 7: 7}
}
输出：[0, 1, 2, 8, 6, 5, 4, 3, 7] (最小生成树)
```

#### 解决思路
Kruskal's algorithm 的基本思想是：计算图中的最小生成树。

- 算法步骤
    - 初始化最小生成树；
    - 对图中的所有边按照权值进行排序；
    - 从权值最小的边开始，如果边的两个顶点不在同一个连通分量中，则将边加入最小生成树；
    - 重复步骤3，直到最小生成树中包含了图中所有的顶点。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>

class DisjointSet {
public:
    DisjointSet(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find(int u) {
        if (u != parent[u]) {
            parent[u] = find(parent[u]);
        }
        return parent[u];
    }

    void merge(int u, int v) {
        int pu = find(u);
        int pv = find(v);
        if (pu != pv) {
            if (rank[pu] > rank[pv]) {
                parent[pv] = pu;
            } else if (rank[pu] < rank[pv]) {
                parent[pu] = pv;
            } else {
                parent[pu] = pv;
                rank[pv]++;
            }
        }
    }

private:

    std::vector<int> parent;
    std::vector<int> rank;
};

std::vector<int> kruskal(std::unordered_map<int, std::unordered_map<int, int>>& graph) {
    std::vector<int> result;
    std::vector<std::pair<int, std::pair<int, int>>> edges;
    for (auto& [u, neighbors] : graph) {
        for (auto& [v, weight] : neighbors) {
            edges.push_back({weight, {u, v}});
        }
    }
    std::sort(edges.begin(), edges.end());
    DisjointSet ds(graph.size());
    for (auto& [weight, edge] : edges) {
        int u = edge.first;
        int v = edge.second;
        if (ds.find(u) != ds.find(v)) {
            ds.merge(u, v);
            result.push_back(u);
            result.push_back(v);
        }
    }
    return result;
}

int main() {
    std::unordered_map<int, std::unordered_map<int, int>> graph = {
        {0, {1, 4, 7, 8}},
        {1, {0, 4, 7, 11, 2, 8}},
        {2, {1, 8, 8, 2, 5, 4, 3, 7}},
        {3, {2, 7, 5, 14, 4, 9}},
        {4, {3, 9, 5, 10}},
        {5, {4, 10, 3, 14, 2, 4, 6, 2}},
        {6, {5, 2, 8, 6, 7, 1}},
        {7, {6, 1, 8, 7, 1, 11, 0, 8}},
        {8, {2, 2, 6, 6, 7, 7}}
    };
    std::vector<int> result = kruskal(graph);
    for (int node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    return 0;
}
```
## Question 29: Greedy Algorithm - Prim's minimum spanning tree
#### 题目描述
Greedy Algorithm - Prim's minimum spanning tree algorithm. Prim's algorithm 是一种用于计算图中的最小生成树的算法。最小生成树是一个图的子图，它是一个树，包含图中所有的顶点，但是只包含足够的边来保持树的连通性，并且这些边的权值之和最小。

- 示例：
```
输入：graph = {
    0: {1: 4, 7: 8},
    1: {0: 4, 7: 11, 2: 8},
    2: {1: 8, 8: 2, 5: 4, 3: 7},
    3: {2: 7, 5: 14, 4: 9},
    4: {3: 9, 5: 10},
    5: {4: 10, 3: 14, 2: 4, 6: 2},
    6: {5: 2, 8: 6, 7: 1},
    7: {6: 1, 8: 7, 1: 11, 0: 8},
    8: {2: 2, 6: 6, 7: 7}
}

输出：[0, 1, 2, 8, 6, 5, 4, 3, 7] (最小生成树)
```

#### 解决思路
Prim's algorithm 的基本思想是：计算图中的最小生成树。

- 算法步骤
    - 初始化最小生成树；
    - 初始化优先队列，将源节点加入队列；
    - 从优先队列中取出一个节点，对于该节点的所有邻接节点，如果新的权值小于原来的权值，则更新权值，并将节点加入队列；
    - 重复步骤3，直到最小生成树中包含了图中所有的顶点。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>

std::vector<int> prim(std::unordered_map<int, std::unordered_map<int, int>>& graph) {
    std::vector<int> result;
    std::vector<int> dist(graph.size(), std::numeric_limits<int>::max());
    std::vector<bool> visited(graph.size(), false);
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, 0});
    dist[0] = 0;
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        visited[u] = true;
        for (auto& [v, weight] : graph[u]) {
            if (!visited[v] && weight < dist[v]) {
                dist[v] = weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

int main() {
    std::unordered_map<int, std::unordered_map<int, int>> graph = {
        {0, {1, 4, 7, 8}},
        {1, {0, 4, 7, 11, 2, 8}},
        {2, {1, 8, 8, 2, 5, 4, 3, 7}},
        {3, {2, 7, 5, 14, 4, 9}},
        {4, {3, 9, 5, 10}},
        {5, {4, 10, 3, 14, 2, 4, 6, 2}},
        {6, {5, 2, 8, 6, 7, 1}},
        {7, {6, 1, 8, 7, 1, 11, 0, 8}},
        {8, {2, 2, 6, 6, 7, 7}}
    };
    std::vector<int> result = prim(graph);
    for (int node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    return 0;
}
```