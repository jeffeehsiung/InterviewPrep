# Time and Space Complexity
## Sorting Algorithms Time and Space Complexity:

**O(n log n) Time Complexity:**
1. Merge Sort (O(n) space complexity)
2. Heap Sort 
    - (O(1) space complexity, in-place variant exists) 
    - (O(n) space complexity for the heap structure)
3. Quick Sort (O(log n) space complexity on average, in-place)

**O(n^2) Time Complexity:**
1. Bubble Sort (O(1) space complexity, in-place)
2. Insertion Sort (O(1) space complexity, in-place)
3. Selection Sort (O(1) space complexity, in-place)

## Searching Algorithms (sorted) Time and Space Complexity:
**Time Complexity:**
1. **O(1) - Constant Time:**
   - Direct Access Table (DAT) lookup (when indexing an array, for example).
2. **O(log n) - Logarithmic Time:**
   - Binary Search (for sorted arrays).
3. **O(n) - Linear Time:**
   - Linear Search (for unsorted arrays or lists).
4. **O(n log n) - Linearithmic Time:**
   - Balanced Tree Search (e.g., AVL tree, Red-Black tree).

**Space Complexity:**
1. **O(1) - Constant Space:**
   - Iterative algorithms with a constant number of variables.
2. **O(log n) - Logarithmic Space:**
   - Recursive algorithms with logarithmic call stack depth.
3. **O(n) - Linear Space:**
   - Algorithms that require storing the entire input.

## Searching Algorithms (non-sorted) Time and Space Complexity:
**Time Complexity:**
1. **O(n) - Linear Time:**
   - Linear Search: Simply iterate through the array until the target element is found or the end is reached.
2. **O(n log n) - Linearithmic Time:**
   - There aren't many commonly used algorithms with this time complexity for non-sorted arrays. Algorithms like Timsort use this time complexity but are primarily sorting algorithms.

**Space Complexity:**
1. **O(1) - Constant Space:**
   - Linear Search: Typically requires only a constant amount of space for storing a few variables.
2. **O(n) - Linear Space:**
   - In certain situations, you might need to use additional space for data structures or variables.

For non-sorted arrays, Linear Search (O(n) time complexity) is often a straightforward and practical choice. The absence of any order makes more efficient algorithms like binary search (O(log n)) unsuitable.