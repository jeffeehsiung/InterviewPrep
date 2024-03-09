### 数据结构和算法的八股文通常包括以下几个方面的问题：

1. 数据结构基础：包括数组、链表、栈、队列、哈希表、树（包括二叉树、二叉搜索树、平衡树、红黑树等）、图（包括有向图、无向图、权重图等）等基础知识。

2. 基础算法：包括排序算法（冒泡排序、选择排序、插入排序、快速排序、归并排序、堆排序等）、查找算法（二分查找、深度优先搜索、广度优先搜索等）、动态规划、贪心算法、分治算法等。

3. 高级数据结构和算法：包括B树、B+树、跳表、布隆过滤器、LRU缓存、一致性哈希等。

4. 算法设计技巧：包括递归、迭代、双指针、滑动窗口、位运算等。

5. 算法复杂度分析：理解并能够计算时间复杂度和空间复杂度。

6. 数据结构和算法的应用：如何在实际问题中选择和使用合适的数据结构和算法。

7. 常见的编程题：例如LeetCode、剑指Offer等上的题目，以及它们的解题思路和代码实现。

### Problem Description
The "standard questions" for data structures and algorithms usually include the following aspects:

1. Basics of Data Structures: This includes arrays, linked lists, stacks, queues, hash tables, trees (including binary trees, binary search trees, balanced trees, red-black trees, etc.), graphs (including directed graphs, undirected graphs, weighted graphs, etc.).

2. Basic Algorithms: This includes sorting algorithms (bubble sort, selection sort, insertion sort, quick sort, merge sort, heap sort, etc.), search algorithms (binary search, depth-first search, breadth-first search, etc.), dynamic programming, greedy algorithms, divide and conquer algorithms, etc.

3. Advanced Data Structures and Algorithms: This includes B-trees, B+ trees, skip lists, Bloom filters, LRU cache, consistent hashing, etc.

4. Algorithm Design Techniques: This includes recursion, iteration, two pointers, sliding window, bit manipulation, etc.

5. Algorithm Complexity Analysis: Understanding and being able to calculate time complexity and space complexity.

6. Application of Data Structures and Algorithms: How to choose and use appropriate data structures and algorithms in practical problems.


## Question 1
```python
'''
Image. Background & foreground objects (convex)
Goal: Count the number of obiects: Provide bounding boxes for them.
Assumption: Background is darker than foreground
Approach:
1. Convert to grayscale
2. Apply Gaussian blur
3. Apply thresholding
4. Find contours
5. Draw bounding boxes
6. Count the number of objects
7. Display the output

- Preprocessing: 
    Since the assumption is that the background is darker than the foreground, you might first want to apply a thresholding operation to separate the foreground from the background. Adaptive thresholding could be beneficial if the lighting across the image isn't uniform.
- Finding Contours: 
    After thresholding, you can find contours in the image. Contours can be thought of as the boundaries of the foreground objects. OpenCV provides a function findContours to do this.
- Filtering Contours: 
    Depending on the quality of the image and the result of thresholding, you might get several contours, not all of which correspond to actual objects of interest. You can filter out the smaller contours based on area or other criteria like contour perimeter, aspect ratio, etc.
- Bounding Boxes: 
    For each of the filtered contours, you can calculate the bounding box. OpenCV has a function boundingRect that calculates the minimal upright bounding rectangle for a set of points (the contour in this case).
- Counting Objects and Drawing Bounding Boxes: 
    The number of objects will be equal to the number of bounding boxes (filtered contours), and you can draw these bounding boxes on the original image to visually identify the objects.
'''
import cv2
import numpy as np

# Load the image
image = cv2.imread('path_to_your_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) # Use ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C to apply adaptive thresholding. 

"""
Key Features of ADAPTIVE_THRESH_GAUSSIAN_C:
- Local Thresholding: 
    Unlike global thresholding, which applies a single threshold value across the entire image, ADAPTIVE_THRESH_GAUSSIAN_C adjusts thresholds locally for each region of the image. This results in better handling of images with uneven lighting conditions.
- Gaussian Weighted Sum: 
    For each target pixel, the method calculates a threshold by considering the weighted sum of nearby pixel values. The weights are determined by a Gaussian window, which gives more importance to pixels closer to the center of the window. This weighting approach helps in smoothing the image and reducing noise, leading to more accurate segmentation of foreground from the background.
- Parameter Tuning: 
    When using ADAPTIVE_THRESH_GAUSSIAN_C, you can adjust several parameters, including the size of the neighborhood (the window size) and a constant C subtracted from the computed weighted mean. These parameters allow fine-tuning of the thresholding operation to better adapt to specific image characteristics.
"""

# Find contours
# Use RETR_EXTERNAL to only get the outer contours; RETR_TREE to get all contours; and CHAIN_APPROX_SIMPLE to compress horizontal, vertical, and diagonal segments and leave only their end points.
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 


# Filter contours and draw bounding boxes
# Depending on the quality of the image and the result of thresholding, you might get several contours, not all of which correspond to actual objects of interest. You can filter out the smaller contours based on area or other criteria like contour perimeter, aspect ratio, etc.
object_count = 0
for contour in contours:
    if cv2.contourArea(contour) > minimum_area_threshold:  # Define minimum_area_threshold based on your needs.
        object_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Count the number of objects
print(f"Number of objects: {object_count}")
cv2.imshow('Objects Bounded', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```


## Question 2
```python
'''
Receptive Field of the Network?
Input: 64x64x3 ch
Layer1: Conv2D 3x3 spatial output channels: 10, Stride: 1:
Layer2: Conv2D 1x1 spatial output channels: 4, Stride: 1
Layer3: Conv2D 3x3 spatial output channels 4, Stride: 1

Solution:
Definition:
Receptive Field (RF) of a neuron in a network is the area in the input image that affects the output of that neuron. It is the area in the input image that the neuron is looking at.

'''

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0)
        self.layer2 = nn.Conv2d(10, 4, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

model = Net()
print(model)

```

The receptive field (RF) of a convolutional neural network (CNN) describes the size of the input area that influences a particular feature in the output. To calculate the receptive field of the network for the given layers, we can follow a simple formula. The formula for calculating the receptive field of a layer in a CNN, considering only the size of the convolution kernels and the stride, and assuming the stride of all layers before it is 1 (which is the case here), is as follows:

`RF_new = RF_prev + ((KernelSize - 1) * ProductOfAllPreviousStrides)`

Given your network configuration:

1. **Input**: 64x64x3
2. **Layer1**: Conv2D 3x3, output channels: 10, Stride: 1
3. **Layer2**: Conv2D 1x1, output channels: 4, Stride: 1
4. **Layer3**: Conv2D 3x3, output channels: 4, Stride: 1

<!-- ### Definitions:
**Kernel**
- A kernel (also known as a filter) in a CNN is a small matrix used to apply operations like convolution across an input image or a feature map from the previous layer. Kernels systematically slide (or "convolve") across the input to produce feature maps, which highlight certain types of features (like edges, textures, or patterns) depending on the kernel's weights.
- The primary role of a kernel is to extract spatial hierarchy of features from the input by performing convolution operations. These features become increasingly abstract with each subsequent convolutional layer.
Each kernel in a CNN layer looks for a specific feature in the input data.

**Neuron**
- A neuron is a computational unit that receives input, processes it (often with a non-linear activation function), and produces output. In the context of CNNs, each element in a feature map can be considered as the output of a neuron. These neurons are arranged in layers, and the output of one layer becomes the input for the next.
- Neurons in CNNs are responsible for learning the weights and biases applied during the convolution operations (performed by kernels) and other transformations (like pooling, normalization, etc.).
- In fully connected layers (also known as dense layers), which often come after convolutional layers in a CNN architecture, each neuron receives input from all neurons in the previous layer, contributing to the network's ability to make decisions based on the entire input.

**Key Differences**
- A kernel is specifically a set of weights that convolve over the input to produce feature maps, focusing on spatial features extraction.
- A neuron refers to the broader concept of a computational unit in neural networks, including those in CNNs, which performs weighted input summations followed by a non-linear operation.
- In the context of CNNs, a neuron can be thought of as the output of a feature map element, while a kernel is the set of weights used to perform convolution operations. 

**Stride**
- The stride in a CNN refers to the number of pixels by which the kernel (or filter) moves across the input image or feature map during the convolution operation. A stride of 1 means the kernel moves one pixel at a time, while a stride of 2 means it moves two pixels at a time, and so on.

Key Aspects of Strides:
- Dimension Reduction: 
    A stride greater than one reduces the size of the output feature map compared to the input. This is because, with larger strides, the kernel jumps over more pixels at a time and covers the input image with fewer steps. For instance, a stride of 2 means the kernel moves 2 pixels at a time, both horizontally and vertically, effectively reducing the size of the output feature map by roughly a factor of 4 compared to a stride of 1.

- Computation Efficiency: 
    Increasing the stride reduces the computational load. By skipping over pixels, the network performs fewer convolutions, which can speed up the training and inference processes. This can be particularly beneficial when dealing with large images.

- Field of View: 
    A larger stride also impacts the kernel's field of view on the input. By moving in larger steps, each position of the kernel integrates information from a broader area of the input image, albeit at the cost of potentially missing finer details that a smaller stride might capture.

- Control Overfitting: 
    In some cases, using larger strides instead of pooling layers to reduce feature map sizes can help control overfitting by reducing the network's capacity (i.e., the number of trainable parameters).

- Strides in Pooling Layers:
    Strides are not exclusive to convolutional layers; they are also used in pooling layers (like max pooling or average pooling). In pooling layers, the stride determines how the pooling window moves across the input feature map. Similar to convolutional layers, a larger stride in a pooling layer results in a smaller output size.

- Setting Strides:
    When defining a convolutional or pooling layer in a neural network architecture, you specify the stride. For example, in many deep learning frameworks, you might see a parameter setting like stride=(2, 2), indicating that the kernel or pooling window moves 2 pixels across both dimensions of the input for each step.

-->

Let's calculate the receptive field (RF) step by step for each layer:

### Initial Condition

- For the input layer (before any convolution), the receptive field is 1, as each pixel sees only itself.

### Layer 1

- **Kernel Size**: 3
- **Stride**: 1

`RF_Layer1 = 1 + ((3 - 1) * 1) = 3`

### Layer 2

Layer 2 has a kernel size of 1, which doesn't increase the receptive field in terms of spatial extent. It serves more to combine features from the previous layer without looking at additional context from the input image.

- **Kernel Size**: 1
- **Stride**: 1

`RF_Layer2 = RF_Layer1 + ((1 - 1) * 1) = 3`

### Layer 3

- **Kernel Size**: 3
- **Stride**: 1

`RF_Layer3 = RF_Layer2 + ((3 - 1) * 1) = 5`

### Conclusion

The receptive field of the network after Layer 3 is 5x5. This means that each feature in the output after passing through these three layers is influenced by a 5x5 patch in the original input image.



## Question 3
```python
'''
Input 1D letter array = [E R T R, P O T Q, R I T A, S O W E]. 
    width 4, height 4 → 2D matrix
    Input 1D search string: [R O T]
ERTR
POTQ 
RITA
SOWE
Check if search string matches the 1st matrix diagonally. Output first location of letter 'R' → (2,0)

Approach:
1. Convert the 1D letter array to a 2D matrix
2. Iterate through the 2D matrix and check for the first letter of the search string
3. If found, start a diagonal search from that position to check if the search string matches
4. If found, return the position
5. If not found, return "Not found"

Detail: To solve this problem, we first need to convert the 1D letter array into a 2D matrix representation, given the width and height. After that, we will search for the input 1D search string within this matrix diagonally, and if a match is found, we output the first location of the letter 'R' from the search string.

Here's a breakdown of the steps:

1. Convert 1D letter array to 2D matrix: Given the width and height, we can reshape the input 1D letter array into a 2D matrix.
2. Search for the search string diagonally: Starting from each element in the matrix, we attempt to match the search string diagonally (both from left to right and potentially from right to left, if needed). In this case, it seems we are only looking in a diagonal from the top left to bottom right direction.
3. Output the first location of 'R': Once a match is found, we return the first location of the letter 'R' in the matrix based on where the search string matches.

The matrix looks like this when formed:
E R T R
P O T Q
R I T A
S O W E

'''

def create_2d_matrix(letter_array, width, height):
    # Create 2D matrix from the letter array
    matrix = [list(row.replace(' ', '')) for row in letter_array]
    return matrix

def search_diagonally(matrix, search_str):
    height = len(matrix)
    width = len(matrix[0])
    
    for y in range(height):
        for x in range(width):
            if matrix[y][x] == search_str[0]:  # Found the first letter of the search string
                # Check if all subsequent letters of the search string match diagonally
                if all((y+i) < height and (x+i) < width and matrix[y+i][x+i] == search_str[i] for i in range(len(search_str))):
                    return (x, y)  # Return the position of the first letter
    return None

# Input data
letter_array = ["E R T R", "P O T Q", "R I T A", "S O W E"]
search_str = "ROT" 
width, height = 4, 4

# Process
matrix = create_2d_matrix(letter_array, width, height)
first_location = search_diagonally(matrix, search_str)

print(f"First location of '{search_str[0]}': {first_location}")
    
```

## Question 4
问题：给定一个链表，判断是否成环。

#### 解决方案

解决方法的话，咱们一般情况下会立马想到有三种：

- 快慢指针法
- 标记法
- 修改节点值法
下面，详细来说说看~

#### 快慢指针法
初始化两个指针，一个慢指针（slow）和一个快指针（fast），初始位置为链表的头节点（head）。

迭代遍历链表，每次循环中，慢指针前进一步，快指针前进两步。

如果链表中有环，快指针最终会追上慢指针，即两者会相遇。

如果链表中没有环，快指针会在某一点到达链表的末尾（即为None），此时退出循环。

最终判断，如果快指针在遍历过程中与慢指针相遇，则链表成环；否则，链表不成环。

#### 标记法
初始化一个集合（或哈希表）用于存储已经访问过的节点，初始为空。

迭代遍历链表，每次循环中，检查当前节点是否已经在集合中。

如果当前节点已经在集合中，表示链表成环，结束循环。

如果当前节点不在集合中，将当前节点添加到集合中，并将指针移动到下一个节点。

最终判断，如果在遍历过程中遇到已经访问过的节点，则链表成环；否则，链表不成环。

#### 修改节点值法
初始化一个特殊值（例如None）用于标记已经访问过的节点。

迭代遍历链表，每次循环中，检查当前节点的值是否等于特殊值。

如果当前节点的值等于特殊值，表示链表成环，结束循环。

如果当前节点的值不等于特殊值，将当前节点的值修改为特殊值，并将指针移动到下一个节点。

最终判断，如果在遍历过程中遇到值为特殊值的节点，则链表成环；否则，链表不成环。

总的来说，上面三种方法都利用了链表中是否有环时快慢指针相遇的特点，通过不同的实现方式来达到判断链表是否成环的目的。

通常来说，大家都喜欢用第一种，快慢指针法，很多情况可以用这种方式解决。

最后，给出C、Python和Java的代码，供大家参考。

当然，下面分别给出 C、Python 和 Java 的代码示例，使用的是快慢指针法来判断链表是否成环。在每种语言的代码中，hasCycle 函数都是实现了快慢指针算法。用于判断链表是否成环。

#### C 语言代码
```c
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int value;
    struct ListNode* next;
};

int hasCycle(struct ListNode* head) {
    if (head == NULL || head->next == NULL) {
        return 0;
    }

    struct ListNode* slow = head;
    struct ListNode* fast = head->next;

    while (slow != fast) {
        if (fast == NULL || fast->next == NULL) {
            return 0;
        }
        slow = slow->next;
        fast = fast->next->next;
    }

    return 1;
}

int main() {
    // 创建链表节点
    struct ListNode* head = (struct ListNode*)malloc(sizeof(struct ListNode));
    struct ListNode* second = (struct ListNode*)malloc(sizeof(struct ListNode));
    struct ListNode* third = (struct ListNode*)malloc(sizeof(struct ListNode));

    // 设置节点值和连接关系
    head->value = 1;
    head->next = second;
    second->value = 2;
    second->next = third;
    third->value = 3;
    third->next = head; // 创建一个环

    // 判断链表是否成环
    if (hasCycle(head)) {
        printf("The linked list has a cycle.\n");
    } else {
        printf("The linked list does not have a cycle.\n");
    }

    // 释放内存
    free(head);
    free(second);
    free(third);

    return 0;
}
```


#### Python 代码
```python

class ListNode:
    def __init__(self, value):
        self.value = value
        self.next = None
        
    
def has_cycle(head):
    if not head or not head.next:
        return False

    slow = head
    fast = head.next

    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next

    return True


def main():
    # 创建链表节点
    head = ListNode(1)
    second = ListNode(2)
    third = ListNode(3)

    # 设置节点连接关系
    head.next = second
    second.next = third
    third.next = head  # 创建一个环

    # 判断链表是否成环
    if has_cycle(head):
        print("The linked list has a cycle.")
    else:
        print("The linked list does not have a cycle.")

if __name__ == "__main__":
    main()

```

## Question 5
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

你可以按任意顺序返回答案。

示例：

输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：nums[0] + nums[1] = 2 + 7 = 9，因此返回 [0, 1]。
解决方式
当涉及到 "Two Sum" 问题时，不同的解决方法有不同的思路。

这里给出常见的 4 种解决办法，

#### 哈希表：

遍历数组，将每个元素及其索引存储在哈希表中。
对于每个元素，计算目标值与当前元素的差值，检查差值是否在哈希表中。
优点是时间复杂度为 O(n)，哈希表的查询操作是常数时间的。

#### 双指针法：

对有序数组使用双指针，一个指向数组的开头，另一个指向数组的结尾。
根据两个指针所指元素的和与目标值的关系，逐步调整指针的位置。
适用于有序数组，时间复杂度为 O(n log n)。

#### 集合（Set）：

遍历数组，对于每个元素，检查目标值减去当前元素的差值是否在集合中。
将当前元素添加到集合中，以便后续元素的检查。
与哈希表方法类似，时间复杂度为 O(n)。

#### 代码实现
下面的代码中，我这边使用的是哈希表（Hash Map）的解决办法。

大家可以提供其他的解决办法，随时交流~

具体而言，使用了一个字典（Python 中的 dict 或者在 C/C++ 中可以用 std::unordered_map）来存储数组中的元素及其索引。

遍历数组时，对于每个元素，计算目标值与当前元素的差值，然后检查差值是否在字典中。如果在字典中找到了差值，就说明找到了符合条件的两个元素，返回它们的索引。

这种方法的优点是时间复杂度较低，为 O(n)，其中 n 是数组的长度。由于哈希表的查询操作是常数时间的，因此这个算法在性能上比一些其他方法更优。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>

// 找到两个数的索引，使它们的和等于目标值
std::vector<int> twoSum(std::vector<int>& nums, int target) {
    std::unordered_map<int, int> numMap;
    std::vector<int> result;

    for (int i = 0; i < nums.size(); ++i) {
        int complement = target - nums[i];
        
        // 检查差值是否在哈希表中
        if (numMap.find(complement) != numMap.end()) {
            result.push_back(numMap[complement]);
            result.push_back(i);
            return result;
        }
        
        // 如果差值不在哈希表中，将当前元素及其索引加入哈希表
        numMap[nums[i]] = i;
    }

    return result;
}

int main() {
    std::vector<int> nums = {2, 7, 11, 15};
    int target = 9;

    std::vector<int> result = twoSum(nums, target);

    // 输出结果
    for (int i : result) {
        std::cout << i << " ";
    }

    return 0;
}

```

#### Python 代码：
```python
# 找到两个数的索引，使它们的和等于目标值
def twoSum(nums, target):
    num_dict = {}

    for i, num in enumerate(nums):
        complement = target - num
        
        # 检查差值是否在字典中
        if complement in num_dict:
            return [num_dict[complement], i]
        
        # 如果差值不在字典中，将当前元素及其索引加入字典
        num_dict[num] = i

    return []

nums = [2, 7, 11, 15]
target = 9
result = twoSum(nums, target)
print(result)
Go 代码：
package main

import "fmt"

// 找到两个数的索引，使它们的和等于目标值
func twoSum(nums []int, target int) []int {
    numMap := make(map[int]int)

    for i, num := range nums {
        complement := target - num
        
        // 检查差值是否在映射中
        if index, ok := numMap[complement]; ok {
            return []int{index, i}
        }
        
        // 如果差值不在映射中，将当前元素及其索引加入映射
        numMap[num] = i
    }

    return []int{}
}

func main() {
    nums := []int{2, 7, 11, 15}
    target := 9

    // 调用函数
    result := twoSum(nums, target)
    
    // 输出结果
    fmt.Println(result)
}

```

## Question 6
#### 题目描述
给定两个非空链表，表示两个非负整数。它们每位数字都是按照逆序方式存储的，并且每个节点只能存储一位数字。请将两个数相加并以相同形式返回一个表示和的链表。

例如：

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
解释：342 + 465 = 807。
```

#### 解题思路
思路这边，我给到5种解决方案，以及详细的说明。

1. 模拟加法过程：

初始化一个虚拟节点作为结果链表的头结点，并定义一个指针指向当前节点。
使用一个变量 carry 来记录进位，初始化为0。
遍历两个链表，对于每一位进行相加，同时加上前一位的进位。
如果相加结果大于等于10，则更新进位。
创建一个新节点，其值为相加结果的个位数，连接到结果链表上。
最后检查是否有额外的进位，如果有，则添加一个值为1的新节点。
返回结果链表的头结点。

代码是使用模拟加法过程的思路，也是最常见、最直观的解法。该方法的基本思想是模拟手工相加的过程，逐位相加，考虑进位，生成新的链表。

这种方法最大的优点是清晰易懂，直接反映了加法的本质。缺点是可能稍微繁琐，需要考虑一些边界情况和细节。

遍历两个链表，逐位相加，并考虑进位的情况。最终得到的结果即为相加后的链表。

#### C++ 语言：
```cpp
#include <iostream>

using namespace std;

// 定义单链表节点
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(NULL) {}
};

// 函数声明
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);

int main() {
    // 创建两个链表
    ListNode* l1 = new ListNode(2);
    l1->next = new ListNode(4);
    l1->next->next = new ListNode(3);

    ListNode* l2 = new ListNode(5);
    l2->next = new ListNode(6);
    l2->next->next = new ListNode(4);

    // 调用函数，计算结果链表
    ListNode* result = addTwoNumbers(l1, l2);

    // 输出结果链表的值
    while (result != NULL) {
        cout << result->val << " ";
        result = result->next;
    }

    return 0;
}

// 函数定义
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* current = &dummy;
    int carry = 0;

    while (l1 || l2 || carry) {
        int sum = (l1 ? l1->val : 0) + (l2 ? l2->val : 0) + carry;
        carry = sum / 10;

        current->next = new ListNode(sum % 10);
        current = current->next;

        if (l1) l1 = l1->next;
        if (l2) l2 = l2->next;
    }

    return dummy.next;
}
```

#### Python 语言：
```python
# 定义单链表节点
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 函数定义
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    current = dummy
    carry = 0

    while l1 or l2 or carry:
        sum_val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
        carry = sum_val // 10

        current.next = ListNode(sum_val % 10)
        current = current.next

        if l1: l1 = l1.next
        if l2: l2 = l2.next

    return dummy.next

# 创建两个链表
l1 =

 ListNode(2, ListNode(4, ListNode(3)))
l2 = ListNode(5, ListNode(6, ListNode(4)))

# 调用函数，计算结果链表
result = addTwoNumbers(l1, l2)

# 输出结果链表的值
while result:
    print(result.val, end=" ")
    result = result.next
```

## Question 7
#### 题目描述
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

- 示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

- 示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

- 示例 3:

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。

- 请设计一个时间复杂度为 O(n) 或是更優解的算法解决此问题。

#### 解题思路
这个问题可以使用`滑动窗口的思想`和`哈希表`来解决，时间复杂度为`O(n)`。滑动窗口是数组/字符串问题中常用的抽象概念。窗口通常是在数组/字符串中由开始和结束索引定义的一系列元素的集合，即[i, j)（左闭，右开）。

我们使用哈希集合来检查字符是否重复，使用两个指针表示字符串中的某个子串（的左右边界）。在每一步的操作中，我们会将左指针向右移动一格，表示我们开始枚举下一个字符作为起始位置，然后我们可以不断地向右移动右指针，但需要保证这两个指针对应的子串中没有重复的字符。在移动结束后，这个子串的长度即为当前无重复字符子串的最大长度。
以下是C++的实现：
```cpp
#include <iostream>
#include <unordered_set>
#include <string>

int lengthOfLongestSubstring(std::string s) {
    // 哈希集合，记录每个字符是否出现过
    std::unordered_set<char> occ;
    // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
    int n = s.size();
    int rk = -1, ans = 0;
    for (int i = 0; i < n; i++) {
        if (i != 0) {
            // 左指针向右移动一格，移除一个字符
            occ.erase(s[i - 1]);
        }
        while (rk + 1 < n && !occ.count(s[rk + 1])) {
            // 不断地移动右指针
            occ.insert(s[rk + 1]);
            ++rk;
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
        }
        ans = std::max(ans, rk - i + 1);
    }
    return ans;
}

int main() {
    std::string s = "abcabcbb";
    std::cout << lengthOfLongestSubstring(s) << std::endl;
    return 0;
}
```

以下是Python的实现：
```python
def lengthOfLongestSubstring(s):
    # 哈希集合，记录每个字符是否出现过
    occ = set()
    n = len(s)
    # 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
    rk, ans = -1, 0
    for i in range(n):
        if i != 0:
            # 左指针向右移动一格，移除一个字符
            occ.remove(s[i - 1])
        while rk + 1 < n and s[rk + 1] not in occ:
            # 不断地移动右指针
            occ.add(s[rk + 1])
            rk += 1
        # 第 i 到 rk 个字符是一个极长的无重复字符子串
        ans = max(ans, rk - i + 1)
    return ans
```

这个算法的时间复杂度为O(n)，其中n是字符串的长度。左指针和右指针分别会遍历整个字符串一次。

空间复杂度为O(Σ)，其中Σ表示字符集的大小。在ASCII码中，Σ为128，我们需要O(Σ)的空间来存储哈希集合。在最坏的情况下，整个字符串全部由相同的字符组成，哈希

## Question 8
#### 题目描述
链表最最常见的操作，一文读懂链表的代码操作，使用3种语言实现链表的7种操作。
链表的常见操作包括：

1. 插入

- 在头部插入： 在链表的开头添加一个新元素。
- 在尾部插入： 在链表的末尾添加一个新元素。
- 在中间插入： 在链表的任意位置插入一个新元素。
2. 删除

- 删除头节点： 删除链表的第一个元素。
- 删除尾节点： 删除链表的最后一个元素。
- 删除中间节点： 删除链表中的任意一个元素。
3. 遍历

- 正向遍历： 从链表的头部开始，依次访问每个元素，直到尾部。
- 反向遍历： 从链表的尾部开始，依次访问每个元素，直到头部。
4. 查找

- 按值查找： 在链表中查找特定值的节点。
5. 反转

- 翻转链表： 将链表的顺序颠倒过来。
6. 合并

- 合并两个有序链表： 将两个有序链表合并成一个有序链表。
7. 环检测

- 检测链表中是否存在环： 判断链表是否形成了循环。

这些基本的链表操作在算法和数据结构中经常被使用，它们构成了链表数据结构的基础。链表相对于数组的优势之一是在插入和删除元素时更为高效，但在查找元素时效率较低。

#### 解题思路
#### C++：
```cpp
#include <iostream>

class Node {
public:
    int data;
    Node* next;

    Node(int value) : data(value), next(nullptr) {}
};

// 在头部插入
Node* insertAtHead(Node* head, int value) {
    Node* newNode = new Node(value);
    newNode->next = head;
    return newNode;
}

// 在尾部插入
Node* insertAtTail(Node* head, int value) {
    Node* newNode = new Node(value);
    newNode->next = nullptr;

    if (head == nullptr) {
        return newNode;
    }

    Node* current = head;
    while (current->next != nullptr) {
        current = current->next;
    }
    current->next = newNode;

    return head;
}

// 在中间插入
Node* insertInMiddle(Node* head, int value, int position) {
    Node* newNode = new Node(value);

    if (position == 1) {
        newNode->next = head;
        return newNode;
    }

    Node* current = head;
    int count = 1;
    while (count < position - 1 && current != nullptr) {
        current = current->next;
        count++;
    }

    if (current == nullptr) {
        delete newNode;
        return head;  // 插入位置超过链表长度，不进行插入
    }

    newNode->next = current->next;
    current->next = newNode;

    return head;
}

// 删除头节点
Node* deleteHead(Node* head) {
    if (head == nullptr) {
        return nullptr;
    }
    Node* newHead = head->next;
    delete head;
    return newHead;
}

// 删除尾节点
Node* deleteTail(Node* head) {
    if (head == nullptr) {
        return nullptr;
    }

    if (head->next == nullptr) {
        delete head;
        return nullptr;
    }

    Node* current = head;
    while (current->next->next != nullptr) {
        current = current->next;
    }

    delete current->next;
    current->next = nullptr;
    return head;
}

// 删除中间节点
Node* deleteFromMiddle(Node* head, int position) {
    if (head == nullptr) {
        return nullptr;
    }

    if (position == 1) {
        Node* newHead = head->next;
        delete head;
        return newHead;
    }

    Node* current = head;
    int count = 1;
    while (count < position - 1 && current != nullptr) {
        current = current->next;
        count++;
    }

    if (current == nullptr || current->next == nullptr) {
        return head;  // 删除位置超过链表长度，不进行删除
    }

    Node* temp = current->next;
    current->next = current->next->next;
    delete temp;

    return head;
}

// 正向遍历
void traverse(Node* head) {
    while (head != nullptr) {
        std::cout << head->data << " ";
        head = head->next;
    }
    std::cout << std::endl;
}

// 反向遍历
void reverseTraverse(Node* head) {
    if (head == nullptr) {
        return;
    }

    reverseTraverse(head->next);
    std::cout << head->data << " ";
}

// 查找
Node* search(Node* head, int value) {
    while (head != nullptr) {
        if (head->data == value) {
            return head;
        }
        head = head->next;
    }
    return nullptr;
}

// 翻转链表
Node* reverseList(Node* head) {
    Node* prev = nullptr;
    Node* current = head;
    Node* next = nullptr;

    while (current != nullptr) {
        next = current->next;
        current->next = prev;
        prev = current;
        current = next;
    }

    return prev;
}

int main() {
    Node* head = nullptr;

    // 在头部插入
    head = insertAtHead(head, 3);
    head = insertAtHead(head, 2);
    head = insertAtHead(head, 1);

    // 在尾部插入
    head = insertAtTail(head, 4);
    head = insertAtTail(head, 5);

    // 在中间插入
    head = insertInMiddle(head, 10, 3);

    // 删除头节点
    head = deleteHead(head);

    // 删除尾节点
    head = deleteTail(head);

    // 删除中间节点
    head = deleteFromMiddle(head, 2);

    // 正向遍历
    std::cout << "Forward Traverse: ";
    traverse(head);

    // 反向遍历
    std::cout << "Reverse Traverse: ";
    reverseTraverse(head);
    std::cout << std::endl;

    // 查找
    Node* searchResult = search(head, 10);
    if (searchResult != nullptr) {
        std::cout << "Found: " << searchResult->data << std::endl;
    } else {
        std::cout << "Not Found" << std::endl;
    }

    // 翻转链表
    head = reverseList(head);

    // 正向遍历翻转后的链表
    std::cout << "Forward Traverse (Reversed): ";
    traverse(head);

    return 0;
}
```

#### Python：
```python
class Node:
    def __init__(self, value):
        self.data = value
        self.next = None

# 在头部插入
def insert_at_head(head, value):
    new_node = Node(value)
    new_node.next = head
    return new_node

# 在尾部插入
def insert_at_tail(head, value):
    new_node = Node(value)
    new_node.next = None

    if head is None:
        return new_node

    current = head
    while current.next is not None:
        current = current.next
    current.next = new_node

    return head

# 在中间插入
def insert_in_middle(head, value, position):
    new_node = Node(value)

    if position == 1:
        new_node.next = head
        return new_node

    current = head
    count = 1
    while count < position - 1 and current is not None:
        current = current.next
        count += 1

    if current is None:
        return head  # 插入位置超过链表长度，不进行插入

    new_node.next = current.next
    current.next = new_node

    return head

# 删除头节点
def delete_head(head):
    if head is None:
        return None
    new_head = head.next
    head.next = None
    return new_head

# 删除尾节点
def delete_tail(head):
    if head is None:
        return None

    if head.next is None:
        return None

    current = head
    while current.next.next is not None:
        current = current.next

    current.next = None
    return head

# 删除中间节点
def delete_from_middle(head, position):
    if head is None:
        return None

    if position == 1:
        new_head = head.next
        head.next = None
        return new_head

    current = head
    count = 1
    while count < position - 1 and current is not None:
        current = current.next
        count += 1

    if current is None or current.next is None:
        return head  # 删除位置超过链表长度，不进行删除

    temp = current.next
    current.next = current.next.next
    temp.next = None

    return head

# 正向遍历
def traverse(head):
    while head is not None:
        print(head.data, end=" ")
        head = head.next
    print()

# 反向遍历
def reverse_traverse(head):
    if head is None:
        return

    reverse_traverse(head.next)
    print(head.data, end=" ")

# 查找
def search(head, value):
    while head is not None:
        if head.data == value:
            return head
        head = head.next
    return None

# 翻转链表
def reverse_list(head):
    prev = None
    current = head
    next_node = None

    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev

# 主程序
head = None

# 在头部插入
head = insert_at_head(head, 3)
head = insert_at_head(head, 2)
head = insert_at_head(head, 1)

# 在尾部插入
head = insert_at_tail(head, 4)
head = insert_at_tail(head, 5)

# 在中间插入
head = insert_in_middle(head, 10, 3)

# 删除头节点
head = delete_head(head)

# 删除尾节点
head = delete_tail(head)

# 删除中间节点
head = delete_from_middle(head, 2)

# 正向遍历
print("Forward Traverse:", end=" ")
traverse(head)

# 反向遍历
print("Reverse Traverse:", end=" ")
reverse_traverse(head)
print()

# 查找
search_result = search(head, 10)
if search_result is not None:
    print("Found:", search_result.data)
else:
    print("Not Found")

# 翻转链表
head = reverse_list(head)

# 正向遍历翻转后的链表
print("Forward Traverse (Reversed):", end=" ")
traverse(head)
```

## Question 9
#### 题目描述
数组是一种基本的数据结构，它是一组相同类型的元素的集合，通过索引或者下标来访问和操作。

数组的常见操作包括：
各类操作
1. 创建数组：

- 原理： 分配一块连续的内存空间，用于存储数组元素。
- 方式： 在大多数编程语言中，通过声明数组类型和大小来创建数组。例如，在C语言中，可以使用int myArray[5];来创建一个包含5个整数的数组。
2. 访问元素：

- 原理： 数组元素存储在连续的内存地址中，可以通过索引或下标来访问。
- 方式： 使用数组的索引或下标，例如myArray[2]表示访问数组中的第三个元素。
3. 插入元素：

- 原理： 在指定位置插入新元素，需要将插入点后的元素向后移动，腾出位置。
- 方式：
在指定位置插入元素，并移动后续元素。
在末尾插入元素时，直接在数组末尾添加元素即可。

4. 删除元素：

- 原理： 删除指定位置的元素，需要将删除点后的元素向前移动，填补删除位置。
- 方式：
删除指定位置的元素，并移动后续元素。
在末尾删除元素时，直接缩小数组大小。
5. 查找元素：

- 原理： 遍历数组元素，逐一比较查找值。
- 方式：
    - 线性查找：逐一比较，直到找到匹配的元素。
    - 二分查找：仅适用于有序数组，通过比较中间元素确定查找方向。
6. 数组的遍历：

- 原理： 逐一访问数组中的每个元素。
- 方式： 使用循环结构，如for循环或while循环，从数组的第一个元素遍历到最后一个。
7. 数组的合并和拆分：

- 原理：
    - 合并：将两个数组的元素整合到一个新数组中。
    - 拆分：将一个数组的元素拆分成两个或多个数组。
- 方式：
    - 合并：创建一个新数组，将两个数组的元素逐一复制到新数组中。
    - 拆分：指定拆分点，将元素分配到不同的数组中。

#### C++:
```cpp
#include <iostream>
#include <algorithm>

int main() {
    // 1. 创建数组
    int myArray[5];

    // 2. 访问元素
    int value = myArray[2];

    // 3. 插入元素（在指定位置插入新元素）
    int newValue = 10;
    int index = 2;

    for (int i = 4; i >= index; i--) {
        myArray[i + 1] = myArray[i];
    }
    myArray[index] = newValue;

    // 4. 删除元素（在指定位置删除元素）
    int deleteIndex = 2;

    for (int i = deleteIndex; i < 4; i++) {
        myArray[i] = myArray[i + 1];
    }

    // 5. 查找元素（线性查找）
    int searchValue = 10;
    int searchIndex = -1;

    for (int i = 0; i < 5; i++) {
        if (myArray[i] == searchValue) {
            searchIndex = i;
            break;
        }
    }

    // 6. 数组的遍历
    for (int i = 0; i < 5; i++) {
        // 访问myArray[i]
    }

    // 7. 数组的合并和拆分
    int myArray1[5] = {1, 2, 3, 4, 5};
    int myArray2[5] = {6, 7, 8, 9, 10};

    // 合并数组
    int newArray[10];
    std::copy(myArray1, myArray1 + 5, newArray);
    std::copy(myArray2, myArray2 + 5, newArray + 5);

    // 拆分数组
    int splitIndex = 3;
    int newArray1[splitIndex];
    int newArray2[5 - splitIndex];
    std::copy(myArray, myArray + splitIndex, newArray1);
    std::copy(myArray + splitIndex, myArray + 5, newArray2);

    return 0;
}

```

#### Python:
```python
# 1. 创建数组
myArray = [0] * 5

# 2. 访问元素
value = myArray[2]

# 3. 插入元素（在指定位置插入新元素）
newValue = 10
index = 2

myArray.insert(index, newValue)

# 4. 删除元素（在指定位置删除元素）
index = 2
del myArray[index]

# 5. 查找元素（线性查找）
searchValue = 10
searchIndex = -1

for i in range(len(myArray)):
    if myArray[i] == searchValue:
        searchIndex = i
        break

# 6. 数组的遍历
for element in myArray:
    # 访问element

# 7. 数组的合并和拆分
myArray1 = [1, 2, 3, 4, 5]
myArray2 = [6, 7, 8, 9, 10]

# 合并数组
newArray = myArray1 + myArray2

# 拆分数组
splitIndex = 3
newArray1 = myArray[:splitIndex]
newArray2 = myArray[splitIndex:]
```

## Question 10
#### 题目描述
Stack, 中文又稱為堆疊，是一種具有後進先出（Last In First Out, LIFO）特性的抽象資料型別，常用的操作有push（壓入）、pop（彈出）、top（取得頂端元素）、empty（判斷是否為空）等。

Stack的常見應用場景包括：
1. 函數調用堆疊： 當一個函數被調用時，一個新的堆疊框架會被壓入堆疊中。當函數返回時，這個堆疊框架會被彈出。
2. 表達式求值： 在編譯器中，堆疊被用於表達式的求值。例如，中綴表達式（3 + 4）* 5的求值過程中，運算符和操作數被壓入堆疊中，然後彈出進行運算。
3. 括號匹配： 堆疊被用於括號匹配的檢查。例如，當我們讀取到一個左括號時，我們將其壓入堆疊中，當我們讀取到一個右括號時，我們彈出堆疊中的左括號，如果彈出的括號與當前的右括號不匹配，則表示括號不匹配。
4. 瀏覽器的回退功能： 瀏覽器的回退功能通常使用堆疊來實現。當我們訪問一個網頁時，它被壓入堆疊中，當我們點擊回退按鈕時，它被彈出堆疊中。

#### Stack的常見操作包括：
#### C++ 语言
```cpp
#include <iostream>
#include <vector>

class Stack {
private:
    std::vector<int> data;

public:
    // 初始化栈
    Stack() {}

    // 压栈
    void push(int value) {
        data.push_back(value);
    }

    // 弹栈
    int pop() {
        if (isEmpty()) {
            std::cerr << "Stack underflow\n";
            exit(EXIT_FAILURE);
        }
        int value = data.back();
        data.pop_back();
        return value;
    }

    // 获取栈顶元素
    int top() const {
        if (isEmpty()) {
            std::cerr << "Stack is empty\n";
            exit(EXIT_FAILURE);
        }
        return data.back();
    }

    // 判空
    bool isEmpty() const {
        return data.empty();
    }

    // 获取栈的大小
    size_t size() const {
        return data.size();
    }

    // 清空栈
    void clear() {
        data.clear();
    }
};

int main() {
    Stack stack;

    stack.push(10);
    stack.push(20);
    stack.push(30);

    std::cout << "Top element: " << stack.top() << std::endl;
    std::cout << "Stack size: " << stack.size() << std::endl;

    stack.pop();

    std::cout << "After pop, top element: " << stack.top() << std::endl;
    std::cout << "Is stack empty? " << (stack.isEmpty() ? "Yes" : "No") << std::endl;

    stack.clear();

    std::cout << "Is stack empty after clear? " << (stack.isEmpty() ? "Yes" : "No") << std::endl;

    return 0;
}
```

#### Python 语言
```python
class Stack:
    def __init__(self):
        self.data = []

    # 压栈
    def push(self, value):
        self.data.append(value)

    # 弹栈
    def pop(self):
        if self.is_empty():
            print("Stack underflow")
            exit(1)
        return self.data.pop()

    # 获取栈顶元素
    def top(self):
        if self.is_empty():
            print("Stack is empty")
            exit(1)
        return self.data[-1]

    # 判空
    def is_empty(self):
        return len(self.data) == 0

    # 获取栈的大小
    def size(self):
        return len(self.data)

    # 清空栈
    def clear(self):
        self.data = []

stack = Stack()

stack.push(10)
stack.push(20)
stack.push(30)

print("Top element:", stack.top())
print("Stack size:", stack.size())

stack.pop()

print("After pop, top element:", stack.top())
print("Is stack empty?", "Yes" if stack.is_empty() else "No")

stack.clear()

print("Is stack empty after clear?", "Yes" if stack.is_empty() else "No")
```

## Question 11
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

#### Python 实现
```python
class TreeNode:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None

# 前序遍历
def pre_order_traversal(root):
    if root is not None:
        print(root.data, end=" ")
        pre_order_traversal(root.left)
        pre_order_traversal(root.right)

# 中序遍历
def in_order_traversal(root):
    if root is not None:
        in_order_traversal(root.left)
        print(root.data, end=" ")
        in_order_traversal(root.right)

# 后序遍历
def post_order_traversal(root):
    if root is not None:
        post_order_traversal(root.left)
        post_order_traversal(root.right)
        print(root.data, end=" ")

# 在这里创建二叉树并进行测试
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print("Pre-order traversal:", end=" ")
pre_order_traversal(root)
print()

print("In-order traversal:", end=" ")
in_order_traversal(root)
print()

print("Post-order traversal:", end=" ")
post_order_traversal(root)
print()
```

## Question 12
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

int main() {
    int n = 3;
    std::cout << climbStairs(n) << std::endl;
    return 0;
}
```

#### Python 实现
``` python
def climbStairs(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2

    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]
```



在这个函数中，我们首先检查特殊情况，如果 n 为1或2，直接返回1或2，因为在这两种情况下，爬楼梯的方法数量是已知的。然后，我们初始化一个数组 dp，并使用一个循环计算 dp[i] 直到 n。

6. 示例说明

让我们以 n = 3 为例来说明整个过程：
- 初始化 dp[1] = 1 和 dp[2] = 2。
- 对于 dp[3]，根据状态转移方程，dp[3] = dp[2] + dp[1] = 2 + 1 = 3。
- 对于 dp[4]，同样使用状态转移方程，dp[4] = dp[3] + dp[2] = 3 + 2 = 5。
- 以此类推，计算到 dp[n]。
这样，我们就得到了爬楼梯问题的动态规划解决方案。这个方法避免了重复计算，提高了算法的效率。

## Question 13
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
## Question 14
#### 题目描述
给定两个按升序排列的链表，要求合并这两个链表，并且返回合并后的链表。要求新链表仍然按照升序排列。

- 示例：
```
输入：1->2->4, 1->3->5
输出：1->1->2->3->4->5
```
#### 解决思路
为了解决这个问题，我们可以考虑使用迭代或递归的方式。在这里，我将详细说明迭代的解决思路。

- 迭代法解决思路
首先，我们定义一个哑结点(dummy)，它的作用是作为合并后链表的头结点。同时，我们使用两个指针，分别指向两个待合并链表的当前节点。
遍历两个链表，比较当前节点的值，将较小的节点连接到合并链表的后面。然后将指向较小节点的指针向后移动一步。
重复上述步骤，直到其中一个链表遍历完毕。此时，将剩余未遍历的链表直接连接到合并链表的尾部。
返回合并后的链表的头结点。
#### C++ 代码
```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* current = &dummy;

        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val < l2->val) {
                current->next = l1;
                l1 = l1->next;
            } else {
                current->next = l2;
                l2 = l2->next;
            }
            current = current->next;
        }

        if (l1 != nullptr) {
            current->next = l1;
        } else {
            current->next = l2;
        }

        return dummy.next;
    }
};

int main() {
    // Example usage
    ListNode* l1 = new ListNode(1);
    l1->next = new ListNode(2);
    l1->next->next = new ListNode(4);

    ListNode* l2 = new ListNode(1);
    l2->next = new ListNode(3);
    l2->next->next = new ListNode(5);

    Solution solution;
    ListNode* mergedList = solution.mergeTwoLists(l1, l2);

    // Output the merged list
    while (mergedList != nullptr) {
        std::cout << mergedList->val << " ";
        mergedList = mergedList->next;
    }

    return 0;
}
```

#### Python 代码
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        current = dummy

        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next

        if l1 is not None:
            current.next = l1
        else:
            current.next = l2

        return dummy.next

# Example usage
l1 = ListNode(1, ListNode(2, ListNode(4)))
l2 = ListNode(1, ListNode(3, ListNode(5)))

solution = Solution()
merged_list = solution.mergeTwoLists(l1, l2)

# Output the merged list
while merged_list is not None:
    print(merged_list.val, end=" ")
    merged_list = merged_list.next
```

## Question 15
#### 题目描述
LeetCode第25题，题目为`K个一组反转链表`。给定一个链表，每k个节点一组进行翻转，如果节点总数不是k的倍数，则最后剩余的节点应该保持原有顺序。

例如，给定链表1->2->3->4->5，k=2，应返回2->1->4->3->5。如果k=3，应返回3->2->1->4->5。

#### 解决思路
这个问题可以使用迭代或递归的方式来解决。在这里，我将详细说明迭代的解决思路。

- 迭代法解决思路
首先，我们定义一个哑结点(dummy)，它的作用是作为翻转后链表的头结点。同时，我们使用两个指针，分别指向待翻转链表的前一个节点和后一个节点。
然后，我们使用一个循环来遍历链表。在每次循环中，我们首先判断剩余的链表长度是否大于等于 k。如果是，我们调用一个辅助函数 reverseKNodes 来翻转 k 个节点，并将翻转后的链表连接到原链表中。如果不是，我们直接返回原链表。
在辅助函数 reverseKNodes 中，我们使用三个指针来翻转 k 个节点。具体来说，我们首先定义一个哑结点(dummy)，然后使用两个指针分别指向待翻转链表的前一个节点和当前节点。然后，我们使用一个循环来遍历 k 个节点，每次循环中，我们将当前节点的 next 指针指向前一个节点，然后将前一个节点和当前节点向后移动一步。最后，我们将翻转后的链表的头结点返回。
#### C++ 代码
```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* reverseKNodes(ListNode* head, int k) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;

        ListNode* pre = dummy;
        ListNode* end = dummy;

        while (end->next != nullptr) {
            for (int i = 0; i < k && end != nullptr; i++) {
                end = end->next;
            }
            if (end == nullptr) break;

            ListNode* start = pre->next;
            ListNode* next = end->next;

            end->next = nullptr;
            pre->next = reverseList(start);
            start->next = next;

            pre = start;
            end = pre;
        }

        return dummy->next;
    }

    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* current = head;

        while (current != nullptr) {
            ListNode* next = current->next;
            current->next = prev;
            prev = current;
            current = next;
        }

        return prev;
    }
};

int main() {
    ListNode* head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    head->next->next->next = new ListNode(4);
    head->next->next->next->next = new ListNode(5);

    Solution solution;
    ListNode* reversedList = solution.reverseKNodes(head, 2);

    while (reversedList != nullptr) {
        std::cout << reversedList->val << " ";
        reversedList = reversedList->next;
    }

    return 0;
}
```

#### Python 代码
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKNodes(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head

        pre = dummy
        end = dummy

        while end.next is not None:
            for i in range(k):
                end = end.next
                if end is None:
                    break
            if end is None:
                break

            start = pre.next
            next = end.next

            end.next = None
            pre.next = self.reverseList(start)
            start.next = next

            pre = start
            end = pre

        return dummy.next

    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        current = head

        while current is not None:
            next = current.next
            current.next = prev
            prev = current
            current = next

        return prev

# Example usage
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))

solution = Solution()
reversed_list = solution.reverseKNodes(head, 2)

# Output the reversed list
while reversed_list is not None:
    print(reversed_list.val, end=" ")
    reversed_list = reversed_list.next
```

## Question 16
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

#### Python 代码
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Example usage
n = 5
print(factorial(n))
```

## Question 17
#### 题目描述
字母异位词分组。给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

- 示例：
```
输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```
#### 解决思路
为了解决这个问题，我们可以使用哈希表来存储每个字符串的字母异位词。具体来说，我们遍历字符串数组，对于每个字符串，我们将其排序后的结果作为哈希表的键，原始字符串作为哈希表的值。这样，我们就可以将字母异位词分组在一起。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
    std::unordered_map<std::string, std::vector<std::string>> hash_map;

    for (const std::string& str : strs) {
        std::string sorted_str = str;
        std::sort(sorted_str.begin(), sorted_str.end());
        hash_map[sorted_str].push_back(str);
    }

    std::vector<std::vector<std::string>> result;
    for (const auto& pair : hash_map) {
        result.push_back(pair.second);
    }

    return result;
}

int main() {
    std::vector<std::string> strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
    std::vector<std::vector<std::string>> result = groupAnagrams(strs);

    for (const auto& group : result) {
        std::cout << "[";
        for (const std::string& str : group) {
            std::cout << str << ",";
        }
        std::cout << "]" << std::endl;
    }
    // Output: [[eat,tea,ate,],[tan,nat,],[bat,]]

    return 0;
}
```

#### Python 代码
```python
from typing import List
import collections

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    hash_map = collections.defaultdict(list)

    for s in strs:
        sorted_str = "".join(sorted(s))
        hash_map[sorted_str].append(s)

    return list(hash_map.values())

# Example usage
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(groupAnagrams(strs))
# Output: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

## Question 18
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

#### Python 代码
```python
from typing import List

def trap(height: List[int]) -> int:
    n = len(height)
    result = 0
    stack = []

    for i in range(n):
        while stack and height[i] > height[stack[-1]]:
            top = stack.pop()
            if not stack:
                break

            distance = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[top]
            result += distance * bounded_height
        stack.append(i)

    return result

# Example usage
height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(trap(height))
# Output: 6
```

## Question 19
#### 题目描述
最長連續子數組。给定一个未排序的整数数组，找到最长连续子序列的长度。要求算法的时间复杂度为 O(n)。

- 示例：
```
输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续子序列是 [1, 2, 3, 4]。它的长度是 4。
```
#### 解决思路
为了解决这个问题，我们可以使用哈希表来存储每个元素的连续子序列的长度。具体来说，我们首先将所有元素添加到哈希表中。然后，我们遍历数组，对于每个元素，我们首先判断它的前一个元素是否在哈希表中。如果在，我们将当前元素的连续子序列的长度设置为前一个元素的连续子序列的长度加一。然后，我们更新当前元素的连续子序列的长度。最后，我们更新最长连续子序列的长度。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>

int longestConsecutive(std::vector<int>& nums) {
    std::unordered_map<int, int> hash_map;
    int result = 0;

    for (int num : nums) {
        if (hash_map.find(num) == hash_map.end()) {
            int left = hash_map.find(num - 1) != hash_map.end() ? hash_map[num - 1] : 0;
            int right = hash_map.find(num + 1) != hash_map.end() ? hash_map[num + 1] : 0;
            int sum = left + right + 1;
            hash_map[num] = sum;
            result = std::max(result, sum);
            hash_map[num - left] = sum;
            hash_map[num + right] = sum;
        }
    }

    return result;
}

int main() {
    std::vector<int> nums = {100, 4, 200, 1, 3, 2};
    std::cout << longestConsecutive(nums) << std::endl;
    return 0;
}
```

#### Python 代码
```python
from typing import List

def longestConsecutive(nums: List[int]) -> int:
    hash_map = {}
    result = 0

    for num in nums:
        if num not in hash_map:
            left = hash_map[num - 1] if num - 1 in hash_map else 0
            right = hash_map[num + 1] if num + 1 in hash_map else 0
            sum = left + right + 1
            hash_map[num] = sum
            result = max(result, sum)
            hash_map[num - left] = sum
            hash_map[num + right] = sum

    return result

# Example usage
nums = [100, 4, 200, 1, 3, 2]
print(longestConsecutive(nums))
# Output: 4
```

## Question 20
#### 题目描述
移动零。给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

- 示例：
```
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
```
#### 解决思路
为了解决这个问题，我们可以使用双指针的方式。具体来说，我们使用两个指针，一个指针用于遍历数组，另一个指针用于记录非零元素的位置。在遍历数组的过程中，我们将非零元素移动到数组的前面。最后，我们将数组剩余的位置填充为 0。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

void moveZeroes(std::vector<int>& nums) {
    int n = nums.size();
    int left = 0;
    int right = 0;

    while (right < n) {
        if (nums[right] != 0) {
            std::swap(nums[left], nums[right]);
            left++;
        }
        right++;
    }
}

int main() {
    std::vector<int> nums = {0,1,0,3,12};
    moveZeroes(nums);
    for (int i = 0; i < nums.size(); i++) {
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

#### Python 代码
```python
from typing import List

def moveZeroes(nums: List[int]) -> None:
    n = len(nums)
    left = 0
    right = 0

    while right < n:
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
        right += 1

# Example usage
nums = [0,1,0,3,12]
moveZeroes(nums)
print(nums)
# Output: [1,3,12,0,0]
```

## Question 21
#### 题目描述
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
- 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
- 返回容器可以储存的最大水量。
- 说明：你不能倾斜容器。

- 示例：
```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

#### 解决思路
为了解决这个问题，我们可以使用双指针的方式。具体来说，我们使用两个指针，分别指向数组的左右两端。然后，我们计算当前两个指针之间的水的容量。具体来说，我们首先计算两个指针之间的距离，然后计算两个指针指向的高度的最小值。最后，我们将两个指针向中间移动，直到两个指针相遇。在移动的过程中，我们不断更新水的容量的最大值。

#### C++ 代码
```cpp
#include <iostream>
#include <vector>

int maxArea(std::vector<int>& height) {
    int n = height.size();
    int left = 0;
    int right = n - 1;
    int result = 0;

    while (left < right) {
        int distance = right - left;
        int h = std::min(height[left], height[right]);
        result = std::max(result, distance * h);
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }

    return result;
}

int main() {
    std::vector<int> height = {1,8,6,2,5,4,8,3,7};
    std::cout << maxArea(height) << std::endl;
    return 0;
}
```

#### Python 代码
```python
from typing import List

def maxArea(height: List[int]) -> int:
    n = len(height)
    left = 0
    right = n - 1
    result = 0

    while left < right:
        distance = right - left
        h = min(height[left], height[right])
        result = max(result, distance * h)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return result

# Example usage
height = [1,8,6,2,5,4,8,3,7]
print(maxArea(height))
# Output: 49
```

## Question 22
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

#### Python 代码
```python
from typing import List

def threeSum(nums: List[int]) -> List[List[int]]:
    result = []
    n = len(nums)
    nums.sort()

    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left = i + 1
        right = n - 1

        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif sum < 0:
                left += 1
            else:
                right -= 1

    return result

# Example usage
nums = [-1,0,1,2,-1,-4]
print(threeSum(nums))
# Output: [[-1,-1,2],[-1,0,1]]
```




## Question 25
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

#### Python 代码
```python
def bubbleSort(nums):
    n = len(nums)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]

# Example usage
nums = [5, 2, 9, 1, 5, 6]
bubbleSort(nums)
print(nums)
# Output: [1, 2, 5, 5, 6, 9]
```

## Question 26
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

#### Python 代码
```python

def selectionSort(nums):
    n = len(nums)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            if nums[j] < nums[min_index]:
                min_index = j
        nums[i], nums[min_index] = nums[min_index], nums[i]

# Example usage
nums = [5, 2, 9, 1, 5, 6]
selectionSort(nums)
print(nums)
# Output: [1, 2, 5, 5, 6, 9]
```

## Question 27
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

#### Python 代码
```python

def insertionSort(nums):
    n = len(nums)
    for i in range(1, n):
        key = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = key

# Example usage
nums = [5, 2, 9, 1, 5, 6]
insertionSort(nums)
print(nums)
# Output: [1, 2, 5, 5, 6, 9]
```

## Question 28
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

#### Python 代码
```python

def merge(nums, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid
    L = [0] * n1
    R = [0] * n2
    for i in range(n1):
        L[i] = nums[left + i]
    for j in range(n2):
        R[j] = nums[mid + 1 + j]
    i = 0
    j = 0
    k = left
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            nums[k] = L[i]
            i += 1
        else:
            nums[k] = R[j]
            j += 1
        k += 1
    while i < n1:
        nums[k] = L[i]
        i += 1
        k += 1
    while j < n2:
        nums[k] = R[j]
        j += 1
        k += 1

def mergeSort(nums, left, right):
    if left < right:
        mid = left + (right - left) // 2
        mergeSort(nums, left, mid)
        mergeSort(nums, mid + 1, right)
        merge(nums, left, mid, right)

# Example usage
nums = [5, 2, 9, 1, 5, 6]
mergeSort(nums, 0, len(nums) - 1)
print(nums)
# Output: [1, 2, 5, 5, 6, 9]
```

## Question 29
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

#### Python 代码
```python

def partition(nums, low, high):
    pivot = nums[high]
    i = low - 1
    for j in range(low, high):
        if nums[j] < pivot:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1], nums[high] = nums[high], nums[i + 1]
    return i + 1

def quickSort(nums, low, high):
    if low < high:
        pi = partition(nums, low, high)
        quickSort(nums, low, pi - 1)
        quickSort(nums, pi + 1, high)

# Example usage
nums = [5, 2, 9, 1, 5, 6]
quickSort(nums, 0, len(nums) - 1)
print(nums)
# Output: [1, 2, 5, 5, 6, 9]
```

## Question 30
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

#### Python 代码
```python

def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heapSort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

# Example usage
arr = [5, 2, 9, 1, 5, 6]
heapSort(arr)
print(arr)
# Output: [1, 2, 5, 5, 6, 9]
```

## Question 31
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
    visited.insert(node);
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

#### Python 代码
```python

def dfs(node, graph, visited, result):
    visited.add(node)
    result.append(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, graph, visited, result)

def depthFirstSearch(graph):
    visited = set()
    result = []
    for node in graph:
        if node not in visited:
            dfs(node, graph, visited, result)
    return result

# Example usage
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
print(depthFirstSearch(graph))
# Output: [0, 1, 2, 3]
```

## Question 32
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

#### Python 代码
```python

def breadthFirstSearch(graph):
    visited = set()
    result = []
    for node in graph:
        if node not in visited:
            q = [node]
            visited.add(node)
            while q:
                curr = q.pop(0)
                result.append(curr)
                for neighbor in graph[curr]:
                    if neighbor not in visited:
                        q.append(neighbor)
                        visited.add(neighbor)
    return result

# Example usage
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
print(breadthFirstSearch(graph))

# Output: [0, 1, 2, 3]
```

## Question 33
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

#### Python 代码
```python

def maxSubArray(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

# Example usage
nums = [1, 2, 3, 4, 5]
print(maxSubArray(nums))
# Output: 15
```

## Question 34
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

#### Python 代码
```python

def maxProfit(prices):
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += prices[i] - prices[i - 1]
    return max_profit

# Example usage
prices = [7, 1, 5, 3, 6, 4]
print(maxProfit(prices))
# Output: 7
```

## Question 35
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

#### Python 代码
```python

import heapq
import math

def dijkstra(graph, source):
    dist = [math.inf] * len(graph)
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        for v, weight in graph[u].items():
            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    return dist

# Example usage
graph = {
    0: {1: 4, 2: 1},
    1: {3: 1},
    2: {1: 2, 3: 5},
    3: {}
}
print(dijkstra(graph, 0))
# Output: [0, 3, 1, 2]
```

## Question 36
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

#### Python 代码
```python

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if u != self.parent[u]:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def merge(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            if self.rank[pu] > self.rank[pv]:
                self.parent[pv] = pu
            elif self.rank[pu] < self.rank[pv]:
                self.parent[pu] = pv
            else:
                self.parent[pu] = pv
                self.rank[pv] += 1

def kruskal(graph):
    result = []
    edges = []
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            edges.append((weight, u, v))
    edges.sort()
    ds = DisjointSet(len(graph))
    for weight, u, v in edges:
        if ds.find(u) != ds.find(v):
            ds.merge(u, v)
            result.append(u)
            result.append(v)
    return result

# Example usage
graph = {
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

print(kruskal(graph))
# Output: [0, 1, 2, 8, 6, 5, 4, 3, 7]
```

## Question 37
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

#### Python 代码
```python

import heapq
import math

def prim(graph):
    result = []
    dist = [math.inf] * len(graph)
    visited = [False] * len(graph)
    pq = [(0, 0)]
    dist[0] = 0
    while pq:
        d, u = heapq.heappop(pq)
        visited[u] = True
        for v, weight in graph[u].items():
            if not visited[v] and weight < dist[v]:
                dist[v] = weight
                heapq.heappush(pq, (dist[v], v))
    return dist

# Example usage
graph = {
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
print(prim(graph))
# Output: [0, 4, 8, 7, 1, 2, 5, 9, 2]
```

## Question 38
#### 题目描述



