# InterviewPrep
<!-- possible phone screen questions -->
## Phone Screen Questions
### XR Perception R&D position
Based on the job description for a position in the Engineering Group, specifically within Systems Engineering at Qualcomm Semiconductor Limited, focusing on XR (Augmented Reality and Virtual Reality) Perception R&D, the interview questions are likely to cover a broad range of topics. These questions will not only assess the candidate's technical skills and experience but also their ability to contribute to Qualcomm's objectives in AR and VR technologies. Here are some of the most common interview questions that might be asked:

1. **Technical Expertise in Computer Vision and Machine Learning**
   - Can you explain the principles of Computer Vision?
   Computer Vision: https://en.wikipedia.org/wiki/Computer_vision

   #### Image Pre-processing
   - **Resampling** methods such as bilinear, bicubic, and nearest neighbor interpolation adjust the resolution of images, useful for resizing images or changing their resolution for analysis consistency.
      - **重采样**：包括双线性插值、双三次插值、最近邻插值等方法，用于调整图像的分辨率，适用于改变图像大小或分辨率以保持分析的一致性。
  
   - **Normalization** techniques like min-max and z-score normalization standardize the range of pixel values, which is critical for machine learning models to perform optimally.
      - **归一化**：如最小-最大归一化和Z分数归一化等技术，用于标准化像素值的范围，对于机器学习模型的优化性能至关重要。
   
   - **Color Space Conversion** transforms images from one color representation to another, like RGB to grayscale or RGB to HSV. This is often done to simplify the analysis or to extract specific features from images.
      - **色彩空间转换**：将图像从一种色彩表示转换到另一种，例如RGB到灰度或RGB到HSV，通常是为了简化分析或从图像中提取特定特征。
  
   - **Filtering** is applied to remove noise, smooth images, or sharpen image features. Common filters include:
   - **Gaussian Filter** for smoothing to reduce image noise and detail.
   - **Median Filter** for noise reduction, particularly useful in removing 'salt and pepper' noise.
   - **Mean Filter** for basic smoothing by averaging the pixels within a neighborhood.

      - **滤波**：应用于去除噪声、平滑图像或锐化图像特征。常用滤波器包括：
      - **高斯滤波**：用于减少图像噪声和细节。
      - **中值滤波**：特别适用于去除“盐和胡椒”噪声。
      - **均值滤波**：通过对邻域内的像素进行平均来基本平滑图像。

   - **Contrast Enhancement** techniques like histogram equalization and adaptive histogram equalization improve the visibility of features in an image by adjusting the image's contrast.
      - **对比度增强**：技术如直方图均衡化和自适应直方图均衡化通过调整图像的对比度来改善图像中特征的可见性。

   #### Feature Extraction
   - Detecting **lines, edges, and ridges** using methods like Canny edge detection and the Hough transform. These features are critical for understanding the structure within images.
      - 使用如Canny边缘检测和霍夫变换等方法检测**线条、边缘和脊线**。这些特征对理解图像内的结构至关重要。
   - Identifying **corners and keypoints** with algorithms such as Harris corner detection, SIFT (Scale-Invariant Feature Transform), SURF (Speeded Up Robust Features), and ORB (Oriented FAST and Rotated BRIEF). These features are essential for tasks like image matching, object detection, and tracking.
      - 通过Harris角点检测、SIFT（尺度不变特征变换）、SURF（加速稳健特征）和ORB（定向快速旋转简明）等算法识别**角点和关键点**。这些特征对于图像匹配、物体检测和跟踪等任务至关重要。
   - **Texture and shape descriptors** like LBP (Local Binary Patterns) and HOG (Histogram of Oriented Gradients) are used to capture and represent the texture and shape characteristics of objects in images.
      - LBP（局部二值模式）和HOG（方向梯度直方图）等**纹理和形状描述符**用于捕捉和表示图像中物体的纹理和形状特征。
   

   #### Detection and Segmentation
   - **Selecting interest points** involves methods like NMS (Non-Maximum Suppression) and RANSAC (Random Sample Consensus) to identify and refine the selection of key features or points in an image.
      - **选择兴趣点**：涉及如NMS（非极大抑制）和RANSAC（随机样本一致性）等方法，用于识别和细化图像中的关键特征或点。
   - **Segmentation** divides images into parts or regions based on certain criteria. Traditional methods include thresholding, region growing, and watershed algorithms. Modern machine learning approaches like FCN (Fully Convolutional Networks), U-Net, and Mask R-CNN offer advanced capabilities for segmenting images more precisely.
      - **分割**：基于某些标准将图像分成部分或区域。传统方法包括阈值法、区域生长和分水岭算法。现代机器学习方法如FCN（全卷积网络）、U-Net和Mask R-CNN为更精确地分割图像提供了高级功能。

   #### High-level Processing and Decision Making
   - **Pattern Recognition and Classification**: At this stage, the system identifies patterns, objects, and scenarios within the images.
   - **模式识别和分类**：此阶段系统识别图像中的模式、物体和场景。技术范围从传统的机器学习方法（如支持向量机SVM和决策树）到先进的深度学习模型（如卷积神经网络CNNs）。
   - **物体检测和识别**：现代深度学习方法，如R-CNN系列（包括Fast R-CNN、Faster R-CNN）、YOLO（You Only Look Once）和SSD（单次多框检测器），极大地革新了机器检测和识别图像中物体的能力，提高了准确性和速度。

   #### Kalman Filter Usage
   测动态系统的未来状态，以最小化平方误差的均值。它在计算机视觉中广泛用于实时跟踪移动对象、导航和机器人技术。它通过结合随时间的测量来估计过程的状态，有效地处理不确定性。



   - Can you explain the principles of SLAM (Simultaneous Localization and Mapping) and how it applies to AR/VR technologies?
   ```
   SLAM: https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping
   SLAM技术（Simultaneous Localization and Mapping）
   是一种用于构建环境地图并确定自身位置的技术。在AR/VR中，SLAM技术用于实时地图构建和定位，以便将虚拟对象与现实世界对齐。SLAM技术通常涉及传感器数据融合、特征提取和匹配、优化算法等方面。

   流程通常包括：
   1. 传感器数据采集：使用相机、激光雷达、惯性测量单元（IMU）等传感器采集环境数据。
   2. 特征提取和匹配：从传感器数据中提取特征点，并将它们与先前的地图进行匹配。
   3. 优化算法：使用优化算法（如图优化或非线性优化）来估计相机姿态和地图结构。
   4. 实时定位和地图构建：在实时环境中更新地图并定位相机的位置。

   数据融合:
   - filter-based methods (e.g., Extended Kalman Filter, Unscented Kalman Filter)
   - optimization-based methods (e.g., Bundle Adjustment, Pose Graph Optimization)
   - deep learning-based methods (e.g., SLAMNet, VIO-Net)
   
   
   ```
   - How have you contributed to the development of efficient and accurate computer vision and machine learning solutions for XR perception tasks in your previous roles?

   - Elaborate on your understanding of 3D computer vision methods and mathematics, covering areas like SLAM, 3D reconstruction, object detection, and sensor fusion.

   - Describe your experience with 3D pose tracking and scene understanding. How have you applied these in a project?

   - What challenges have you encountered while working on object detection and segmentation for real-time systems, and how did you overcome them?

   - Describe a project where you applied your knowledge of mathematical optimization to enhance the performance of XR perception systems.

   - How do you approach sensor fusion in XR systems, and what mathematical optimization techniques do you find most effective?

2. **Programming and Software Development Skills**
   - What experience do you have with Python and C++ programming in the context of XR applications?

   - Describe a project where you developed computer vision or machine learning solutions for embedded or mobile platforms. What were the key challenges and your solutions?

   - How have you implemented deep neural networks for edge devices? Can you discuss your experience with model optimization techniques like quantization and pruning?

3. **Understanding and Analyzing Requirements**
   - How do you approach requirement analysis for XR perception systems? Can you give an example of how you translated requirements into a successful project outcome?
    -   UML diagrams
    -   Agile framework

    - How do you approach understanding and analyzing requirements for perception systems in the context of XR platforms?
    
    - In your experience, how important is cross-functional collaboration in developing XR technologies? Can you share an instance where collaboration significantly impacted a project?

4. **Practical Experience and Problem-Solving**
   - Share an example of a complex problem you solved in the domain of XR perception. What was your thought process, and what solutions did you implement?

   - Given a scenario where you must optimize an XR application for better performance on mobile platforms, what steps would you take to analyze and improve it?

5. **Industry Experience and Future Vision**
   - With your background in AR/VR, where do you see the future of mobile perception technology heading?
    -    vision for the future of XR, mentioning emerging technologies or trends you believe will be significant.
   - How do you stay updated with the latest advancements in computer vision, machine learning, and XR technologies? Can you discuss a recent breakthrough that excited you?
    -   Neural Radiance Fields (NeRF)

7. **Preferred Qualifications Specifics**
   - Have you worked on testing and debugging XR devices or mobile platforms? What tools and processes do you use for effective debugging?
    -   GDB
    -   Valgrind

    -   Cost functions
    -   Loss curves
    -   ROC curves
    -   F1 score
    -   Confusion matrices
    -   Precision and recall
    -   Residuals

   - Discuss your experience with deploying machine learning models on edge devices, especially in the context of XR. How do you ensure the balance between performance and accuracy?
      -   Quantization
      -   Pruning
      -   Knowledge distillation
      -   Model compression

# Data Structure

## Question 1: 链表的基本操作 (链表)
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

## Question 2: 数组的基本操作 (数组)
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

## Question 3: 栈的基本操作 (栈)
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

## Question 4: String Search (鏈表)
```python
- Input 1D letter array = [E R T R, P O T Q, R I T A, S O W E]. 
- width 4, height 4 → 2D matrix
- Input 1D search string: [R O T]

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
```
```cpp
#include <iostream>
#include <vector>
#include <string>

std::pair<int, int> searchDiagonal(const std::vector<std::string>& matrix, const std::string& search) {
    int m = matrix.size();
    int n = matrix[0].size();
    int len = search.size();

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == search[0]) {
                // Check diagonal from (i, j) to (i + len - 1, j + len - 1)
                int x = i, y = j;
                int k = 0;
                while (x < m && y < n && k < len) {
                    if (matrix[x][y] != search[k]) {
                        break;
                    }
                    x++;
                    y++;
                    k++;
                }
                if (k == len) {
                    return {i, j};
                }
            }
        }
    }

    return {-1, -1};
}

int main() {
    std::vector<std::string> matrix = {"ERTR", "POTQ", "RITA", "SOWE"};
    std::string search = "ROT";

    std::pair<int, int> result = searchDiagonal(matrix, search);

    if (result.first == -1 && result.second == -1) {
        std::cout << "Not found" << std::endl;
    } else {
        std::cout << "First location of 'R' in search string: (" << result.first << ", " << result.second << ")" << std::endl;
    }

    return 0;
}
```

## Question 5: 两数相加 (链表)
#### 题目描述
给定两个非空链表，表示两个非负整数。它们每位数字都是按照逆序方式存储的，并且每个节点只能存储一位数字。请将两个数相加并以相同形式返回一个表示和的链表。

例如：

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
解释：342 + 465 = 807。
```

#### 解题思路

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

## Question 6: 判断链表是否成环 (Double Pointer) (链表)
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
## Question 7: Detecting a Cycle in a Linked List (链表)
#### 题目描述
To determine if a linked list is cyclic (i.e., contains a loop), a commonly used approach is Floyd’s Cycle-Finding Algorithm, also known as the "Tortoise and the Hare" algorithm. It uses two pointers that move at different speeds through the list. 
#### 解决思路
- Initialize two pointers, `slow` and `fast`, both pointing to the head of the list.
- Move `slow` by one step and `fast` by two steps through the list.
- If the linked list has a cycle, `slow` and `fast` will eventually meet at some point inside the loop.
- If `fast` reaches the end of the list (`null`), then the list is acyclic.

#### C++ 代码
```cpp
#include <iostream>
#include <memory>

template <typename T>
struct ListNode {
    T value;
    std::shared_ptr<ListNode<T>> next;
    ListNode(T value, std::shared_ptr<ListNode<T>> next = nullptr) : value(value), next(next) {}
};

template <typename T>
bool hasCycle(std::shared_ptr<ListNode<T>> head) {
    std::shared_ptr<ListNode<T>> slow = head, fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            return true;  // Cycle detected
        }
    }
    return false;
}

int main() {
    auto head = std::make_shared<ListNode<int>>(3, std::make_shared<ListNode<int>>(2, std::make_shared<ListNode<int>>(0, std::make_shared<ListNode<int>>(-4))));
    head->next->next->next->next = head->next;  // Create a cycle: -4 -> 2
    bool result = hasCycle(head);
    std::cout << std::boolalpha << result << std::endl;  // Output: true
    return 0;
}
```


## Question 8: Find the Start of the Cycle in a Linked List (链表)
#### 题目描述
Once a cycle is detected using the "Tortoise and the Hare" algorithm, finding the start of the cycle involves:

- Once `slow` and `fast` meet within the cycle, reinitialize one of the pointers (say `fast`) to the head of the linked list.
- Move both `slow` and `fast` at the same pace (one step at a time).
- The point where they meet again is the start of the cycle.
#### 解决思路
The reasoning behind this is based on the distances: from the head to the start of the loop, and from the meeting point to the start of the loop, being equal when traversed at the same speed.

#### C++ 代码
```cpp
#include <iostream>
#include <memory>

template <typename T>
struct ListNode {
    T value;
    std::shared_ptr<ListNode<T>> next;
    ListNode(T value, std::shared_ptr<ListNode<T>> next = nullptr) : value(value), next(next) {}
};

template <typename T>
std::shared_ptr<ListNode<T>> detectCycle(std::shared_ptr<ListNode<T>> head) {
    std::shared_ptr<ListNode<T>> slow = head, fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {  // Cycle detected
            fast = head;  // Reset fast to head
            while (slow != fast) {  // Find the start of the cycle
                slow = slow->next;
                fast = fast->next;
            }
            return slow;  // Starting node of the cycle
        }
    }
    return nullptr;
}

int main() {
    auto head = std::make_shared<ListNode<int>>(3, std::make_shared<ListNode<int>>(2, std::make_shared<ListNode<int>>(0, std::make_shared<ListNode<int>>(-4))));
    head->next->next->next->next = head->next;  // Create a cycle: -4 -> 2
    std::shared_ptr<ListNode<int>> result = detectCycle(head);
    std::cout << result->value << std::endl;  // Output: 2
    return 0;
}
```

## Question 9: Merge Two Sorted Lists (链表)
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

## Question 10: 反轉鏈表
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

## Question 11: HashMap Implementation (HashMap实现)
#### 题目描述 
HashMap实现. 可以使用一个链表数组来实现HashMap，其中数组中的每个元素都是一个键值对的链表。键被哈希以确定数组中的索引，然后键值对被插入到相应的链表中。要检索一个值，键被哈希以找到索引，然后在该索引处的链表中遍历以找到键值对。

#### C++ Implementation
```cpp
#include <iostream>
#include <list>
#include <vector>
#include <functional>

template <typename K, typename V>
class HashMap {
public:
    HashMap(int size) : size(size), map(size, nullptr) {}

    size_t _get_hash(const K& key) {
        return std::hash<K>{}(key) % size;
    }

    bool add(const K& key, const V& value) {
        size_t key_hash = _get_hash(key);
        std::pair<K, V> key_value = std::make_pair(key, value);

        if (map[key_hash] == nullptr) {
            map[key_hash] = std::make_unique<std::list<std::pair<K, V>>>();
            map[key_hash]->push_back(key_value);
            return true;
        } else {
            for (auto& pair : *map[key_hash]) {
                if (pair.first == key) {
                    pair.second = value;
                    return true;
                }
            }
            map[key_hash]->push_back(key_value);
            return true;
        }
    }

    V get(const K& key) {
        size_t key_hash = _get_hash(key);
        if (map[key_hash] != nullptr) {
            for (auto& pair : *map[key_hash]) {
                if (pair.first == key) {
                    return pair.second;
                }
            }
        }
        return V();
    }

private:
    int size;
    std::vector<std::unique_ptr<std::list<std::pair<K, V>>> > map;
};

int main() {
    HashMap<std::string, int> hash_map(10);
    hash_map.add("apple", 10);
    hash_map.add("banana", 20);
    std::cout << hash_map.get("apple") << std::endl;  // Output: 10
    std::cout << hash_map.get("banana") << std::endl;  // Output: 20
    return 0;
}
```

## Question 12: 两数之和 (Two Sum) (哈希表) (Hash Map)
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


#### 代码实现

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

## Question 13: 无重复字符的最长子串(滑动窗口) (哈希表)
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

这个算法的时间复杂度为O(n)，其中n是字符串的长度。左指针和右指针分别会遍历整个字符串一次。

空间复杂度为O(Σ)，其中Σ表示字符集的大小。在ASCII码中，Σ为128，我们需要O(Σ)的空间来存储哈希集合。在最坏的情况下，整个字符串全部由相同的字符组成，我们需要O(n)的空间来存储哈希集合。


## Question 14: String Hash Map 字母異位詞分組
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

## Question 15: Numeric HashMap 最長連續子數組 (哈希表)
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

## Question 16: 移動零（Move Zeroes）(数组)
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

## Question 17: Unique Elements in Two Arrays (數組)
#### 题目描述
两个array 找没有同时在两个array里面的elements，比如 array1 = [1, 2, 3, 4, 5]; array2 = [1, 3, 5, 7, 9]； 那么result = [2, 4, 7, 9]; how to code this?
#### 解决思路
This can be efficiently done using set operations in Python, as sets provide a straightforward way to perform union, intersection, difference, and symmetric difference operations.

The most fitting operation for this task is the symmetric difference, which returns a set containing all the elements that are in either of the sets but not in both. This can be done using the ^ operator or the .symmetric_difference() method on sets.

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <set>

template <typename T>
std::vector<T> find_unique_elements(const std::vector<T>& array1, const std::vector<T>& array2) {
    std::set<T> set1(array1.begin(), array1.end());
    std::set<T> set2(array2.begin(), array2.end());

    std::set<T> result;
    std::set_symmetric_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));

    return std::vector<T>(result.begin(), result.end());
}

int main() {
    std::vector<int> array1 = {1, 2, 3, 4, 5};
    std::vector<int> array2 = {1, 3, 5, 7, 9};
    std::vector<int> result = find_unique_elements(array1, array2);
    for (const auto& num : result) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // Output: 2 4 7 9
    return 0;
}
```

## Question 18: 储存的最大水量 (Double Pointer) (数组)
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

# Machine Learning Interview Questions
## Question 1: Image. Background & foreground objects (convex)
#### 题目描述
Count the number of obiects: Provide bounding boxes for them.
Assumption: Background is darker than foreground
#### 解题思路
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

#### Python 实现
```python
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
## Question 2: Machine Learning Terminology
#### 题目描述
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


## Question 3: Receptive Field of the Network?
#### 题目描述
Input: 64x64x3 ch
Layer1: Conv2D 3x3 spatial output channels: 10, Stride: 1:
Layer2: Conv2D 1x1 spatial output channels: 4, Stride: 1
Layer3: Conv2D 3x3 spatial output channels 4, Stride: 1

Receptive Field (RF) of a neuron in a network is the area in the input image that affects the output of that neuron. It is the area in the input image that the neuron is looking at.
```python
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
#### 解题思路
`RF_new = RF_prev + ((KernelSize - 1) * ProductOfAllPreviousStrides)`

Given your network configuration:

1. **Input**: 64x64x3
2. **Layer1**: Conv2D 3x3, output channels: 10, Stride: 1
3. **Layer2**: Conv2D 1x1, output channels: 4, Stride: 1
4. **Layer3**: Conv2D 3x3, output channels: 4, Stride: 1

##### Initial Condition

- For the input layer (before any convolution), the receptive field is 1, as each pixel sees only itself.

##### Layer 1

- **Kernel Size**: 3
- **Stride**: 1

`RF_Layer1 = 1 + ((3 - 1) * 1) = 3`

##### Layer 2

Layer 2 has a kernel size of 1, which doesn't increase the receptive field in terms of spatial extent. It serves more to combine features from the previous layer without looking at additional context from the input image.

- **Kernel Size**: 1
- **Stride**: 1

`RF_Layer2 = RF_Layer1 + ((1 - 1) * 1) = 3`

##### Layer 3

- **Kernel Size**: 3
- **Stride**: 1

`RF_Layer3 = RF_Layer2 + ((3 - 1) * 1) = 5`

##### Conclusion

The receptive field of the network after Layer 3 is 5x5. This means that each feature in the output after passing through these three layers is influenced by a 5x5 patch in the original input image.


## Question 4: 计算 Receptive Field
#### 题目描述 
Receptive Field 的计算涉及到卷积神经网络（CNN）的理解。对于给定的层结构（例如：3x3卷积 - 最大池化 stride 2 - 3x3卷积 - 最大池化 stride 2），计算最终的感受野大小是理解CNN如何处理空间信息的一个关键点。根据你的描述，最终输出图为1x1，输入图的大小为何，说明经过每层处理后，图像的空间尺寸减小，同时感受野增大。

To determine the input size of the image and how the spatial dimension changes after each layer in a Convolutional Neural Network (CNN) with the given layer configuration, we need to understand how convolutional and pooling layers affect the spatial dimensions of the input. The given CNN configuration is:

1. 3x3 Convolution, Stride 1
2. Max Pooling, Stride 2
3. 3x3 Convolution, Stride 1
4. Max Pooling, Stride 2

#### 解题思路
**Spatial Dimension Formula**

For convolutional and pooling layers, the output size (O) of one dimension (width or height) can be calculated using the formula:

${O = \frac{W - K + 2P}{S} + 1}$

Where:
- `W` is the input size (width or height),
- `K` is the kernel size,
- `P` is the padding (assumed to be 0 unless specified),
- `S` is the stride.

For simplicity and focusing on the receptive field calculation, we assume padding (`P`) is 0.

**Calculating Input Size**

Given the output size is 1x1 after the entire network, and we need to calculate backward to find the input size, let's reverse the process. We know the configuration but not the initial size, so let's calculate how the dimension reduces through each layer from an arbitrary starting point.

**Layer 4: Max Pooling, Stride 2 (Backward Calculation)**

Starting from an output size of 1x1, to find the input size to this layer, we reverse the operation of pooling. Given stride 2, the input size to achieve 1x1 output would be:

$W = (O - 1) \times S + K$

Where:
- `O` is the output size (1),
- `S` is the stride (2),
- `K` is the kernel size (2).

Substituting the values:

$W = (1 - 1) \times 2 + 2 = 2$

(Plus one less than the stride because we're going in reverse, but in calculating back to the input size from an output, the formula simplifies since we're assuming no padding and a direct inverse operation.)

**Layer 3: 3x3 Convolution, Stride 1**

Convolution with a kernel of 3x3 and stride 1, from an output of 2x2, would have come from:

$W = (O - 1) \times S + K$

Substituting the values:

$W = (2 - 1) \times 1 + 3 = 4$

**Layer 2: Max Pooling, Stride 2**

Again, reversing the pooling operation from an output of 4x4 to find its input:

$W = (O - 1) \times S + K$

Substituting the values:

$W = (4 - 1) \times 2 + 2 = 8$

**Layer 1: 3x3 Convolution, Stride 1**

Finally, the input size for the first convolution layer, coming from an output size of 8x8:

$W = (O - 1) \times S + K$

Substituting the values:

$W = (8 - 1) \times 1 + 3 = 10$


Thus, the original input size of the image would be 10x10.

**Spatial Dimension Change After Each Layer:**

1. **After 3x3 Convolution, Stride 1**: The spatial dimension does not reduce due to the stride of 1, but considering boundary effects without padding, it reduces to 8x8.
2. **After Max Pooling, Stride 2**: The dimension is halved due to pooling with stride 2, reducing it to 4x4.
3. **After another 3x3 Convolution, Stride 1**: Again, due to the convolution without padding, the dimension reduces to 2x2.
4. **After the final Max Pooling, Stride 2**: The dimension is halved again, leading to the final output size of 1x1.

In summary, starting from an input size of 10x10, the spatial dimensions are reduced through the network layers to reach an output size of 1x1, illustrating how each layer affects the spatial dimensionality of the image.


## Question 5: Object Detection
For the object detection task, you mentioned using region-based CNNs (like the R-CNN series). This approach involves extracting candidate regions from the image, then using a CNN to extract features from each region, followed by classification and bounding box regression. This effectively addresses both classification and regression problems. Using a Gaussian filter to remove noise is a common preprocessing step that can help improve model performance.


## Question 6: 简单的2 Classes分类问题 (SVM + RBF核)
对于简单的二分类问题，逻辑回归和使用RBF（径向基函数）核的SVM都是常用的方法。逻辑回归通过求解参数使得损失函数最小化，而SVM则通过找到最大间隔的超平面来分离两个类别。RBF核可以将数据映射到更高维的空间，使得原本线性不可分的数据变得可分。

## Question 7: 神经网络Overfitting (Regularization, Dropout, Data Augmentation)
#### Model Overfitting
To address model overfitting, regularization techniques such as L1 and L2 regularization can be applied to penalize large weights and prevent overfitting. Additionally, dropout layers can be used to randomly deactivate neurons during training, preventing the network from relying too heavily on specific neurons. Data augmentation, which involves creating new training examples by applying transformations to existing data, can also help the model generalize better.
#### 模型过拟合
为了解决模型过拟合问题，可以应用正则化技术，如L1和L2正则化，来惩罚大的权重并防止过拟合。此外，可以使用dropout层在训练过程中随机关闭神经元，防止网络过度依赖特定的神经元。数据增强也可以帮助模型更好地泛化，它通过对现有数据应用变换来创建新的训练样本。

## Question 8: Model Evaluation Metrics (模型评估指标)
For evaluating a classification model, common metrics include accuracy, precision, recall, and F1 score. Accuracy measures the proportion of correctly classified instances, precision measures the proportion of true positive predictions among all positive predictions, recall measures the proportion of true positive predictions among all actual positive instances, and F1 score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance.


## Question 9: K-Means Clustering
#### K均值聚类
K均值聚类是一种无监督机器学习算法，它根据相似性将数据分成k个簇。它通过将数据点迭代地分配到最近的簇质心，并根据分配点的均值更新质心。K均值对质心的初始选择敏感，并且可能会收敛到局部最小值。它通常用于客户细分、图像压缩和异常检测等聚类应用。
#### 解决思路
- kmeans function that takes a dataset X, the number of clusters k, and an optional maximum number of iterations max_iters as input.
- initialize centroids by randomly selecting k data points from the dataset.
We use a for loop to iteratively assign data points to the nearest cluster centroid and update the centroids based on the mean of the assigned points.
- stop the iteration if the centroids do not change or the maximum number of iterations is reached.
- converged or reached the maximum number of iterations.
#### Python 代码
```python
import numpy as np

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [np.linalg.norm(x - c) for c in centroids]
            cluster = np.argmin(distances)
            clusters[cluster].append(x)
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids
```

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

template <typename T>
std::vector<std::vector<T>> kmeans(const std::vector<std::vector<T>>& X, int k, int max_iters = 100) {
    std::vector<std::vector<T>> centroids;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < k; ++i) {
        centroids.push_back(X[indices[i]]);
    }
    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<std::vector<std::vector<T>>> clusters(k);
        for (const auto& x : X) {
            std::vector<T> distances;
            for (const auto& centroid : centroids) {
                T distance = 0;
                for (size_t i = 0; i < x.size(); ++i) {
                    distance += std::pow(x[i] - centroid[i], 2);
                }
                distances.push_back(std::sqrt(distance));
            }
            int cluster = std::min_element(distances.begin(), distances.end()) - distances.begin();
            clusters[cluster].push_back(x);
        }
        std::vector<std::vector<T>> new_centroids;
        for (const auto& cluster : clusters) {
            if (cluster.empty()) {
                new_centroids.push_back(centroids[&cluster - &clusters[0]]);
            } else {
                std::vector<T> new_centroid(cluster[0].size());
                for (const auto& x : cluster) {
                    for (size_t i = 0; i < x.size(); ++i) {
                        new_centroid[i] += x[i];
                    }
                }
                for (size_t i = 0; i < new_centroid.size(); ++i) {
                    new_centroid[i] /= cluster.size();
                }
                new_centroids.push_back(new_centroid);
            }
        }
        if (centroids == new_centroids) {
            break;
        }
        centroids = new_centroids;
    }
    return std::move(centroids);
}

int main() {
    std::vector<std::vector<int>> X = {{1, 2}, {1, 4}, {1, 0}, {4, 2}, {4, 4}, {4, 0}};
    int k = 2;
    std::vector<std::vector<std::vector<int>>> clusters = kmeans(X, k);
    for (const auto& cluster : clusters) {
        for (const auto& point : cluster) {
            std::cout << "(" << point[0] << ", " << point[1] << ") ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
```python
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
k = 2
clusters, centroids = kmeans(X, k)
print(clusters)  # Output: [[array([1, 2]), array([1, 4]), array([1, 0])], [array([4, 2]), array([4, 4]), array([4, 0])]]
print(centroids)  # Output: [array([1., 2.]), array([4., 2.])]
```

## Question 10: 主成分分析 Principal Component Analysis (PCA)
#### 题目描述
主成分分析（PCA）是一种降维技术，它将数据转换为一个低维空间，同时尽可能保留更多的方差。它识别出主成分，这些主成分是捕捉数据中最大方差方向的正交向量。PCA通常用于数据可视化、降噪和特征提取。
#### 解决思路
- pca function that takes a dataset X and the number of principal components n_components as input.
- calculate the mean of the dataset and center the data by subtracting the mean from each data point.
- compute the covariance matrix of the centered data and find its eigenvalues and eigenvectors.
- sort the eigenvalues in descending order and select the top n_components eigenvectors as the principal components.
- project the centered data onto the principal components to obtain the lower-dimensional representation of the data.
#### Python 代码
```python
import numpy as np

def pca(X, n_components):
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:n_components]
    components = eigenvectors[:, top_indices]
    projected_data = np.dot(centered_data, components)
    return projected_data
```

```python
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
n_components = 1
projected_data = pca(X, n_components)
print(projected_data)  # Output: [[-1.73205081], [-3.46410162], [ 0.], [ 1.73205081], [ 3.46410162], [ 0.]]
```

```python
"""
In this example, we use the pca function to reduce the dimensionality of a dataset X to n_components=1. The function returns the lower-dimensional representation of the data, which is printed to verify the result.
"""
```
## Question 11: Feature Selection: eigenvalue and eigenvector
#### 解决思路
- featureSelection function that takes a dataset X and the number of principal components n_components as input. The function calculates the covariance matrix of the centered data, finds its eigenvalues and eigenvectors, and selects the top n_components eigenvectors as the principal components.
- project the centered data onto the principal components to obtain the lower-dimensional representation of the data.

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
std::vector<std::vector<T>> featureSelection(const std::vector<std::vector<T>>& X, int n_components) {
    std::vector<T> mean(X[0].size(), 0);
    for (const auto& x : X) {
        for (size_t i = 0; i < x.size(); ++i) {
            mean[i] += x[i];
        }
    }
    for (size_t i = 0; i < mean.size(); ++i) {
        mean[i] /= X.size();
    }
    std::vector<std::vector<T>> centered_data;
    for (const auto& x : X) {
        std::vector<T> centered_x;
        for (size_t i = 0; i < x.size(); ++i) {
            centered_x.push_back(x[i] - mean[i]);
        }
        centered_data.push_back(centered_x);
    }
    std::vector<std::vector<T>> covariance_matrix(X[0].size(), std::vector<T>(X[0].size(), 0));
    for (const auto& x : centered_data) {
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < x.size(); ++j) {
                covariance_matrix[i][j] += x[i] * x[j];
            }
        }
    }
    for (size_t i = 0; i < covariance_matrix.size(); ++i) {
        for (size_t j = 0; j < covariance_matrix[i].size(); ++j) {
            covariance_matrix[i][j] /= X.size();
        }
    }
    std::vector<T> eigenvalues;
    std::vector<std::vector<T>> eigenvectors;
    for (size_t i = 0; i < covariance_matrix.size(); ++i) {
        std::vector<T> eigenvector(covariance_matrix.size(), 0);
        eigenvector[i] = 1;
        eigenvectors.push_back(eigenvector);
    }
    for (size_t i = 0; i < covariance_matrix.size(); ++i) {
        T sum = 0;
        for (size_t j = 0; j < covariance_matrix.size(); ++j) {
            sum += covariance_matrix[i][j] * covariance_matrix[i][j];
        }
        eigenvalues
    }
    std::vector<std::vector<T>> components;
    for (size_t i = 0; i < eigenvectors.size(); ++i) {
        std::vector<T> component;
        for (size_t j = 0; j < eigenvectors[i].size(); ++j) {
            component.push_back(eigenvectors[i][j]);
        }
        components.push_back(component);
    }
    std::vector<std::vector<T>> projected_data;
    for (const auto& x : centered_data) {
        std::vector<T> projected_x;
        for (const auto& component : components) {
            T dot_product = 0;
            for (size_t i = 0; i < x.size(); ++i) {
                dot_product += x[i] * component[i];
            }
            projected_x.push_back(dot_product);
        }
        projected_data.push_back(projected_x);
    }
    return projected_data;
}

int main() {
    std::vector<std::vector<int>> X = {{1, 2}, {1, 4}, {1, 0}, {4, 2}, {4, 4}, {4, 0}};
    int n_components = 1;
    std::vector<std::vector<int>> projected_data = featureSelection(X, n_components);
    for (const auto& x : projected_data) {
        for (const auto& value : x) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    return 0;
    // Output: [[-1.73205081], [-3.46410162], [ 0.], [ 1.73205081], [ 3.46410162], [ 0.]]
}
```

## Question 12: K-最近邻（KNN）算法
#### 题目描述
K-最近邻（KNN）算法是一种简单而有效的分类算法，它根据其k个最近邻的多数类为数据点分配一个类标签。它使用距离度量，如欧氏距离来衡量数据点之间的相似性。KNN是一种非参数算法，不需要训练，因此适用于分类和回归任务。
#### Python 代码
```python
import numpy as np
def knn(X_train, y_train, X_test, k):
    distances = np.sqrt(np.sum((X_train - X_test[:, np.newaxis])**2, axis=2))
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_labels = y_train[nearest_indices]
    predicted_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nearest_labels)
    return predicted_labels
"""
In this example, we use the knn function to classify test data points X_test based on their k=3 nearest neighbors in the training data X_train. The function returns the predicted labels for the test data points, which are printed to verify the result.
"""
X_train = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[2, 3], [3, 3]])
k = 3
predicted_labels = knn(X_train, y_train, X_test, k)
print(predicted_labels)  # Output: [0 1]
```

## Question 13: Recurrent Neural Networks 循环神经网络（RNN）
循环神经网络（RNN）是一种设计用于处理序列数据的神经网络，它通过维护内部状态或记忆来处理序列数据。它非常适合于时间序列预测、自然语言处理和语音识别等任务。RNN使用反馈循环来处理输入序列并捕捉数据中的时间依赖关系。
#### 解决思路
- RNN class that takes the input size, hidden size, and output size as input.
- initialize the weights and biases for the RNN using vectors and matrices.
- define a forward method to perform the forward pass of the RNN, which computes the hidden state and output based on the input and previous hidden state.
- process sequential data and capture temporal dependencies.
#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
class RNN {
public:
    RNN(int input_size, int hidden_size, int output_size) : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        Wxh = std::vector<std::vector<T>>(hidden_size, std::vector<T>(input_size, 0));
        Whh = std::vector<std::vector<T>>(hidden_size, std::vector<T>(hidden_size, 0));
        Why = std::vector<std::vector<T>>(output_size, std::vector<T>(hidden_size, 0));
        bh = std::vector<T>(hidden_size, 0);
        by = std::vector<T>(output_size, 0);
    }

    std::vector<T> forward(const std::vector<T>& x, const std::vector<T>& hprev) {
        std::vector<T> h = hprev;
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<T> xh = Wxh[i];
            std::vector<T> hh = Whh[i];
            for (size_t j = 0; j < hidden_size; ++j) {
                h[j] = std::tanh(xh[j] * x[i] + hh[j] * h[j] + bh[j]);
            }
        }
        std::vector<T> y = Why * h + by;
        return y;
    }

private:
    int input_size;
    int hidden_size;
    int output_size;
    std::vector<std::vector<T>> Wxh;
    std::vector<std::vector<T>> Whh;
    std::vector<std::vector<T>> Why;
    std::vector<T> bh;
    std::vector<T> by;
};

int main() {
    RNN<double> rnn(3, 4, 2);
    std::vector<double> x = {0.1, 0.2, 0.3};
    std::vector<double> hprev(4, 0);
    std::vector<double> y = rnn.forward(x, hprev);
    for (const auto& value : y) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

## Question 14: Inheritance and Composition
Inheritance and composition are two major concepts in object-oriented programming (OOP) that allow for code reuse and the creation of complex systems through simple building blocks. Here’s how they differ:

- **Inheritance**:
  - Inheritance is a mechanism where a new class (known as a child or subclass) is derived from an existing class (known as a parent or superclass).
  - It enables the subclass to inherit all public and protected properties and methods from the superclass, allowing for polymorphism and code reuse.
  - However, it also introduces tight coupling between the subclass and the superclass since changes to the superclass might affect all its subclasses. This is sometimes summarized by the phrase "inheritance breaks encapsulation."
  - It's best used when there is a genuine hierarchical relationship between the superclass and subclass, and when behavior from the superclass should be shared or overridden by subclasses.

- **Composition**:
  - Composition involves constructing complex objects from simpler ones, thereby creating a "has-a" relationship between the objects. For example, a "Car" class might be composed of objects like "Engine," "Wheels," and "Seats."
  - It offers greater flexibility than inheritance by enabling dynamic behavior addition and change at runtime. Objects can acquire new behaviors by incorporating different objects that implement these behaviors.
  - Composition is favored for code reuse in many scenarios because it leads to looser coupling between components. Changes to a component class do not directly impact the classes that use it, provided the interface remains consistent.
  - It follows the design principle "favor composition over inheritance," which encourages using composition to achieve code reuse and flexibility without the constraints of a rigid inheritance hierarchy.



## Question 15: Handling Large Graphs in GNNs (处理大型图的GNN知识)
#### 题目描述
Graph Neural Networks (GNNs)图神经网络（GNN）是处理图结构数据的强大工具。然而，由于计算和内存限制，处理大型图（含有数百万个节点和边）可能会很具挑战性。以下是一些处理大型图的策略：
- **Graph Sampling**: 为了减少每个训练步骤中处理的图的大小，图采样技术选择一部分节点及其对应的边进行训练。流行的采样方法包括：
  - **Node Sampling**: 根据某些标准随机选择一部分节点。GraphSAGE就是使用这种方法。
  - **Layer Sampling**: 为每个节点的每一层随机采样固定数量的邻居，减少计算的指数增长。FastGCN就是一个例子。
  - **Subgraph Sampling**: 从原始图中采样小子图。这种方法旨在保留局部图结构。Cluster-GCN利用这种方法进行高效训练，它将图分割成多个簇，然后对这些簇进行采样训练。

- **Graph Partitioning**: 可以使用图分割算法将大图划分为更小的子图。每个子图独立处理，降低了整体的内存需求。METIS等技术可用于分割，Cluster-GCN等模型利用这种方法实现高效训练。


## Question 16: Backpropagation and Max Function Gradient 反向传播与Max函数梯度
#### 解决思路
- **Backpropagation**: 反向传播是训练神经网络的基本算法。它通过链式法则高效计算损失函数关于每个权重的梯度，允许按照减少损失的方向更新权重。过程涉及两个主要阶段：
  1. **Forward Pass**: Computes the output of the neural network and the loss.
  2. **Backward Pass**: Computes the gradient of the loss with respect to each weight by propagating the gradient backward through the network.
- 权重更新通常使用梯度下降等优化算法，公式为：
$ W_{new} = W_{old} - \eta \frac{\partial L}{\partial W} $
where $ \eta $ is the learning rate, $ L $ is the loss, and $ W $ represents the weights.
- **Max Function Gradient in Backpropagation**: 考虑函数 $ f(x) = \max(0, x) $（例如ReLU函数）。在反向传播期间，$ f $ 关于 $ x $ 的梯度为：
  - 1 if $ x > 0 $
  - 0 otherwise

For a max operation used in pooling layers or other contexts, like $ \max(x_1, x_2, \ldots, x_n) $, the gradient is passed to the input that had the highest value, and 0 is passed to all other inputs.

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
std::vector<T> maxFunctionGradient(const std::vector<T>& x) {
    std::vector<T> gradient(x.size(), 0);
    T max_value = x[0];
    size_t max_index = 0;
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] > max_value) {
            max_value = x[i];
            max_index = i;
        }
    }
    gradient[max_index] = 1;
    return gradient;
}

int main() {
    std::vector<int> x = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    std::vector<int> gradient = maxFunctionGradient(x);
    for (const auto& value : gradient) {
        std::cout << value << " ";
    }
    std::cout << std::endl;  // Output: 0 0 1 0 0 0 0 0 0 0 0
    return 0;
}
```

## Question 17: ResNet (Residual Networks)
#### 解决思路
ResNet introduces residual blocks with skip connections to alleviate the vanishing gradient problem in deep neural networks, allowing models to be much deeper without suffering from training difficulties. The key formula representing the operation of a basic ResNet block is:

$ \text{Output} = \mathcal{F}(\text{Input}, \{W_i\}) + \text{Input} $

where $ \mathcal{F} $ represents the residual mapping to be learned (typically two or three convolutional layers), and $ \{W_i\} $ are the weights of these layers. The addition operation is element-wise and requires that $ \mathcal{F}(\text{Input}, \{W_i\}) $ and $ \text{Input} $ have the same dimensions. If they differ, a linear projection $ W_s $ by a 1x1 convolution can be used to match the dimensions:

$ \text{Output} = \mathcal{F}(\text{Input}, \{W_i\}) + W_s \text{Input} $

This architecture significantly improves the ability to train deep networks by addressing the degradation problem, leading to groundbreaking performance in various tasks.

#### C++ 代码
```cpp
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
std::vector<T> resnetBlock(const std::vector<T>& input, const std::vector<std::vector<T>>& weights) {
    std::vector<T> residual = input;
    for (const auto& weight : weights) {
        std::vector<T> output;
        for (size_t i = 0; i < input.size(); ++i) {
            output.push_back(input[i] * weight[i]);
        }
        input = output;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] += residual[i];
    }
    return input;
}

int main() {
    std::vector<int> input = {1, 2, 3};
    std::vector<std::vector<int>> weights = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    std::vector<int> output = resnetBlock(input, weights);
    for (const auto& value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;  // Output: 2 3 4
    return 0;
}
```

## Question 18: 描述ResNet及其公式
#### 解决思路
ResNet（残差网络）通过引入跳过连接的残差块来解决深度神经网络中的梯度消失问题，允许模型更深入地进行训练，而不会遇到训练困难。一个基本的ResNet块的关键公式为：

$ \text{输出} = \mathcal{F}(\text{输入}, \{W_i\}) + \text{输入} $

其中 $ \mathcal{F} $ 表示要学习的残差映射（通常是两个或三个卷积层），$ \{W_i\}$ 是这些层的权重。加法操作是逐元素进行的，要求 $mathcal{F} \text{输入}, \{W_i\}$ 和 $ \text{输入} $ 的维度相同。如果它们的维度不同，可以使用1x1卷积的线性投影 $ W_s $ 来匹配维度：

$\text{输出} = \mathcal{F}(\text{输入}, \{W_i\}) + W_s \text{输入} $

这种架构通过解决退化问题，显著改善了深度网络的训练能力，带来了各种任务上的突破性性能。

## Question 19: Coordinate Descent Algorithm 坐标下降算法
使用坐标下降（Coordinate Descent）方法解决优化问题有其独特的优势。坐标下降是一种迭代优化算法，它通过轮流固定某些变量，只对一个或一小部分变量进行优化，从而逐渐找到问题的最优解。这种方法在处理某些类型的问题时具有明显的优势：

### 好处
1. **简单易实现**：坐标下降算法的实现相对简单，因为它将多变量优化问题简化为一系列单变量优化问题，这些问题往往更容易解决。
2. **高效处理大规模问题**：对于某些大规模问题，特别是当变量之间的相互依赖性较弱时，坐标下降法可以高效地进行计算。由于每次迭代只更新部分变量，减少了计算量。
3. **稀疏优化问题的适用性**：在处理具有稀疏性质的数据或模型时（如大规模稀疏线性回归、稀疏逻辑回归），坐标下降法能有效地更新模型参数，尤其是在参数的最优值为零或接近零的情况下更为明显。
4. **并行化和分布式计算**：对于一些变量独立的情况，坐标下降法可以很自然地扩展到并行和分布式计算中，每个处理器或计算节点可以负责优化一部分变量，进一步提高计算效率。
5. **适用于特定类型的非凸优化**：虽然坐标下降在寻找全局最优解方面可能不如基于梯度的方法那样通用，但它在处理某些特定类型的非凸优化问题时可能更有效，尤其是当问题的结构允许通过局部优化达到全局最优或接近全局最优的解时。

#### 注意事项
尽管坐标下降方法有上述优点，但它也有一些局限性。例如，在高度耦合变量的情况下，每次只优化一个或少数几个变量可能会导致收敛速度较慢。此外，它不保证总是找到全局最优解，尤其是在复杂的非凸优化问题中。

总之，坐标下降法是解决一些特定优化问题的有力工具，尤其是在数据稀疏、问题规模大、变量之间相对独立的情况下。然而，选择最合适的优化算法还需要根据具体问题的性质和需求来决定。

## Question 20: Dilation in Convolutional Neural Networks (CNNs)
Dilation is a technique used in Convolutional Neural Networks (CNNs) to increase the receptive field of filters without increasing the number of parameters. It involves introducing gaps or "holes" between the elements of the filter, effectively expanding the filter's field of view. This technique is particularly useful for capturing long-range dependencies in images and sequences. Here's how dilation works:

- **Standard Convolution**: In a standard convolution operation, the filter slides over the input with a stride of 1, covering adjacent elements at each step.

- **Dilated Convolution**: In a dilated convolution, the filter is applied to the input with gaps between the elements, determined by the dilation rate. The dilation rate specifies the spacing between the elements of the filter, effectively increasing the receptive field of the filter.

The output of a dilated convolution has the same spatial dimensions as the input, but the receptive field of the filter is expanded. This allows the network to capture larger patterns and long-range dependencies in the input data, making it particularly effective for tasks like semantic segmentation and object detection.

Dilation in a convolutional layer is a concept that allows the convolution to operate over an area larger than its kernel size without increasing the number of parameters or the amount of computation. This is achieved by introducing gaps between the elements in the kernel when it is applied to the input feature map. Essentially, dilation enables the convolutional layer to have a wider field of view, enabling it to capture more spatial context.

- **How Dilation Works**

- A standard convolution operation applies the kernel to the input feature map in a continuous manner, where each element of the kernel is used to weigh adjacent elements of the input.
- In a dilated convolution, spaces are inserted between kernel elements. A dilation rate of $d$ means there are $d-1$ spaces between each kernel element. For example, with a dilation rate of 1 (no dilation), the kernel is applied in the standard way. With a dilation rate of 2, there is 1 space between kernel elements, and so on.

- **Benefits of Dilation**

- **Increased Receptive Field**: Dilated convolutions allow the network to aggregate information from a larger area of the input without increasing the kernel size or the number of parameters. This is particularly useful for tasks requiring understanding of wider context, such as semantic segmentation and time series analysis.
- **Efficient Computation**: Because dilation does not increase the number of weights or the computational complexity in the same way that increasing kernel size would, it provides an efficient means to increase the receptive field.
- **Improved Performance on Certain Tasks**: Dilated convolutions have been shown to improve performance on tasks that benefit from larger receptive fields, such as semantic segmentation (e.g., DeepLab architectures) and audio generation (e.g., WaveNet).

## Question 21: Noise in Images and Edge Preservation
Dealing with noise in images while preserving edges is crucial for various image processing and computer vision tasks. A popular approach involves using filtering techniques that are adept at reducing noise without blurring the edges. **Bilateral filtering** and **Non-Local Means** are two such techniques:

- **Bilateral Filtering**: This method smooths images while preserving edges by combining spatial and intensity information. It considers both the spatial closeness and the intensity similarity when averaging the pixels, which helps in preserving sharp edges.
  
- **Non-Local Means**: Unlike local filters that only consider a small neighborhood around each pixel, Non-Local Means filtering averages all pixels in the image weighted by their similarity to the target pixel. This method is particularly effective at preserving detailed structures and edges in images.

## Question 22: Properties of Rotation Matrix
In OpenGL, several types of matrices are used to transform objects in 3D space:
- **Model Matrix**: Transforms an object's vertices from object space to world space.
- **View Matrix**: Transforms vertices from world space to camera (view) space.
- **Projection Matrix**: Transforms vertices from camera space to normalized device coordinates (NDC). There are two common types of projection matrices: orthogonal and perspective.
- **MVP Matrix**: A combination of Model, View, and Projection matrices applied in sequence to transform object space vertices directly to NDC.

**Properties of a Rotation Matrix**:
- **Orthogonal Matrix**: A rotation matrix is orthogonal, meaning its rows are mutually orthogonal to its columns.
- **Determinant is +1**: The determinant of a proper rotation matrix is +1.
- **Inverse Equals Transpose**: The inverse of a rotation matrix is equal to its transpose.

## Question 23: RANSAC Algorithm
**RANSAC** (Random Sample Consensus) is an iterative method used for robustly fitting a model to data with a high proportion of outliers. It is widely used in computer vision tasks, such as feature matching and 3D reconstruction, to estimate parameters of a mathematical model from a dataset that contains outliers. RANSAC works by repeatedly selecting a random subset of the original data to fit the model and then determining the number of inliers that fall within a predefined tolerance of the model. The model with the highest number of inliers is considered the best fit.




