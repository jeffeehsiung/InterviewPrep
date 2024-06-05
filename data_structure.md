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