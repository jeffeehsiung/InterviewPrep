// binary tree functions and main program

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <cmath>
#include <climits>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <utility>
#include <functional>
#include <iterator>
#include <list>
#include <deque>
#include <bitset>
#include <array>
#include <forward_list>
#include <random>
#include <chrono>
#include <numeric>
#include <memory>


// binary tree node
template <typename T>
struct TreeNode {
    T val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(T x) : val(x), left(nullptr), right(nullptr) {}
};

// binary tree class
template <typename T>
class BinaryTree {
public:
    // constructor
    BinaryTree() : root(nullptr) {}

    // destructor
    ~BinaryTree() {
        destroyTree(root);
    }

    // destroy tree
    void destroyTree(TreeNode<T>*& root) {
        if (root) {
            destroyTree(root->left);
            destroyTree(root->right);
            delete root;
            root = nullptr;
        }
    }

    // insert node
    void insertNode(T val) {
        if (!root) {
            root = new TreeNode<T>(val);
            return;
        }
        TreeNode<T>* curr = root;
        while (curr) {
            if (val < curr->val) {
                if (curr->left) {
                    curr = curr->left;
                }
                else {
                    curr->left = new TreeNode<T>(val);
                    return;
                }
            }
            else if (val > curr->val) {
                if (curr->right) {
                    curr = curr->right;
                }
                else {
                    curr->right = new TreeNode<T>(val);
                    return;
                }
            }
            else {
                return;
            }
        }
    }

    // delete node
    void deleteNode(T val) {
        root = deleteNode(root, val);
    }

    // delete node
    TreeNode<T>* deleteNode(TreeNode<T>* root, T val) {
        if (!root) {
            return nullptr;
        }
        if (val < root->val) {
            root->left = deleteNode(root->left, val);
        }
        else if (val > root->val) {
            root->right = deleteNode(root->right, val);
        }
        else {
            if (!root->left) {
                TreeNode<T>* temp = root->right;
                delete root;
                return temp;
            }
            else if (!root->right) {
                TreeNode<T>* temp = root->left;
                delete root;
                return temp;
            }
            TreeNode<T>* temp = findMin(root->right);
            root->val = temp->val;
            root->right = deleteNode(root->right, temp->val);
        }
        return root;
    }

    // find min node
    TreeNode<T>* findMin(TreeNode<T>* root) {
        while (root->left) {
            root = root->left;
        }
        return root;
    }

    // find max node
    TreeNode<T>* findMax(TreeNode<T>* root) {
        while (root->right) {
            root = root->right;
        }
        return root;
    }

    // search node
    bool searchNode(T val) {
        TreeNode<T>* curr = root;
        while (curr) {
            if (val < curr->val) {
                curr = curr->left;
            }
            else if (val > curr->val) {
                curr = curr->right;
            }
            else {
                return true;
            }
        }
        return false;
    }

    // preorder traversal
    void preorderTraversal() {
        preorderTraversal(root);
    }

    // preorder traversal
    void preorderTraversal(TreeNode<T>* root) {
        if (root) {
            std::cout << root->val << " ";
            preorderTraversal(root->left);
            preorderTraversal(root->right);
        }
    }

    // inorder traversal
    void inorderTraversal() {
        inorderTraversal(root);
    }

    // inorder traversal
    void inorderTraversal(TreeNode<T>* root) {
        if (root) {
            inorderTraversal(root->left);
            std::cout << root->val << " ";
            inorderTraversal(root->right);
        }
    }

    // postorder traversal
    void postorderTraversal() {
        postorderTraversal(root);
    }

    // postorder traversal
    void postorderTraversal(TreeNode<T>* root) {
        if (root) {
            postorderTraversal(root->left);
            postorderTraversal(root->right);
            std::cout << root->val << " ";
        }
    }

    // posterior traversal
    void posteriorTraversal() {
        posteriorTraversal(root);
    }

    // posterior traversal
    void posteriorTraversal(TreeNode<T>* root) {
        if (root) {
            std::stack<TreeNode<T>*> stk;
            TreeNode<T>* curr = root;
            // go to leftmost node of tree and push all nodes along the way to stack
            while (curr || !stk.empty()) {
                if (curr) {
                    stk.push(curr);
                    curr = curr->left;
                }
                else {
                    // if no left child, then visit right child
                    // The stk.top() function in C++ returns the most recently added (topmost) element of the stack.
                    TreeNode<T>* temp = stk.top()->right;
                    if (!temp) {
                        temp = stk.top();
                        stk.pop();
                        std::cout << temp->val << " ";
                        while (!stk.empty() && temp == stk.top()->right) {
                            temp = stk.top();
                            stk.pop();
                            std::cout << temp->val << " ";
                        }
                    }
                    else {
                        curr = temp;
                    }
                }
            }
        }
    }


    // level order traversal (Breadth-First Search, BFS)
    void levelOrderTraversal() {
        levelOrderTraversal(root);
    }

    // level order traversal
    void levelOrderTraversal(TreeNode<T>* root) {
        if (!root) {
            return;
        }
        std::queue<TreeNode<T>*> que;
        que.push(root);
        while (!que.empty()) {
            TreeNode<T>* curr = que.front();
            que.pop();
            std::cout << curr->val << " ";
            if (curr->left) {
                que.push(curr->left);
            }
            if (curr->right) {
                que.push(curr->right);
            }
        }
    }

    // height of tree
    int height() {
        return height(root);
    }

    // height of tree
    int height(TreeNode<T>* root) {
        if (!root) {
            return 0;
        }
        int leftHeight = height(root->left);
        int rightHeight = height(root->right);
        return std::max(leftHeight, rightHeight) + 1;
    }

    // diameter of tree
    int diameter() {
        return diameter(root);
    }

    // diameter of tree
    int diameter(TreeNode<T>* root) {
        if (!root) {
            return 0;
        }
        int leftHeight = height(root->left);
        int rightHeight = height(root->right);
        int leftDiameter = diameter(root->left);
        int rightDiameter = diameter(root->right);
        return std::max(leftHeight + rightHeight, std::max(leftDiameter, rightDiameter));
    }

    // root
    TreeNode<T>* root;

    // print tree
    void printTree() {
        std::cout << "Preorder Traversal: ";
        preorderTraversal();
        std::cout << std::endl;
        std::cout << "Inorder Traversal: ";
        inorderTraversal();
        std::cout << std::endl;
        std::cout << "Postorder Traversal: ";
        postorderTraversal();
        std::cout << std::endl;
        std::cout << "Level Order Traversal: ";
        levelOrderTraversal();
        std::cout << std::endl;
        std::cout << "Height of Tree: " << height() << std::endl;
        std::cout << "Diameter of Tree: " << diameter() << std::endl;
        std::cout << "Posterior Traversal: ";
        posteriorTraversal();
    }

};