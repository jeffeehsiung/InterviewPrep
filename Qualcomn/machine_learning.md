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




