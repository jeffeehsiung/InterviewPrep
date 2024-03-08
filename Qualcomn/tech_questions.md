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


