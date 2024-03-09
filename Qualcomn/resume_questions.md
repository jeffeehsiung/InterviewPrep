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

9. Share your experience with software development, testing, and debugging on XR devices, mobile platforms, or other embedded systems.

10. Discuss your proficiency in Python and/or C++ programming, and how you have utilized these languages in the deployment of deep neural networks and model optimization on edge devices.

12. Explain your experience using machine learning toolboxes such as PyTorch or TensorFlow, especially in the context of XR perception tasks.

13. Reflect on a situation where you faced challenges in deploying deep neural networks on edge devices and how you overcame them.

16. Can you explain your role and contributions in the SRIR Dataset and Partial Optimal Transport Interpolation for Dynamic Binaural Auralization project? How did you synthesize the SRIR model and achieve an interaural cross-correlation of 52%?

16. As a Master Thesis Student in Deep Learning 3-D Multistatic ISAR for Person Identification, what was your research objective and how did you develop the M-InISAR pipeline for non-cooperative target 3D reconstruction? Can you explain the role of PointNet++ and LSTM models in person identification?

17. Tell us more about your experience developing the Pixel War Game. How did you design the architecture for scalability and implement features like multiplayer, auto-play, A* pathfinding, and animation?

18. In your Bachelor Thesis on the Baby Monitoring and Cry-Detection System, how did you prototype the cry detection model using an ESP32 microcontroller? Can you explain the signal processing techniques you used and the accuracy achieved?

19. Can you elaborate on your work in Pet Facial Expression Recognition for Enhanced XR Human-Pet Interactions? How did you design and compare the fully-connected baseline model and the advanced CNN model? What were the key findings and performance metrics?


21. Can you describe a challenging situation you faced during your work or studies and how you overcame it?

26. Can you discuss any leadership roles or experiences you have had? How did you motivate and guide your team towards achieving a common goal?

27. Can you tell us about any specific technical skills or tools you have expertise in that would be relevant to the role you are applying for?