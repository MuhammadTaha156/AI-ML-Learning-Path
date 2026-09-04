# Deep Learning (Part 4): Neural Network Architectures & Introduction to CNNs

## 1. Overview of Neural Network Architectures
Building upon Artificial Neural Networks (ANNs), different architectures are specialized for different types of data:
*   **FNN (Feedforward Neural Network):** Used primarily for tabular data (e.g., CSV, Excel files) for regression and classification tasks. Data flows strictly in one direction (input to output).
*   **CNN (Convolutional Neural Network):** Specialized for **Computer Vision** (extracting meaningful information from images, videos, and visual data).
*   **RNN (Recurrent Neural Network):** Specialized for **NLP (Natural Language Processing)** and sequential data.
*   **Transformers:** Advanced architectures driving modern Large Language Models (LLMs).
*   **GANs:** Generative Adversarial Networks used for generating synthetic data and media.

---

## 2. Introduction to Computer Vision & CNNs
Computer Vision involves extracting meaningful patterns and information from visual data. While FNNs can process images by flattening them into 1D vectors, they fail miserably with high-dimensional images because:
1.  **Massive Feature Spaces:** A Full HD image ($1920 \times 1080 \times 3$ channels) creates over **6.2 million features (pixels)**, leading to astronomical computational complexity and resource requirements for an FNN.
2.  **Loss of Spatial Structure:** Flattening an image destroys the spatial relationship between neighboring pixels (e.g., separating an object's subject/foreground from its background).

---

## 3. Image Representation in Code
*   **Grayscale Images:** Represented as a 2D matrix (height $\times$ width) where pixel values range from **0 (black) to 255 (white)**. In tensor shape notation, this is $(H \times W \times 1)$ where 1 represents the single channel.
*   **Colored Images (RGB):** Represented as a 3D tensor consisting of 3 color channels: **Red, Green, and Blue (RGB)**. Each channel is a 2D matrix ranging from 0 to 255. Tensor shape: $(H \times W \times 3)$.

---

## 4. Convolutional Layer Mechanics
The core building block of a CNN is the **Convolutional Layer**, which acts as a feature detector.

### Preprocessing Step:
1.  **Min-Max Scaling:** Pixel values are typically normalized from $[0, 255]$ to a range of $[0, 1]$ to stabilize training.

### Core Concepts:
1.  **Filter (Kernel):** A small matrix (e.g., $3 \times 3$ or $5 \times 5$) containing learnable weights ($w, b, k$). 
2.  **Convolution Operation:** The filter slides across the input image matrix, performs element-wise multiplication, and sums the results to produce a feature map (output matrix).
    *   *Example:* A $3 \times 3$ filter has 9 values, meaning it acts as 9 learnable weights.
3.  **Feature Extraction:** Different filters detect different patterns in an image, such as **vertical edges, horizontal edges, textures, corners, and sharpness**.
4.  **Stride:** The number of pixels by which the filter matrix shifts over the input matrix (e.g., Stride = 1 or Stride = 2).

---

## 5. CNN Architecture Pipeline
A standard CNN architecture consists of three main types of layers stacked sequentially before feeding into an ANN:
1.  **Convolutional Layer:** Extracts spatial features using filters.
2.  **Pooling Layer:** Downsamples the feature map to reduce dimensionality and computational load.
3.  **Fully Connected (FC) Layer:** A traditional ANN layer that takes the extracted features and performs the final classification or regression.

# Deep Learning (Part 4 - Continued): CNN Mechanics, Pooling, and Fully Connected Layers

## 1. Convolutional Layer Mechanics (Recap & Formulas)
A Convolutional Layer applies a filter (kernel) across an input image to generate a feature map. 

### Output Dimension Formula:
$$\text{Output Size} = \frac{n - f + 2p}{s} + 1$$
*   **$n$**: Input image dimension ($n \times n$)
*   **$f$**: Filter/Kernel size ($f \times f$)
*   **$p$**: Padding size (e.g., $p = 0$)
*   **$s$**: Stride size (e.g., $s = 1$)

### Padding Options:
*   **Valid Padding:** No padding ($p = 0$), resulting in a smaller output matrix.
*   **Same Padding:** Padding added so that the output size matches the input size.

---

## 2. Pooling Layer (Downsampling)
The pooling layer reduces the spatial dimensions (width and height) of feature maps while retaining strong and important features, thereby lowering computational complexity.

### Typical Pipeline Structure:
$$\text{Image} \rightarrow (\text{Conv} + \text{ReLU}) \rightarrow \text{Pooling} \rightarrow (\text{Conv} + \text{ReLU} + \text{Pooling}) \dots \rightarrow \text{Fully Connected Layer (FCL)}$$

### Types of Pooling:
1.  **Max Pooling:** Extracts the maximum value from a given window (e.g., taking the strongest feature from a $2 \times 2$ block with stride 2).
2.  **Min Pooling:** Extracts the minimum value.
3.  **Average Pooling:** Calculates the average value of the window.

---

## 3. Fully Connected Layer (FCL) & Classification Pipeline
After feature extraction through multiple Convolutional and Pooling layers, the data is prepared for final classification.

1.  **Flattening:** The multi-dimensional feature maps are flattened into a 1D vector.
2.  **Dense ANN:** The 1D vector is fed into a dense Artificial Neural Network (fully connected layers).
3.  **Output Layer:** 
    *   Generates the final predictions for classes (e.g., dog, cat, horse).
    *   A **Softmax Activation Function** is utilized at the output layer to convert raw scores into probability distributions.