# Deep Learning (Part 2): Training a Neural Network

## 1. Loss Functions
A Loss Function calculates the error between the model's predictions and the actual target values. The goal of training is to minimize this loss.

*   **Loss:** Error for a *single* sample (e.g., $\frac{1}{2}(y - \hat{y})^2$).
*   **Cost:** Average error across the *entire dataset* (e.g., $\frac{1}{2N} \sum (y_i - \hat{y}_i)^2$).

### A. Loss Functions for Regression
*(Output layer activation: Linear)*

1.  **Mean Squared Error (MSE):**
    $$J(\theta) = \frac{1}{2N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
    *   *Pros:* Easily differentiable, has 1 global minima (convex shape).
    *   *Cons:* Highly sensitive to outliers.
2.  **Mean Absolute Error (MAE):**
    $$J(\theta) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
    *   *Pros:* Less sensitive to outliers.
    *   *Cons:* Not a smooth gradient (V-shaped graph), making optimization harder.
3.  **Huber Loss:**
    Combines the best of MSE and MAE using a hyperparameter **delta ($\delta$)**.
    *   Acts quadratically (like MSE) for small errors ($err \le \delta$).
    *   Acts linearly (like MAE) for large errors ($err > \delta$).
    $$
    J(\theta) = 
    \begin{cases} 
      \frac{1}{2N} \sum (y_i - \hat{y}_i)^2 & \text{if } error \le \delta \\
      \frac{\delta}{N} \sum (|y_i - \hat{y}_i| - \frac{\delta}{2}) & \text{if } error > \delta
    \end{cases}
    $$

### B. Loss Functions for Classification
1.  **Binary Classification:**
    *   *Output layer activation:* Sigmoid.
    *   *Loss Function:* **Binary Cross Entropy (Log Loss)**.
    $$J(\theta) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)]$$
2.  **Multi-Class Classification:**
    *   *Target preparation:* Must be One-Hot Encoded.
    *   *Output layer activation:* Softmax (converts raw values into probabilities that sum to 1).
    $$f(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
    *   *Loss Function:* **Categorical Cross Entropy**.
    $$J(\theta) = - \sum_{i=1}^{c} y_i \log(\hat{y}_i)$$
    *(Where $c$ = total classes).*

---

## 2. Weight Updation (Gradient Descent)
Once the loss is calculated, the network must update its weights to reduce the error. The general formula for updating a weight ($w$) or bias ($b$) is:

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w_{old}}$$
$$b_{new} = b_{old} - \eta \cdot \frac{\partial L}{\partial b_{old}}$$

*   **$\eta$ (eta):** Learning Rate.
*   **$\frac{\partial L}{\partial w_{old}}$:** The gradient (slope of the loss curve).

**How it works (The Gradient Descent Curve):**
*   **Case 1 (Positive Slope):** If the current weight is on the right side of the curve, the slope is positive ($+ve$).
    *   $w_{new} = w_{old} - \eta(+ve) \implies w_{new} < w_{old}$ (Shifts left towards the minima).
*   **Case 2 (Negative Slope):** If the current weight is on the left side of the curve, the slope is negative ($-ve$).
    *   $w_{new} = w_{old} - \eta(-ve) \implies w_{new} > w_{old}$ (Shifts right towards the minima).

---

## 3. Backpropagation & The Chain Rule
To calculate the gradient $\left(\frac{\partial L}{\partial w}\right)$ for weights hidden deep within the network, we must use the **Chain Rule of Derivatives**.

If $y = f(g(x))$, let $g(x) = u$, so $y = f(u)$. The Chain Rule states:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Application in a Neural Network:**
Imagine a sequence: $x_1 \xrightarrow{w_1} \text{Neuron 1} \xrightarrow{w_2} \text{Neuron 2} \rightarrow out_2 (\hat{y})$.
We calculate the Loss ($L$) using $out_2$. To update weight $w_2$, we need the partial derivative of the Loss with respect to $w_{2\_old}$:

$$\frac{\partial L}{\partial w_{2\_old}} = \frac{\partial L}{\partial out_2} \cdot \frac{\partial out_2}{\partial w_{2\_old}}$$

Backpropagation calculates these partial derivatives layer by layer, starting from the output and moving backward, multiplying the gradients using the chain rule to update every weight in the network.

#  Optimizers & Advanced Concepts

## 1. Optimizers
Optimizers are algorithms or methods used to change the attributes of a neural network (such as weights and learning rate) to reduce the losses. They determine *how* the network updates its weights using the gradients calculated during backpropagation.

**Weight Update Formula:**
$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w_{old}}$$

### Common Optimization Algorithms:
1.  **Batch Gradient Descent:** Uses the entire training dataset to calculate the gradient and update weights.
2.  **Stochastic Gradient Descent (SGD):** Updates weights after evaluating a *single* training sample.
3.  **Mini-Batch Gradient Descent:** Divides the dataset into small batches and updates weights after each batch. (The most common approach).
4.  **Gradient Descent with Momentum**
5.  **RMSprop**
6.  **Adam (Adaptive Moment Estimation):** Currently the most popular and widely used optimizer as it combines the best properties of Momentum and RMSprop.

### Epoch vs. Batch vs. Iteration
*   **Batch:** A subset of the training data. For example, if you have 1000 samples and a batch size of 100, you have 10 batches.
    *   *Why use batches?* Computing gradients for the entire dataset at once is slow and requires massive amounts of RAM/CPU.
*   **Iteration:** One complete forward pass and backward pass (1 FP + 1 BP) on *one batch* of data. In the example above, 1 epoch would require 10 iterations.
*   **Epoch:** One epoch is completed when the neural network has seen the *entire* training dataset exactly once.

---

## 2. The Vanishing Gradient Problem (VGP)
The Vanishing Gradient Problem occurs during backpropagation in deep neural networks. As the error signal travels backward through the layers, the gradients ($rac{\partial L}{\partial w}$) become exponentially smaller (they "vanish"). 

**Why is this a problem?**
If the gradient becomes a tiny value (close to 0), the weight update step becomes insignificant:
$$w_{new} = w_{old} - \eta \cdot (tiny\_val) \implies w_{new}  pprox w_{old}$$
If $w_{new}  pprox w_{old}$, the early layers of the network **stop learning**, preventing the model from converging.

**What causes VGP?**
It is primarily caused by certain Activation Functions, specifically the **Sigmoid** and **Tanh** functions.
*   **Sigmoid Derivative:** The maximum value of the derivative of the Sigmoid function is only **0.25**. When applying the Chain Rule during backpropagation, you are multiplying many numbers between 0 and 0.25 together. Multiplying small fractions repeatedly results in an even smaller fraction, causing the gradient to vanish.
*   **Tanh Derivative:** The derivative ranges from $(0, 1]$, which is better than Sigmoid, but still suffers from VGP in very deep networks.

---

## 3. Solving VGP: ReLU & Its Variants
To solve the Vanishing Gradient Problem, modern Deep Learning networks use **ReLU (Rectified Linear Unit)** and its variants for hidden layers.

### ReLU (Rectified Linear Unit)
$$f(x) = \max(0, x)$$
*   **Derivative:** The derivative of ReLU is $1$ for $x \ge 0$, and $0$ for $x < 0$. Because the derivative is exactly 1 (not a fraction), multiplying gradients during the chain rule does not cause them to vanish.
*   **Pros:** Simple, fast, popular, solves VGP.
*   **Cons:** "Dying ReLU" problem. If $x < 0$, the gradient is exactly 0, meaning the neuron "dies" and stops learning entirely.

### Variants to solve "Dying ReLU":
1.  **Leaky ReLU:** Instead of outputting 0 for negative inputs, it outputs a very small, fixed value (e.g., $0.01x$).
    $$f(x) = \max(x, 0.01x)$$
2.  **Parametric ReLU (PReLU):** Similar to Leaky ReLU, but the multiplier ($ lpha$) for negative inputs is not fixed; it is treated as a parameter that the network learns during training.
    $$f(x) = \max(x, \alpha x)$$
3.  **Exponential Linear Unit (ELU):** Uses a smooth, exponential curve for negative values.
    $$
    f(x) = 
    \begin{cases} 
      x & \text{if } x \ge 0 \\
      \alpha(e^x - 1) & \text{if } x < 0
    \end{cases}
    $$

# Advanced Optimizers

## 4. Advanced Optimizers (Beyond Basic Gradient Descent)
Standard Gradient Descent can sometimes oscillate or get stuck in ravines/flat regions during optimization. To solve this, advanced optimizers incorporate history (momentum) or adapt the learning rate per parameter.

### A. Gradient Descent with Momentum
Momentum helps accelerate gradient descent in the relevant direction and dampens oscillations by adding a fraction ($ \beta $) of the past gradients to the current update.

*   **Velocity Term ($v_t$):**
    $$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \frac{\partial L}{\partial w_{t-1}}$$
    *(Where $\beta$ is the momentum hyperparameter, typically set to values like $0.9$).*
*   **Weight Update:**
    $$w_t = w_{t-1} - \eta \cdot v_t$$
    $$b_t = b_{t-1} - \eta \cdot v_t$$

### B. RMSprop (Root Mean Square Propagation)
RMSprop divides the learning rate by an exponentially decaying average of squared gradients. This means different parameters get different adaptive learning rates, helping the model move quickly through flat regions and take smaller steps on steep slopes.

*   **Moving Average of Squared Gradients ($s_t$):**
    $$s_t = \beta \cdot s_{t-1} + (1 - \beta) \left( \frac{\partial L}{\partial w_{t-1}} \right)^2$$
*   **Effective Learning Rate ($\eta'$):**
    $$\eta' = \frac{\eta}{\sqrt{s_t + \epsilon}}$$
    *(Where $\epsilon$ is a tiny value to prevent division by zero).*
*   **Weight Update:**
    $$w_t = w_{t-1} - \eta' \cdot \frac{\partial L}{\partial w_{t-1}}$$

### C. Adam (Adaptive Moment Estimation)
Adam is the **most commonly used optimizer** in modern deep learning [cite: file-tag: b25118c9-342f-409d-a00c-c616719c674d]. It combines the best properties of both **Gradient Descent with Momentum** (storing past velocities $v_t$) and **RMSprop** (storing past squared gradients $s_t$).

*   **Combination:** Momentum ($v_t$) + RMSprop ($s_t$)
*   **Weight Update:** Uses adaptive learning rates driven by RMSprop alongside momentum-adjusted update steps, providing fast and stable convergence.