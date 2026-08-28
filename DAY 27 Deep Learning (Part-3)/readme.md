# Deep Learning (Part 3): Building & Training ANNs with PyTorch


## Project 1: ANN for Regression (Power Plant Dataset)
### Objective
#### Predict the Energy Output (PE) of a power plant based on four input features:

AT: Temperature

V: Vacuum

AP: Pressure

RH: Humidity

Pipeline
Load the data

Data → Tensors (Convert to PyTorch tensors)

TensorDataset & DataLoader (Batch processing)

Define ANN model

Train the model (and Save the best model)

### Evaluate

Key Concepts (From Notes)
Tensor Reshaping: Target variable (y_train) is reshaped to (n, 1) using .view(-1, 1) so that it matches the output shape of the network.

Tensor Dimensions: 1D (scalar), 2D (matrix/vector), 3D, and 4D+ (tensors).

### Architecture:

Input Layer (4 features)

2 Hidden Layers (6 neurons each)

Output Layer (1 neuron)

Activation Function: ReLU

### Batching: 
Mini-batch Gradient Descent is used. TensorDataset aligns features and targets, while DataLoader automatically creates batches (e.g., batch size = 32) and shuffles the data.

### Training Loop:

Compute forward pass (output = model(xb))

Calculate loss (e.g., nn.MSELoss)

Backpropagation (loss.backward())

Parameter updates (optimizer.step())

Average loss per epoch = running_loss / number of batches

Saving the Model: The model weights are saved (torch.save) when validation loss reaches a minimum to ensure you keep the best model.

Evaluation: Training MSE, Testing MSE, and R² Score (r2_score).


# Model Definition
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6),  # Input -> Hidden
            nn.ReLU(),
            nn.Linear(6, 6),  # Hidden -> Hidden
            nn.ReLU(),
            nn.Linear(6, 1)   # Hidden -> Output
        )
    def forward(self, x):
        return self.model(x)

# Training the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# (Training loop logic...)
# Save best model using: torch.save(model.state_dict(), "best_model.pt")
Project 2: ANN for Classification (Date Fruit Dataset)
Objective
Classify 7 different types of date fruits (BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY) based on 34 extracted shape and color features.

Key Concepts (From Notes)
Architecture:

Input Layer (34 features)

1-2 Hidden Layers (64 neurons each as per code, notes mention 32 neurons for tackling underfitting)

Output Layer (7 neurons)

Activation Function: ReLU

Logits vs. Probabilities:

The output layer produces raw values known as logits.

By default, nn.CrossEntropyLoss applies a Softmax function internally to convert logits to probabilities.

The inputs (targets) must be long integers.

Prediction: To get the predicted class, use torch.max(outputs, 1) which returns the index of the highest value. This index corresponds to the category (e.g., index 2 = the third fruit type).

Code Snippet (ANN_Classification.ipynb)
python
# Label Encoding the target
le = LabelEncoder()
y = le.fit_transform(y)

# Model Definition (Output layer has 7 neurons)
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(34, 64),  # Input -> Hidden
            nn.ReLU(),
            nn.Linear(64, 64),  # Hidden -> Hidden
            nn.ReLU(),
            nn.Linear(64, 7)    # Hidden -> Output (7 classes)
        )
    def forward(self, x):
        return self.model(x)

# Loss Function and Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Evaluation Logic (Calculating Accuracy)
_, predicted = torch.max(outputs, 1)
correct += (predicted == yb).sum().item()
Summary of Differences
Feature	Regression	Classification
Dataset	Power Plant	Date Fruit
Output Shape	1 Neuron (Continuous Value)	7 Neurons (Class Probabilities)
Loss Function	nn.MSELoss	nn.CrossEntropyLoss
Target Format	Float Tensor (n, 1)	Long Tensor (Integer Encoded)
Evaluation	MSE, R² Score	Accuracy Percentage
## 1. End-to-End Workflow for ANN (Regression Example)
When building an Artificial Neural Network (ANN) for a regression task (e.g., predicting energy output using the PowerPlant dataset), the typical workflow follows these steps:
1. Load the data.
2. Convert data into PyTorch **Tensors**.
3. Create **TensorDataset** and **DataLoader** objects.
4. Define the ANN model architecture.
5. Train the model (and save the best model weights).
6. Evaluate the model.

---

## 2. Data Preparation & PyTorch Core Data Structures (Tensors)
PyTorch uses **Tensors** as its core data structure, similar to NumPy arrays but optimized for GPU acceleration.

### Tensor Dimensions:
*   **1D Tensor:** Scalar-like / vector elements
*   **2D Tensor:** Matrix or vector (Rows $\times$ Columns, e.g., created using `.view(-1, 1)` or specifying shapes like $(n, 1)$ for target variables like `y_train`).
*   **3D / 4D+ Tensors:** Used for complex data like images and batches.

### Dataset & DataLoader
To efficiently feed data into a neural network using mini-batch gradient descent (e.g., batch size = 32 samples out of 1000 total samples):
*   **`TensorDataset` class:** Accesses data sample-by-sample, binding features and targets together into rows `(features, target)`.
*   **`DataLoader` class:** Defines how data will be loaded for training. It handles creating batches and can optionally shuffle the data.

---

## 3. Defining the ANN Model Architecture (PyTorch)
An ANN model is defined using a Python class that inherits from `nn.Module`.

### Example Architecture for Regression:
*   **Input Layer:** Matches the number of features in the dataset (e.g., 4 input features like Temperature, Vacuum, Pressure, Humidity).
*   **Hidden Layers:** E.g., 2 hidden layers with 6 neurons each. The hidden layers use non-linear activation functions like **ReLU**.
*   **Output Layer:** For regression tasks, the output layer typically has **1 neuron** with a **Linear** activation function to predict a continuous value ($\hat{y}$).

### Code Structure Concept:
```python
import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.linear1 = nn.Linear(4, 6) # Input to Hidden 1
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(6, 6) # Hidden 1 to Hidden 2
        self.output = nn.Linear(6, 1)  # Hidden 2 to Output
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.output(x)
        return x
```
*   **`Autograd`:** PyTorch's automatic differentiation engine that computes gradients (`loss.backward()`) during backpropagation automatically.

---

## 4. Training the ANN
Training involves passing data through multiple epochs (e.g., 100 epochs) using mini-batches.

*   **Epoch:** One complete pass of the neural network over the entire training dataset.
*   **Batch vs. Running Loss:**
    *   `loss`: Error calculated for a single batch.
    *   `running loss`: Cumulative loss across all batches in an epoch.
    *   `Avg Loss per batch for 1 epoch` = $\frac{\text{running loss}}{\text{number of batches}}$

---

## 5. Saving and Loading the Best Model
During training, we track validation loss across epochs. 
*   The **Best Model** is defined as the model state that achieves the **minimal validation loss**.
*   Instead of saving the entire model architecture (which increases file size), we typically save only the **learnable parameters (weights $w$ and biases $b$)** as a state dictionary (`dict`) to disk.

---

## 6. Evaluation Metrics for Regression
Once trained and loaded, the model is evaluated on unseen test data using:
1.  **Training MSE:** Mean Squared Error on training data.
2.  **Testing MSE:** Mean Squared Error on test data.
3.  **$R^2$ Score (`r2_score`):** Measures the goodness of fit for the regression model.

---

## 7. ANN for Classification (Date Fruit Dataset Example)
When transitioning an ANN from Regression to Classification (e.g., classifying 7 different types of date fruits from 34 features):

1.  **Network Architecture Adjustments:**
    *   **Input Layer:** Matches the number of input features ($x_i$).
    *   **Hidden Layers:** Typically 1-2 hidden layers with a moderate number of neurons (e.g., 32 or 64 neurons). *Note: If a model underfits, you can increase the number of neurons.*
    *   **Output Layer:** For multi-class classification with $C$ classes (e.g., $C = 7$), the output layer must contain **$C$ neurons** (7 neurons).
2.  **Output Activation & Loss Functions:**
    *   **Logits:** The raw, unactivated continuous output values from the final linear layer (e.g., values like $[0.4, 0.8, 1.3, ...]$ which do not sum to 1 and can be negative or greater than 1).
    *   **Softmax Activation:** By default in multi-class classification training, raw logits are passed through a **Softmax** function to convert them into proper **probabilities** that sum to 1.
    *   **Loss Function:** **Categorical Cross Entropy Loss** (expects targets to be appropriately encoded, such as one-hot encoded vectors like $y_1 = [1, 0, 0]$).