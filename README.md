
# Early Stopping on 95% Accuracy for Fashion MNIST Classification

## Overview

This script trains a basic neural network to classify clothing items from the Fashion MNIST dataset. It also includes a custom callback that **automatically stops training once accuracy crosses 95%**.

This is a good example of how to prevent unnecessary training time once your model has already reached a satisfactory performance.

---

## Dataset Used

The script uses **Fashion MNIST**, a popular dataset of 28x28 grayscale images of clothing items. It includes:

* 60,000 training images
* 10,000 test images
* 10 clothing categories

---

## Key Features

### 1. Normalization

The pixel values in the images are scaled to the range \[0, 1] by dividing by 255. This helps the model converge faster.

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

---

### 2. Model Architecture

This is a simple feedforward (Dense) network:

* **Flatten Layer**: Converts 2D images into a 1D vector.
* **Dense Layer (512 units, ReLU)**: Learns features from input data.
* **Dense Output Layer (10 units, Softmax)**: Outputs the probability for each clothing category.

```python
tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

---

### 3. Callback for Early Stopping

A custom callback class is defined using `tf.keras.callbacks.Callback`. It checks the accuracy after each epoch and **stops training** once accuracy goes above 95%.

```python
if logs.get('accuracy') > 0.95:
    self.model.stop_training = True
```

This saves time and computing power if your model is already good enough.

---

## How to Run

### Requirements

Make sure you have TensorFlow installed:

```
pip install tensorflow
```

### Run the Script

Save the script as `early_stop_fashion_mnist.py` and run:

```
python early_stop_fashion_mnist.py
```

The model will train and stop early if it reaches over 95% training accuracy.

---

## What You Can Try Next

* Increase or decrease the target accuracy in the callback.
* Try the same callback on more complex CNN models.
* Add validation data and stop based on **validation accuracy** instead of training accuracy.

---

## Why This is Useful

Callbacks like this are practical tools in real-world training scenarios. They help automate model management — especially when training takes time or resources are limited. This pattern is reusable in almost any TensorFlow/Keras project.

Let me know if you’d like to see how to stop based on **validation accuracy** or use other callbacks like `ModelCheckpoint` or `ReduceLROnPlateau`.
