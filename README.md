# Digit-Classifier-KNN
Handwritten digit classification using KNN on the MNIST dataset
# Digit Classifier using KNN

Handwritten digit classification using K-Nearest Neighbors (KNN) on the MNIST dataset.

## Project Overview
This project trains a KNN classifier to recognize handwritten digits (0-9) from grayscale images. 
It covers data preprocessing, dimensionality reduction with PCA, model training, evaluation, and 
an in-depth look at how KNN actually makes predictions.

## Dataset
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — 70,000 handwritten digit images (28x28 grayscale), 
loaded directly via `tensorflow.keras.datasets.mnist`.

## What's Inside
- Data loading, visualization, and preprocessing (flattening + normalization)
- Dimensionality reduction using PCA (784 → 100 dimensions) for speed
- KNN model training and evaluation (accuracy, classification report, confusion matrix)
- Finding the optimal k value
- Visualizing how KNN makes decisions (nearest neighbor analysis)
- Effect of normalization on accuracy
- Per-class accuracy breakdown and misclassification analysis

## Results
- **Accuracy: 93.10%** (k=5)
- **Best result: 93.50%** at k=1
- PCA reduced training time significantly with minimal accuracy loss

## Tech Stack
Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn

## Future Improvements
- Try a CNN (Convolutional Neural Network) for higher accuracy (~99%)
- Train on the full dataset (60,000 images) instead of a subset
- Test with custom, real-world handwritten digit images
