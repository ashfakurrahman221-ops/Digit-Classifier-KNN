# Digit-Classifier-KNN
Handwritten digit classification using KNN on the MNIST dataset
# Digit Classifier using KNN

Handwritten Digit Classifier (KNN on MNIST)

A machine learning project that classifies handwritten digits (0–9) from the MNIST dataset using a K-Nearest Neighbors (KNN) classifier, with PCA for dimensionality reduction and a deep dive into why and how KNN makes its decisions.

🎯 Problem Statement

Recognizing handwritten digits automatically is a classic computer vision problem with real applications — reading postal codes, digitizing forms, bank check processing, etc. This project builds and analyzes a KNN-based classifier on the MNIST dataset, focusing not just on accuracy but on understanding the model's behavior: which digits it confuses, why normalization/PCA matter, and how the "nearest neighbor" decision actually works under the hood.

📂 Dataset
MNIST handwritten digit dataset, loaded directly via tensorflow.keras.datasets.mnist
60,000 training images + 10,000 test images, each 28×28 grayscale
For faster experimentation, a subset was used: 10,000 training / 2,000 test images
⚙️ Preprocessing
Images flattened from 28×28 → 784-length vectors
Pixel values normalized from 0–255 → 0–1
PCA applied to reduce dimensionality from 784 → 100 components, retaining 91.7% of the variance — dramatically speeding up training/inference with minimal accuracy loss
🤖 Model: K-Nearest Neighbors

KNN classifies a digit by looking at its k closest training images (by distance in PCA space) and taking a majority vote among their labels — no traditional "training" step, just distance comparison at prediction time.

Finding the Best k
k	Accuracy
1	93.50%
3	93.10%
5	93.10%
7	93.00%
9	92.70%
11	92.30%

Accuracy peaks at k=1 and gradually declines as k increases — a hallmark of the bias-variance tradeoff (very small k → low bias/high variance, closer to overfitting; larger k → smoother but less locally precise decisions).

Normalization Check
Setting	Accuracy
Without normalization (0–255)	92.05%
With normalization (0–1)	92.30%

Normalization gave a small but consistent improvement even though all pixel features already share the same scale — the effect would be far larger with mixed-scale features.

📊 Evaluation

Overall accuracy: 93.10% (k=5, on 2,000 test images)

Per-class (per-digit) accuracy:

Digit	Accuracy		Digit	Accuracy
0	99.43%		5	91.62%
1	100.00%		6	97.19%
2	90.41%		7	91.22%
3	93.24%		8	83.85% (weakest)
4	92.17%		9	91.75%
Strongest digit: 1 (100.00%)
Weakest digit: 8 (83.85%) — most often confused with digit 3 (10 misclassifications in the test set), since the two share visually similar curved strokes in pixel space

A full confusion matrix was plotted to visualize misclassification patterns across all 10 digits.

🔍 Model Interpretability

Beyond just reporting accuracy, this project visually explains how KNN reaches its decisions:

Nearest-neighbor visualization — for a sample query digit, the 5 actual closest training images (with their distances) are displayed side-by-side, showing exactly which neighbors "voted" for the prediction
Misclassification analysis — for a wrongly predicted digit, its nearest neighbors are shown to reveal why the model got confused
Self-check Q&A embedded in the notebook, reasoning through: why visually similar digits (4/9, 3/8) get confused, why PCA barely affects accuracy despite cutting dimensions 784→100, and why accuracy drops as k increases
🛠️ Tech Stack

Language: Python

Libraries: TensorFlow/Keras (dataset loading), Scikit-learn (KNeighborsClassifier, PCA, metrics), NumPy, Matplotlib, Seaborn

📁 Project Structure
Digit-Classifier-KNN/
│
├── digit-classifier-knn.ipynb   # Full pipeline: load → preprocess → PCA → KNN → evaluation → interpretability
└── README.md                     # Project documentation
▶️ How to Run
Clone the repository
bash
   git clone https://github.com/ashfakurrahman221-ops/Digit-Classifier-KNN.git
   cd Digit-Classifier-KNN
Install dependencies
bash
   pip install tensorflow scikit-learn matplotlib seaborn numpy
Open and run the notebook
bash
   jupyter notebook digit-classifier-knn.ipynb

(MNIST downloads automatically via tensorflow.keras.datasets.mnist)

🚀 Limitations & Future Improvements
Trained on a 10,000/2,000 subset rather than the full 60,000/10,000 MNIST set — accuracy could improve with the full dataset
KNN's prediction time scales with training set size; a CNN would likely outperform KNN in both accuracy and inference speed on the full dataset
Could explore alternative distance metrics or weighted-KNN to better separate visually similar digits (e.g., 8 vs. 3)
👤 Author

Ashfakur Rahman 📧 ashfakurrahman221@gmail.com

📄 License

This project is open-source and available for educational purposes.
