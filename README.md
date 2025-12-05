# 🎓 Machine Learning Foundations — University Coursework

**Final Grade: 92%** ✨

This repository contains my completed coursework for a university Machine Learning module at QMUL.

The assignment was divided into three parts, each exploring fundamental ML concepts through hands-on implementation in Python using **PyTorch**.

---

## 📁 Repository Contents

| Notebook | Report | Topic |
|----------|--------|-------|
| `wk3_Regression_2024.ipynb` | `ML_REPORT_1_Regression.pdf` | Linear Regression |
| `wk4_Classification_I_2024.ipynb` | `ML_REPORT_2_Classification.pdf` | Logistic Regression & Classification |
| `wk6_NN_2024.ipynb` | `ML_REPORT_3_NN.pdf` | Neural Networks |

---

## 📊 Part 1: Regression

**Dataset:** [Diabetes Dataset](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) — predicting disease progression from patient features.

**Key Implementations:**
- Custom **linear regression model** built from scratch using PyTorch
- **Mean Squared Error (MSE)** loss function
- **Gradient descent** optimization with manual weight updates
- **Feature normalization** (z-score standardization)
- Analysis of **learning rate effects** on convergence and divergence
- **Polynomial regression** with L2 regularization to control overfitting

**Insights:** The model identified BMI, blood pressure, and certain serum markers as the strongest predictors of diabetes progression. Learning rate tuning revealed the critical balance between slow convergence (α=0.001) and numerical instability (α≥1.0).

---

## 🌸 Part 2: Classification

**Dataset:** [Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) — classifying iris flowers into three species.

**Key Implementations:**
- **Logistic regression** classifier with sigmoid activation
- **Binary cross-entropy** loss function
- **One-vs-All (OvA)** multi-class classification strategy
- **Softmax normalization** for proper probability distributions
- Visualization of **decision boundaries**
- Exploration of the **XOR problem** and linear separability

**Insights:** The classifier achieved 100% test accuracy on the Iris dataset. Analysis revealed that Setosa is highly separable from other classes, while Versicolor and Virginica share overlapping feature distributions. The XOR problem demonstrated the fundamental limitations of linear classifiers.

---

## 🧠 Part 3: Neural Networks

**Dataset:** Iris Dataset — exploring how network architecture affects classification performance.

**Key Implementations:**
- **Multi-layer perceptron (MLP)** with configurable hidden layer sizes
- **Random weight initialization** to break symmetry
- **ReLU activation** functions
- Systematic comparison of architectures (1 to 32 hidden neurons)
- Training loss curves and accuracy analysis

**Insights:** 
- Zero initialization creates a **symmetry problem** — all neurons learn identical features
- Neural networks solve XOR by learning **nonlinear decision boundaries** through hidden layers
- Underfitting occurs with too few neurons (1-2), while 8+ neurons approach optimal performance
- Beyond 16 neurons, **diminishing returns** are observed on this relatively simple dataset

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **PyTorch** — tensor operations and neural network building blocks
- **scikit-learn** — datasets and train/test splitting
- **Matplotlib & Seaborn** — data visualization
- **Pandas** — data manipulation

---

## 🚀 Running the Notebooks

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install torch scikit-learn pandas matplotlib seaborn
   ```
3. Open any notebook in Jupyter or Google Colab
4. Run cells sequentially

---

## 📝 Reports

Each PDF report contains detailed analysis and discussion of results, including figures, tables, and interpretations of model behaviour. These accompany the notebooks and provide the theoretical context for each implementation.

---

## 👤 Author

**Ahmed Idris**

