# **Mammographic Mass Classification using Machine Learning**

This project aims to predict whether a mammogram mass is benign or malignant using various machine learning techniques. The project is based on the "Mammographic Masses" dataset from the UCI repository. Multiple supervised learning algorithms have been applied and evaluated to find the most accurate model for binary classification.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#Dependencies)
- [Results](#results)


---

## **Project Overview**

Early detection and diagnosis of breast cancer is crucial for effective treatment. This project applies machine learning algorithms to predict whether a mammographic mass is benign or malignant based on four features: age, shape, margin, and density. Several models are trained and evaluated, and their performance is compared to select the best approach for this classification task.

---

## **Dataset**

- **Source**: The "Mammographic Masses" dataset is publicly available from the UCI Machine Learning Repository: [Mammographic Mass Dataset](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
- **Number of Instances**: 961
- **Features**:
  - **Age**: Age of the patient in years.
  - **Shape**: Shape of the mass (1: Round, 2: Oval, 3: Lobular, 4: Irregular).
  - **Margin**: Mass margin (1: Circumscribed, 2: Microlobulated, 3: Obscured, 4: Ill-defined, 5: Spiculated).
  - **Density**: Mass density (1: High, 2: Iso, 3: Low, 4: Fat-containing).
- **Target**: Severity (0: Benign, 1: Malignant).

---

## **Project Structure**

The project is divided into different scripts to improve maintainability and clarity.

- `main.py`: The main script that orchestrates the entire pipeline including data processing, model training, and evaluation.
- `data_cleaning.py`: Contains functions for data loading and cleaning (handling missing values, feature selection).
- `decision_tree_model.py`: Implements the Decision Tree model along with cross-validation and model evaluation.
- `knn_model.py`: Implements the K-Nearest Neighbors (KNN) model.
- `logistic_regression_model.py`: Implements Logistic Regression for binary classification.
- `naive_bayes_model.py`: Implements Naive Bayes classification.
- `nn_model.py`: Implements the Artificial Neural Network (ANN) model using Keras.
- `model_evaluation.py`: Contains functions for evaluating the models using metrics such as accuracy, precision, recall, and F1-score.
- `data folder`: Contains our data (mammographic_masses.data.txt, mammographic_masses.names.txt).
- `decision_tree_visualization.png`: A visualization of the decision tree model.

---

## **Installation**


##### 1. Clone the Project
Select a directory n your local machine, and then clone the repo with thw follow command:

```
git clone https://github.com/GiorgosPapageorgiou/1st-ML-project.git
```

##### 2. Run the Project
To run the entire pipeline (data preprocessing, model training, and evaluation), execute the `main.py` script:

```
python main.py
```

## **Dependencies**

The project relies on the following dependencies, which are listed in `requirements.txt`:

```txt
pandas
numpy
scikit-learn
tensorflow
matplotlib
```

## Results

The following table summarizes the performance of different models:


| **Model**               | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-------------------------|--------------|---------------|------------|---------------|
| Decision Tree           | 0.747        | 0.745         | 0.699      | 0.721         |
| SVM with Linear Kernel  | 0.826        | 0.789         | 0.858      | 0.822         |
| K-Nearest Neighbors     | 0.809        | 0.807         | 0.779      | 0.793         |
| Naive Bayes            | 0.751        | 0.773         | 0.664      | 0.714         |
| Logistic Regression     | 0.822        | 0.802         | 0.823      | 0.812         |
| Neural Network          | 0.830        | 0.795         | 0.858      | 0.826         |
