# Mammogram Mass Classification

Six supervised learning algorithms, trained on the same data, compared on the metric that actually matters for cancer screening.

> **This is not a production system and not client work.**
> I built it on my own, for myself, to go deeper into machine learning — to implement the main supervised algorithms hands-on and compare them properly on the same data, rather than just read about them. The dataset is public (UCI), the work was done alongside a Udemy course, and nothing here was ever deployed. **It is not a clinical tool and must not be used to make a medical decision.**

---

## The problem

A mammogram finds a mass. Is it benign, or malignant?

Radiologists answer this well, but a large share of the biopsies they order come back benign — an invasive, expensive, frightening procedure for nothing. The question is whether four simple, cheap observations are enough to predict severity.

## Why accuracy is the wrong metric

In screening, the two ways to be wrong are not equal:

- **False positive** — a benign mass flagged as malignant. Cost: an unnecessary biopsy.
- **False negative** — a malignant mass flagged as benign. Cost: **a missed cancer.**

So the metric to optimise is **recall** — of all the masses that really were malignant, what fraction did the model catch? A model can post a respectable accuracy score while quietly missing a third of the cancers. One of the models below does exactly that.

## The data

[Mammographic Mass dataset](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass), UCI Machine Learning Repository — 961 masses, each with a BI-RADS assessment and a biopsy-confirmed outcome.

**Four features are used:**

| Feature | |
| --- | --- |
| Age | Patient age in years |
| Shape | Round · Oval · Lobular · Irregular |
| Margin | Circumscribed · Microlobulated · Obscured · Ill-defined · Spiculated |
| Density | High · Iso · Low · Fat-containing |

**BI-RADS is deliberately excluded.** It is the radiologist's own 1–5 assessment of how suspicious the mass looks — effectively their prediction of the answer. Training on it would leak the label and inflate every score in the table below. The point of the exercise is to predict severity from the *raw observations*, not to learn to copy the radiologist.

### Missing values

The dataset has holes — masses with no recorded margin, or no age. The lazy fix is to drop those rows and lose the data.

Instead I use **MICE** (multiple imputation by chained equations, via `IterativeImputer`): each missing value is modelled as a function of the other features and filled with a prediction. All 961 records survive.

Features are then standardised with `StandardScaler` and split 75/25 into train and test.

## Results

Six algorithms, same split, same seed:

| Model | Accuracy | Precision | **Recall** | F1 |
| --- | --- | --- | --- | --- |
| **Neural Network** (16 → 8 → 1) | **0.830** | 0.795 | **0.858** | **0.826** |
| **SVM** (linear kernel) | 0.826 | 0.789 | **0.858** | 0.822 |
| Logistic Regression | 0.822 | 0.802 | 0.823 | 0.812 |
| K-Nearest Neighbors (K=20) | 0.809 | 0.807 | 0.779 | 0.793 |
| Naive Bayes | 0.751 | 0.773 | 0.664 | 0.714 |
| Decision Tree | 0.747 | 0.745 | 0.699 | 0.721 |

### Reading the table

**The neural network and the linear SVM tie on recall at 0.858** — both catch roughly 86% of malignant masses. The SVM gets there with a linear decision boundary and no training loop, which is the more interesting result: the extra capacity of the network buys almost nothing. Four features and 961 rows don't have enough structure to reward a deeper model.

**Naive Bayes is the cautionary tale.** Its accuracy (0.751) looks merely mediocre — 8 points behind the leader. Its recall (0.664) is a different story: **it misses a third of the malignant masses.** Pick a model by accuracy alone and you would never notice. This is the whole argument for choosing your metric before you choose your model.

**The single decision tree overfits**, as an unpruned tree on four features will. It finishes last, and the neural network beats it by more than 10 points of F1.

---

## Running it

**Requirements:** Python 3, and:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib fancyimpute
```

Then, from the repository root:

```bash
python main.py
```

It trains all six models in sequence and prints each one's metrics.

> **Note:** the data path in `main.py` is written with a Windows separator (`data\mammographic_masses.data.txt`). On macOS or Linux, switch it to a forward slash before running.

## Project structure

| File | |
| --- | --- |
| `main.py` | Runs the pipeline — load, impute, split, train all six, evaluate |
| `data_cleaning.py` | Loads the raw data, MICE imputation, standardisation |
| `model_evaluation.py` | Accuracy / precision / recall / F1 for a set of predictions |
| `decision_tree_model.py` | Decision tree, plus k-fold cross-validation and tree plotting |
| `SVM_model.py` | SVM — `main.py` sweeps the linear, poly, rbf and sigmoid kernels |
| `KNN_model.py` | K-nearest neighbours |
| `Naive_Bayes_model.py` | Multinomial Naive Bayes (features re-scaled to [0,1] first) |
| `logistic_regression_model.py` | Logistic regression |
| `neural_network_model.py` | Keras sequential net — 16 → 8 → 1, Adam, 100 epochs |
| `data/` | The UCI dataset and its field descriptions |

## Stack

Python · scikit-learn · TensorFlow / Keras · pandas · fancyimpute · matplotlib

## Credits

Dataset: [UCI Machine Learning Repository — Mammographic Mass](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass).
The problem framing comes from *Data Science and Machine Learning with Python* (Udemy); the pipeline, the model comparison and the analysis here are my own.
