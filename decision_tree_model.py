from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

def train_decision_tree(X_train, y_train, random_state=42):
    """
    Trains a Decision Tree classifier on the training data.
    """
    decision_tree = DecisionTreeClassifier(random_state=random_state)
    decision_tree.fit(X_train, y_train)
    features = ["Age", "Shape", "Margin", "Density"]
    return decision_tree, features

def cross_validate_model(model, X, y, cv=10):
    """
    Performs K-Fold cross validation on the given model.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    #print(f"Cross-validated accuracy scores: {scores}")
    #print(f"Mean accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
    return scores

def plot_decision_tree(model, feature_names):
    """
    Plots the decision tree.
    """
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()
    