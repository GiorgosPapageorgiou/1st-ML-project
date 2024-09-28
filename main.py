from sklearn.model_selection import train_test_split
import data_cleaning as dc
import decision_tree_model as dt
import model_evaluation as me
import SVM_model as svm
import KNN_model as knn
import Naive_Bayes_model as nb

def main():
    # Load and preprocess the data
    input_file = "data\\mammographic_masses.data.txt"
    df = dc.load_and_preprocess_data(input_file)
    # Normalize the data
    X, y = dc.normalize_data(df)
    # Create a train/test split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train the Decision Tree model (imported from decision_tree_model.py)
    decision_tree_model, features = dt.train_decision_tree(X_train, y_train)
    # Predict the labels for X test
    y_pred_tree = decision_tree_model.predict(X_test)
    print("\nDecision Tree Results:")
    # Evaluate the model on the test set
    me.evaluate_model(y_test, y_pred_tree, "Decision Tree")
    # Plot the decision tree
    #dt.plot_decision_tree(decision_tree_model, features)

    # Create a k-Fold Cross Validation Decision Tree model
    cv_decision_tree_model, features = dt.train_decision_tree(X, y)  # Note: we're training only for cross-validation
    # Perform K-Fold cross validation, but not printing the results
    dt.cross_validate_model(cv_decision_tree_model, X, y, cv=10)

    # Train the SVM model (imported from SVM_model.py)
    svm_model = svm.train_svm(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("\nSVM (Linear Kernel) Results:")
    me.evaluate_model(y_test, y_pred_svm, "SVM (Linear)")

    # Train the KNN model with K=10 (imported from KNN_model.py)
    knn_model = knn.train_knn(X_train, y_train, n_neighbors=20)
    y_pred_knn = knn_model.predict(X_test)
    print("\nK-Nearest Neighbors (K=20) Results:")
    me.evaluate_model(y_test, y_pred_knn, "KNN (K=10)")

    # Train the Naive Bayes model
    nb_model, scaler = nb.train_naive_bayes(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_nb = nb_model.predict(X_test_scaled)
    print("\nNaive Bayes Results:")
    me.evaluate_model(y_test, y_pred_nb, "Naive Bayes")


if __name__ == "__main__":
    main()
