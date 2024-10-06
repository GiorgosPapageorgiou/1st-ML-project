import os

# Set the environment variable before importing TensorFlow, in order not to print unnecessary info
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.model_selection import train_test_split
import data_cleaning as dc
import decision_tree_model as dt
import model_evaluation as me
import SVM_model as svm
import KNN_model as knn
import Naive_Bayes_model as nb
import logistic_regression_model as lr
import neural_network_model as nn


def main():
    # Load and preprocess the data
    input_file = "data\\mammographic_masses.data.txt"
    df = dc.load_and_preprocess_data(input_file)
    # Normalize the data
    X, y = dc.normalize_data(df)
    # Create a train/test split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Create a list for keeping its model with its metrics, in order to be displayed in the end
    models_metrics = []

    # Train the Decision Tree model (imported from decision_tree_model.py)
    decision_tree_model, features = dt.train_decision_tree(X_train, y_train)
    # Predict the labels for X test
    y_pred_tree = decision_tree_model.predict(X_test)
    print("\nDecision Tree Results:")
    # Evaluate the model on the test set
    accuracy, precision, recall, f1 = me.evaluate_model(y_test, y_pred_tree, "Decision Tree")
    models_metrics.append(['Decision Tree' , accuracy, precision, recall, f1])
    # Change line
    print(" ")
    # Plot the decision tree
    #dt.plot_decision_tree(decision_tree_model, features)

    # Create a k-Fold Cross Validation Decision Tree model
    cv_decision_tree_model, features = dt.train_decision_tree(X, y)  # Note: we're training only for cross-validation
    # Perform K-Fold cross validation, but not printing the results
    dt.cross_validate_model(cv_decision_tree_model, X, y, cv=10)

    # Train the SVM model (imported from SVM_model.py), and choose the best kernel type
    kernel_type = ['linear', 'poly', 'rbf','sigmoid']
    for kernel in kernel_type :
        svm_model = svm.train_svm(X_train, y_train, kernel)
        y_pred_svm = svm_model.predict(X_test)
        print(f"SVM ({kernel} Kernel) Results:")
        accuracy, precision, recall, f1 = me.evaluate_model(y_test, y_pred_svm, f"SVM ({kernel})")
        if kernel == 'linear' :
            models_metrics.append(['SVM with linear kernel', accuracy, precision, recall, f1])
    # Print the best kernel type
    print("Best SVM approach with Linear.")

    # Train the KNN model with K=10 (imported from KNN_model.py)
    knn_model = knn.train_knn(X_train, y_train, n_neighbors=20)
    y_pred_knn = knn_model.predict(X_test)
    print("\nK-Nearest Neighbors (K=20) Results:")
    accuracy, precision, recall, f1 = me.evaluate_model(y_test, y_pred_knn, "KNN (K=10)")
    models_metrics.append(['K-Nearest Neighbors', accuracy, precision, recall, f1])

    # Train the Naive Bayes model
    nb_model, scaler = nb.train_naive_bayes(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_nb = nb_model.predict(X_test_scaled)
    print("\nNaive Bayes Results:")
    accuracy, precision, recall, f1 = me.evaluate_model(y_test, y_pred_nb, "Naive Bayes")
    models_metrics.append(['Naive Bayes', accuracy, precision, recall, f1])

    # Train the Logistic Regression model
    lr_model = lr.train_logistic_regression(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print("\nLogistic Regression Results:")
    accuracy, precision, recall, f1 = me.evaluate_model(y_test, y_pred_lr, "Logistic Regression")
    models_metrics.append(['Logistic Regression', accuracy, precision, recall, f1])

    # Train the neural network model
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    nn_model = nn.train_neural_network_model( X_train, y_train, epochs=100, input_shape=X_train.shape[1])
    y_pred_nn = nn_model.predict(X_test)
    # Convert probabilities to binary predictions
    y_pred_nn_binary = (y_pred_nn > 0.5).astype(int)  # Thresholding at 0.5
    print("\nNeural Network Results:")
    accuracy, precision, recall, f1 = me.evaluate_model(y_test, y_pred_nn_binary, "Neural Network")
    models_metrics.append(['Neural Network', accuracy, precision, recall, f1])

    print(models_metrics)

if __name__ == "__main__":
    main()
