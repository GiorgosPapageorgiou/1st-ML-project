from sklearn.model_selection import train_test_split
import data_cleaning as dc
import decision_tree_model as dt

def main():
    # Load and preprocess the data
    input_file = "..\mammographic_masses.data.txt"
    df = dc.load_and_preprocess_data(input_file)
    # Normalize the data
    X, y = dc.normalize_data(df)

    # Create a train/test split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Train the Decision Tree model
    decision_tree_model, features = dt.train_decision_tree(X_train, y_train)
    # Evaluate the model on the test set
    dt.evaluate_model(decision_tree_model, X_test, y_test)
    # Plot the decision tree
    #dt.plot_decision_tree(decision_tree_model, features)

    # Create a k-Fold Cross Validation Decision Tree model
    cv_decision_tree_model, features = dt.train_decision_tree(X, y)  # Note: we're training only for cross-validation
    # Perform K-Fold cross validation
    dt.cross_validate_model(cv_decision_tree_model, X, y, cv=10)
    

if __name__ == "__main__":
    main()
