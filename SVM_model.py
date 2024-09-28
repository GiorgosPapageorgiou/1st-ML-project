from sklearn.svm import SVC

def train_svm(X_train, y_train, random_state=42):
    """
    Trains an SVM classifier with a linear kernel on the training data.
    """
    svm_model = SVC(kernel='linear', random_state=random_state)
    svm_model.fit(X_train, y_train)
    return svm_model
