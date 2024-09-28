from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, n_neighbors=10):
    """
    Trains a K-Nearest Neighbors classifier on the training data.
    """
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model
