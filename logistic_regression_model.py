from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model