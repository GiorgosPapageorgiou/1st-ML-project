from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

def train_naive_bayes(X_train, y_train):
    """
    Trains a Naive Bayes classifier using MultinomialNB.
    The features are scaled using MinMaxScaler to fit the requirements of MultinomialNB.
    """
    # Initialize MinMaxScaler to scale features between 0 and 1
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and train the Naive Bayes classifier
    nb_model = MultinomialNB()
    nb_model.fit(X_train_scaled, y_train)
    return nb_model, scaler
