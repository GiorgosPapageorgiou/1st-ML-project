from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def train_neural_network_model(X_train, y_train, epochs, input_shape):
    """
    Builds a simple neural network model for binary classification.
    """
    # Add the models layers
    model = Sequential()
     # Add Input layer
    model.add(Input(shape=(input_shape,)))  # Use Input layer to define input shape
    model.add(Dense(16, activation='relu'))  # Hidden layer with 16 neurons
    model.add(Dense(8, activation='relu'))  # Another hidden layer with 8 neurons
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    return model