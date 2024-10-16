from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from preprocess import preprocess_data

def build_and_train_model():
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data('data/logistics_data.csv')

    # Build the neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save('models/delay_prediction_model.h5')

    return model

if __name__ == "__main__":
    model = build_and_train_model()
    print("Model training completed and saved!")
