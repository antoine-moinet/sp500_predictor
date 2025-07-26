import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error, r2_score

class FFNModel(keras.Sequential):
    def __init__(self, X_train, y_train, learning_rate=0.001, dropout_rate=0.2):
        super().__init__()
        # Store training data
        self.x_train = X_train
        self.y_train = y_train
        # Define the model architecture
        self.add(keras.layers.Dense(100, activation="relu", input_shape=(X_train.shape[1],)))
        self.add(keras.layers.Dropout(dropout_rate))
        self.add(keras.layers.Dense(200, activation="relu"))
        self.add(keras.layers.Dropout(dropout_rate))
        self.add(keras.layers.Dense(y_train.shape[1], activation="linear"))
        self.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="MeanSquaredError", metrics=["accuracy"])

    def train_model(self, epochs=15, batch_size=128, validation_data=None):
        return self.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate_model(self, X, y):
        y_pred = self.predict(X)
        print("R2:", r2_score(y, y_pred))
        print("MSE:", mean_squared_error(y, y_pred))
        