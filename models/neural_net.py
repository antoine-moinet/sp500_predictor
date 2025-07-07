
import torch
import torch.nn as nn
import tensorflow.keras as keras

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fnn1 = nn.Linear(hidden_size, 30)
        self.fnn2 = nn.Linear(30, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fnn1(out)
        out = torch.relu(out)
        out = self.fnn2(out)
        return out

def create_keras_model(input_dim, output_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(100, activation="relu", input_shape=(input_dim,)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(200, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(output_dim, activation="linear"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="MeanSquaredError", metrics=["accuracy"])
    return model
