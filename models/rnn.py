
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

class RNNModel(nn.Module):
    def __init__(self, X_train, y_train, hidden_size, num_layers, dropout = 0.1):
        super().__init__()
        self.x_train = X_train
        self.y_train = y_train
        # RNN layers
        self.rnn = nn.RNN(X_train.shape[2], hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fnn1 = nn.Linear(hidden_size, 30)
        self.fnn2 = nn.Linear(30, y_train.shape[1])
        # default values for loader, optimizer, and loss function
        dataset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fnn1(out)
        out = torch.relu(out)
        out = self.fnn2(out)
        return out
    
    def modify_loader_otimizer_and_loss(self, loader, optimizer, loss_fn):
        self.loader = loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_model(self, n_epochs, x_test, y_test):
        for epoch in range(n_epochs):
            self.train()
            for x_batch, y_batch in self.loader:
                y_pred = self(x_batch)[:, -1, :]
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Validation
            if epoch % 5 == 0:
                self.eval()
                with torch.no_grad(): # temporarily disable the gradient calculation within its block. 
                    y_pred_train = self(self.x_train)[:, -1, :]
                    y_pred_test = self(x_test)[:, -1, :]
                    train_mse = self.loss_fn(y_pred_train, self.y_train).item()
                    test_mse = self.loss_fn(y_pred_test, y_test).item()
                    print(f"Epoch {epoch}: train MSE {train_mse:.4f}, test MSE {test_mse:.4f}")
    
    def use_trained_model(self, inputs):
        self.eval()
        with torch.no_grad():
            prediction = self(inputs)[:, -1, :]  
        return prediction

    def evaluate_model(self, X, y_true):
        self.eval()
        with torch.no_grad():
            y_pred = self(self.x_train)[:,-1,:]
            print("train R2:", r2_score(self.y_train, y_pred))
            print("train MSE:", mean_squared_error(self.y_train, y_pred))
            y_pred = self(X)[:,-1,:]
            print("Test R2:", r2_score(y_true, y_pred))
            print("Test MSE:", mean_squared_error(y_true, y_pred))
        return 
    


