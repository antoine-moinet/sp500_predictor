
import torch
import numpy as np

def train_rnn(model, n_epochs, loader, optimizer, loss_fn, x_train, x_test, y_train, y_test):
    for epoch in range(n_epochs):
        model.train()
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)[:, -1, :]
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_train = model(x_train)[:, -1, :]
                y_pred_test = model(x_test)[:, -1, :]
                train_rmse = np.sqrt(loss_fn(y_pred_train, y_train).item())
                test_rmse = np.sqrt(loss_fn(y_pred_test, y_test).item())
                print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")
