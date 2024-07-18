import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def lstm_predict(df, column, context_start_index, context_end_index, prediction_length, plot=True):
    # Select the relevant data
    data = df[column].values
    data = data[context_start_index:context_end_index]
    
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences of data
    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    SEQ_LENGTH = 20
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Build the model
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    epochs = 1
    for epoch in range(epochs):
        for seq, labels in zip(X_train, y_train):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq.view(1, SEQ_LENGTH, 1))
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        test_preds = []
        for seq in X_train:
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            test_preds.append(model(seq.view(1, SEQ_LENGTH, 1)).item())
    
    test_preds = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1))
    
    # Predict future values
    last_sequence = torch.tensor(data_scaled[-SEQ_LENGTH:], dtype=torch.float32).view(1, SEQ_LENGTH, 1)
    future_predictions = []
    model.eval()
    for _ in range(prediction_length):
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            pred = model(last_sequence)
            future_predictions.append(pred.item())
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred.view(1, 1, 1)), dim=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    # Flatten the future predictions
    future_predictions = future_predictions.flatten()
    
    # Plotting
    if plot:
        n = len(df)

        # plot a window of actual data two prediction windows to each side
        left_most_index = max(0, context_start_index - 2 * prediction_length)
        right_most_index = min(n, context_end_index + 2 * prediction_length)

        plt.figure(figsize=(8, 4))
        plt.title('LSTM Forecasting')

        plt.plot(range(left_most_index,context_start_index), df[column][left_most_index:context_start_index], color="royalblue", label="Reference data")
        plt.plot(range(context_start_index, context_end_index,), df[column][context_start_index:context_end_index], color="green", label="Context data")
        plt.plot(range(context_end_index,right_most_index), df[column][context_end_index:right_most_index], color="royalblue", label="Post Context Reference Data")
        plt.plot(range(context_end_index, context_end_index + prediction_length), future_predictions, color="tomato", label="Median forecast")

        # plt.plot(data, label='Actual Values', color='blue')
        # plt.plot(range(SEQ_LENGTH, SEQ_LENGTH + len(test_preds)), test_preds, label='Predictions', linestyle='--', color='red')
        plt.show()
    return future_predictions
