from datetime import datetime

import pandas as pd
import plotly.io as pio
import torch
import plotly.express as px
import plotly.graph_objects as go
import torch.nn as nn
import torchmetrics
from matplotlib import pyplot as plt
from tqdm import tqdm
# from tqdm.notebook import tqdm
import sys
import time
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')
from Data_Preparation.final_df import get_df
from Data_Preparation.feature_engineering import append_date_features, create_trigonometric_columns
from Data_Preparation.subsets_and_target import create_target_variable, split_train_valid_test


# data
df = get_df("BTCUSDT", "4h", "60000h")
# target, features
coin_df, target, features = create_target_variable(df)
# training, validation, testing subsets
train_data, valid_data, test_data = split_train_valid_test(data=coin_df)


def scale_data(train_data, valid_data, test_data):
    target_mean = train_data[target].mean()
    target_stdev = train_data[target].std()
    train_scaled = train_data.copy()
    valid_scaled = valid_data.copy()
    test_scaled = test_data.copy()

    for c in train_data.columns:
        mean = train_data[c].mean()
        stdev = train_data[c].std()

        train_scaled[c] = (train_data[c] - mean) / stdev
        valid_scaled[c] = (valid_data[c] - mean) / stdev
        test_scaled[c] = (test_scaled[c] - mean) / stdev

    train_scaled = train_scaled.astype(np.float64)
    valid_scaled = valid_scaled.astype(np.float64)
    test_scaled = test_scaled.astype(np.float64)
    return train_scaled, valid_scaled, test_scaled, target_mean, target_stdev


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


def get_dataset_obj(dataframe, features, target, sequence_length):
    sequence_dataset = SequenceDataset(
        dataframe=dataframe,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    return sequence_dataset


def get_dataloader(dataset_obj, batch_size, do_shuffle=False):
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=do_shuffle)
    return loader


def prepare_data(symbol: str, interval: str, lookback: str, sequence_length: int, batch_size: int):
    df = get_df(symbol, interval, lookback)
    df, target, features = create_target_variable(df=df)
    train_df, valid_df, test_df = split_train_valid_test(data=df)
    train_scaled, valid_scaled, test_scaled, target_mean, target_stdev = scale_data(train_df, valid_df, test_df)

    # initialize Dataset objects
    train_dataset = get_dataset_obj(train_scaled, target=target, features=features, sequence_length=sequence_length)
    validation_dataset = get_dataset_obj(valid_scaled, target=target, features=features, sequence_length=sequence_length)
    test_dataset = get_dataset_obj(test_scaled, target=target, features=features, sequence_length=sequence_length)

    # initialize DataLoader objects
    train_loader = get_dataloader(train_dataset, batch_size=batch_size)
    validation_loader = get_dataloader(validation_dataset, batch_size=batch_size)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


class DeepRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.fc1 = nn.Linear(in_features=hidden_units, out_features=64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = hn[-1]  # Select the output of the last LSTM layer
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out).squeeze()  # Squeeze to remove the last dimension of size 1

        return out


# scale data
train_scaled, valid_scaled, test_scaled, target_mean, target_stdev = scale_data(train_data, valid_data, test_data)

# sequence dataset
sequence_length = 3
train_dataset = SequenceDataset(
                        train_scaled,
                        target=target,
                        features=features,
                        sequence_length=sequence_length
                )

# data class
sequence_length = 16
train_dataset = get_dataset_obj(train_scaled, target=target, features=features, sequence_length=sequence_length)
validation_dataset = get_dataset_obj(valid_scaled, target=target, features=features, sequence_length=sequence_length)
test_dataset = get_dataset_obj(test_scaled, target=target, features=features, sequence_length=sequence_length)

# data loader
batch_size = 16
torch.manual_seed(99)
# train_loader = get_dataloader(train_dataset, batch_size=batch_size, do_shuffle=True)
# validation_loader = get_dataloader(validation_dataset, batch_size=batch_size)
# test_loader = get_dataloader(test_dataset, batch_size=batch_size)
# X, y = next(iter(train_loader))

train_loader, validation_loader, test_loader = prepare_data(symbol="BTCUSDT", interval="4h", lookback="60000h", sequence_length=16, batch_size=16)

# model
model = DeepRegressionLSTM(num_sensors=len(features), hidden_units=32, num_layers=2)


def calculate_evaluation_metrics(y_pred, y_true, loss_fn):
    mse = loss_fn(y_pred, y_true)
    mae = torch.mean(torch.abs(y_pred - y_true))
    r2 = torchmetrics.functional.r2_score(y_pred.view(-1), y_true.view(-1))
    rmse = torch.sqrt(torch.mean(torch.pow(y_pred - y_true, 2)))

    return mse, mae, r2, rmse


def train_model(data_loader, model, loss_function, optimizer, ix_epoch) -> dict:
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    mse_list, mae_list, r2_list, rmse_list = [], [], [], []

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        # computes gradients of the loss
        loss.backward()
        # updates the model parameters
        optimizer.step()

        total_loss += loss.item()

        mse, mae, r2, rmse = calculate_evaluation_metrics(y_pred=output, y_true=y, loss_fn=loss_function)
        mse_list.append(mse.item())
        mae_list.append(mae.detach().numpy())
        r2_list.append(r2.detach().numpy())
        rmse_list.append(rmse.detach().numpy())

    mse = sum(mse_list) / num_batches
    mae = sum(mae_list) / num_batches
    r2 = sum(r2_list) / num_batches
    rmse = sum(rmse_list) / num_batches
    print("Epoch {}, Train || MSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}, RMSE: {:.2f}".format(ix_epoch, mse, mae, r2, rmse))
    metrics = {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}
    return metrics


def evaluate_model(data_loader, model, loss_function, ix_epoch=None) -> dict:
    num_batches = len(data_loader)
    total_loss = 0

    mse_list, mae_list, r2_list, rmse_list = [], [], [], []

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()
            mse, mae, r2, rmse = calculate_evaluation_metrics(y_pred=output, y_true=y, loss_fn=loss_function)
            mse_list.append(mse.item())
            mae_list.append(mae.detach().numpy())
            r2_list.append(r2.detach().numpy())
            rmse_list.append(rmse.detach().numpy())

    mse = sum(mse_list) / num_batches
    mae = sum(mae_list) / num_batches
    r2 = sum(r2_list) / num_batches
    rmse = sum(rmse_list) / num_batches
    if ix_epoch is not None:
        print("Epoch {}, Evaluation || MSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}, RMSE: {:.2f}".format(ix_epoch, mse, mae, r2, rmse))
    metrics = {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}
    return metrics


def train_and_evaluate_model(train_loader, val_loader, model, loss_function, learning_rate, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    results = {"hyperparameters": {"learning_rate": learning_rate, "epochs": epochs},
               "train_metrics": {"mse": [], "mae": [], "r2": [], "rmse": []},
               "val_metrics": {"mse": [], "mae": [], "r2": [], "rmse": []}}

    print("Untrained validation\n--------")
    evaluate_model(val_loader, model, loss_function, ix_epoch=0)
    print()

    start = time.time()
    for ix_epoch in tqdm(range(epochs), desc="Training coin"):
        print("\n---------")
        train_metrics = train_model(train_loader, model, loss_function, optimizer=optimizer, ix_epoch=ix_epoch + 1)
        val_metrics = evaluate_model(val_loader, model, loss_function, ix_epoch=ix_epoch + 1)
        print()

        for metric_name, value in train_metrics.items():
            results['train_metrics'][metric_name].append(value)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

    end = time.time()
    exec_time = end - start
    results['time'] = exec_time
    return model, results

print(train_loader)
epochs = 30
learning_rate = 0.001
loss_function = nn.MSELoss()
trained_model, results = train_and_evaluate_model(train_loader, validation_loader, model, loss_function, learning_rate, epochs)


def plot_losses(results, metric: str):
    plt.plot(results['train_metrics'][metric])
    plt.plot(results['val_metrics'][metric])
    plt.title(f'{metric} loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()

plot_losses(results, metric='mse')
plot_losses(results, metric='mae')


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_pred = model(X)
            output = torch.cat((output, y_pred), 0)

    return output


train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

forecast_col = "Model forecast"
train_scaled[forecast_col] = predict(train_eval_loader, trained_model).numpy()
valid_scaled[forecast_col] = predict(validation_loader, trained_model).numpy()

df_out = pd.concat((train_scaled, valid_scaled))[[target, forecast_col]]

# plot train + validation
pio.templates.default = "plotly_white"
plot_template = dict(
    layout=go.Layout({
        "font_size": 18,
        "xaxis_title_font_size": 24,
        "yaxis_title_font_size": 24})
    )

fig = px.line(df_out, labels=dict(created_at="Date", value="Close price"), title="Actual vs Forecast")
fig.add_vline(x=datetime(2022, 3, 1), line_width=4, line_dash="dash")
fig.add_annotation(xref="paper", x=0.8, yref="paper", y=0.8, text="Validation set start", showarrow=False)
fig.update_layout(template=plot_template, legend=dict(orientation='h', y=1.02, title_text=""))
fig.show()


metrics = evaluate_model(test_loader, trained_model, loss_function, ix_epoch=None)
forecast_col = "Model forecast"
test_scaled[forecast_col] = predict(test_loader, model).numpy()
test_df_out = test_scaled.loc[:, [target, forecast_col]]
fig = px.line(test_df_out, x=test_df_out.index, y=[target, forecast_col],
              labels=dict(created_at="Date", value="Close price", title="Actual vs Forecast"))
fig.show()