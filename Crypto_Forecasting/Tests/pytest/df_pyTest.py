import pytest
from Data_Preparation.final_df import *


@pytest.fixture
def sample_data():
    data = {
        'Open': [70507.38],
        'High': [71121.10],
        'Low': [68620.82],
        'Close': [71092.55],
        'Volume': [10154.33622],
        'Year': [2024],
        'Month': [3],
        'Day': [12],
        'Week_of_Year': [11],
        'Year_sin': [0.7273],
        'Year_cos': [0.6864],
        'Month_sin': [0.1411],
        'Month_cos': [-0.99],
        'Day_sin': [-0.5366],
        'Day_cos': [0.8439],
        'RSI': [48.61],
        'EMA': [70663.36],
        'MACD': [-57.10]
    }
    return pd.DataFrame(data, index=pd.to_datetime(['2024-03-12 17:00:00']))

@pytest.fixture
def mock_get_data(monkeypatch, sample_data):
    def mock_get_data_func(symbol, interval, lookback):
        return sample_data

    monkeypatch.setattr('Data_Preparation.final_df.get_data', mock_get_data_func)

@pytest.fixture
def mock_append_date_features(monkeypatch, sample_data):
    def mock_append_date_features_func(data):
        # Dummy function, just return the input data
        return data

    monkeypatch.setattr('Data_Preparation.final_df.append_date_features', mock_append_date_features_func)

@pytest.fixture
def mock_create_trigonometric_columns(monkeypatch, sample_data):
    def mock_create_trigonometric_columns_func(data):
        # Dummy function, just return the input data
        return data

    monkeypatch.setattr('Data_Preparation.final_df.create_trigonometric_columns', mock_create_trigonometric_columns_func)

def test_get_df_returns_dataframe(mock_get_data, mock_append_date_features, mock_create_trigonometric_columns):
    symbol = "BTCUSDT"
    interval = "1h"
    lookback = "100h"

    df = get_df(symbol, interval, lookback)

    assert isinstance(df, pd.DataFrame)

def test_get_df_columns_exist(mock_get_data, mock_append_date_features, mock_create_trigonometric_columns):
    symbol = "BTCUSDT"
    interval = "1h"
    lookback = "100h"

    df = get_df(symbol, interval, lookback)

    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'Month', 'Day', 'Week_of_Year',
                        'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'RSI', 'EMA', 'MACD']

    assert all(col in df.columns for col in expected_columns)

def test_get_df_filled_na_values(mock_get_data, mock_append_date_features, mock_create_trigonometric_columns):
    symbol = "BTCUSDT"
    interval = "1h"
    lookback = "100h"

    df = get_df(symbol, interval, lookback)

    assert not df.isna().any().any()

if __name__ == "__main__":
    pytest.main()