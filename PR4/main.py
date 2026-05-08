import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

from prophet import Prophet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEQUENCE_LEN = 24
FORECAST_STEPS = 72
TRAIN_RATIO = 0.85
BATCH_SIZE = 64
EPOCHS = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['Datetime', 'MW']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    df = df.dropna()
    return df


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(X_train, y_train):
    X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(loader):.6f}")

    return model


def predict_lstm(model, X_test):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        preds = model(X_tensor).squeeze().numpy()
    return preds


def train_arima(train_series):
    model = ARIMA(train_series, order=(2, 1, 2))
    fitted = model.fit()
    return fitted


def predict_arima(fitted_model, steps):
    forecast = fitted_model.forecast(steps=steps)
    return forecast.values


def train_prophet(df_train):
    df_prophet = df_train.rename(columns={'Datetime': 'ds', 'MW': 'y'})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)
    return model


def predict_prophet(model, future_dates):
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)
    return forecast['yhat'].values


def compute_metrics(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    print(f"  {name:<8} — MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | MAPE: {mape:.2f}%")
    return {'model': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}


def plot_results(df_test, lstm_pred, arima_pred, prophet_pred, scaler):
    actual = scaler.inverse_transform(df_test[['MW']].values).flatten()
    lstm_inv = scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Прогнозування енергоспоживання: LSTM vs ARIMA vs Prophet', fontsize=14, fontweight='bold')

    dates = df_test['Datetime'].values[:FORECAST_STEPS]

    ax1 = axes[0]
    ax1.plot(dates, actual[:FORECAST_STEPS], label='Фактичне', color='black', linewidth=1.5)
    ax1.plot(dates, lstm_inv[:FORECAST_STEPS], label='LSTM', color='royalblue', linewidth=1.2, linestyle='--')
    ax1.plot(dates, arima_pred[:FORECAST_STEPS], label='ARIMA', color='tomato', linewidth=1.2, linestyle='-.')
    ax1.plot(dates, prophet_pred[:FORECAST_STEPS], label='Prophet', color='seagreen', linewidth=1.2, linestyle=':')
    ax1.set_title('Прогноз на 72 години вперед')
    ax1.set_ylabel('Енергоспоживання (MW)')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(alpha=0.3)

    metrics_data = {
        'Модель': ['LSTM', 'ARIMA', 'Prophet'],
        'MAE (MW)': [
            mean_absolute_error(actual[:FORECAST_STEPS], lstm_inv[:FORECAST_STEPS]),
            mean_absolute_error(actual[:FORECAST_STEPS], arima_pred[:FORECAST_STEPS]),
            mean_absolute_error(actual[:FORECAST_STEPS], prophet_pred[:FORECAST_STEPS]),
        ]
    }
    ax2 = axes[1]
    colors = ['royalblue', 'tomato', 'seagreen']
    bars = ax2.bar(metrics_data['Модель'], metrics_data['MAE (MW)'], color=colors, width=0.4)
    for bar, val in zip(bars, metrics_data['MAE (MW)']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    ax2.set_title('Порівняння точності моделей (MAE — менше = краще)')
    ax2.set_ylabel('MAE (MW)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nГрафік збережено: forecast_comparison.png")


if __name__ == '__main__':
    print("Завантаження даних...")
    df = load_data('AEP_hourly.csv')
    print(f"Рядків завантажено: {len(df)}")
    print(f"Діапазон: {df['Datetime'].min()} — {df['Datetime'].max()}")

    df_sample = df.tail(5000).reset_index(drop=True)
    print(f"Використовується останніх {len(df_sample)} рядків для прискорення навчання\n")

    split = int(len(df_sample) * TRAIN_RATIO)
    df_train = df_sample.iloc[:split]
    df_test = df_sample.iloc[split:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(df_train[['MW']]).flatten()
    test_scaled = scaler.transform(df_test[['MW']]).flatten()

    print("=== LSTM (PyTorch) ===")
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LEN)
    X_test, y_test = create_sequences(test_scaled, SEQUENCE_LEN)
    lstm_model = train_lstm(X_train, y_train)
    lstm_preds_scaled = predict_lstm(lstm_model, X_test)

    print("\n=== ARIMA ===")
    arima_train = df_train['MW'].values
    arima_fitted = train_arima(arima_train)
    arima_preds = predict_arima(arima_fitted, steps=len(df_test))
    print("  ARIMA навчання завершено")

    print("\n=== Prophet ===")
    prophet_model = train_prophet(df_train)
    future_dates = df_test['Datetime'].values
    prophet_preds = predict_prophet(prophet_model, future_dates)
    print("  Prophet навчання завершено")

    print("\n=== Метрики точності (на тестовій вибірці) ===")
    actual_test = scaler.inverse_transform(test_scaled[SEQUENCE_LEN:].reshape(-1, 1)).flatten()
    lstm_inv = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
    arima_aligned = arima_preds[SEQUENCE_LEN:]
    prophet_aligned = prophet_preds[SEQUENCE_LEN:]

    n = min(len(actual_test), len(arima_aligned), len(prophet_aligned), len(lstm_inv))
    actual_test = actual_test[:n]
    lstm_inv = lstm_inv[:n]
    arima_aligned = arima_aligned[:n]
    prophet_aligned = prophet_aligned[:n]

    metrics = []
    metrics.append(compute_metrics(actual_test, lstm_inv, 'LSTM'))
    metrics.append(compute_metrics(actual_test, arima_aligned, 'ARIMA'))
    metrics.append(compute_metrics(actual_test, prophet_aligned, 'Prophet'))

    metrics_df = pd.DataFrame(metrics)
    best_model = metrics_df.loc[metrics_df['MAE'].idxmin(), 'model']
    print(f"\nНайточніша модель за MAE: {best_model}")

    print("\nПобудова графіку...")
    plot_results(df_test.reset_index(drop=True), lstm_preds_scaled, arima_aligned, prophet_aligned, scaler)
