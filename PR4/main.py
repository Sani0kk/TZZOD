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


def compute_metrics(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    print(f"  {name:<8} — MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | MAPE: {mape:.2f}%")
    return {'model': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}


if __name__ == '__main__':
    print("Завантаження даних...")
    df = load_data('AEP_hourly.csv')
    print(f"Рядків завантажено: {len(df)}")
    print(f"Діапазон: {df['Datetime'].min()} — {df['Datetime'].max()}")

    df_sample = df.tail(5000).reset_index(drop=True)
    print(f"Використовується останніх {len(df_sample)} рядків для прискорення навчання\n")

    split = int(len(df_sample) * TRAIN_RATIO)
    df_train = df_sample.iloc[:split].reset_index(drop=True)
    df_test  = df_sample.iloc[split:].reset_index(drop=True)

    print(f"Тренувальна вибірка: {len(df_train)} рядків")
    print(f"Тестова вибірка:     {len(df_test)} рядків\n")

    print("=== LSTM (PyTorch) ===")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(df_train[['MW']]).flatten()
    test_scaled  = scaler.transform(df_test[['MW']]).flatten()

    all_scaled = np.concatenate([train_scaled, test_scaled])
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LEN)

    X_test_seq = []
    for i in range(len(test_scaled)):
        seq = all_scaled[split - SEQUENCE_LEN + i : split + i]
        if len(seq) < SEQUENCE_LEN:
            seq = np.pad(seq, (SEQUENCE_LEN - len(seq), 0), mode='edge')
        X_test_seq.append(seq)
    X_test_seq = np.array(X_test_seq)

    lstm_model = train_lstm(X_train, y_train)

    lstm_model.eval()
    X_tensor = torch.tensor(X_test_seq, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        lstm_preds_scaled = lstm_model(X_tensor).squeeze().numpy()

    lstm_preds_mw = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
    actual_mw = df_test['MW'].values

    print("\n=== ARIMA ===")
    arima_model = ARIMA(df_train['MW'].values, order=(2, 1, 2))
    arima_fitted = arima_model.fit()
    arima_preds_mw = np.array(arima_fitted.forecast(steps=len(df_test)))
    print("  ARIMA навчання завершено")

    print("\n=== Prophet ===")
    df_prophet_train = df_train.rename(columns={'Datetime': 'ds', 'MW': 'y'})
    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    prophet_model.fit(df_prophet_train)
    future_df = pd.DataFrame({'ds': df_test['Datetime'].values})
    prophet_forecast = prophet_model.predict(future_df)
    prophet_preds_mw = prophet_forecast['yhat'].values
    print("  Prophet навчання завершено")

    n = min(len(actual_mw), len(lstm_preds_mw), len(arima_preds_mw), len(prophet_preds_mw))
    actual_mw        = actual_mw[:n]
    lstm_preds_mw    = lstm_preds_mw[:n]
    arima_preds_mw   = arima_preds_mw[:n]
    prophet_preds_mw = prophet_preds_mw[:n]

    print("\n=== Метрики точності (на тестовій вибірці) ===")
    metrics = []
    metrics.append(compute_metrics(actual_mw, lstm_preds_mw,    'LSTM'))
    metrics.append(compute_metrics(actual_mw, arima_preds_mw,   'ARIMA'))
    metrics.append(compute_metrics(actual_mw, prophet_preds_mw, 'Prophet'))

    metrics_df = pd.DataFrame(metrics)
    best_model = metrics_df.loc[metrics_df['MAE'].idxmin(), 'model']
    print(f"\nНайточніша модель за MAE: {best_model}")

    print("\nПобудова графіку...")
    n_plot = min(FORECAST_STEPS, n)
    dates  = df_test['Datetime'].values[:n_plot]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Прогнозування енергоспоживання: LSTM vs ARIMA vs Prophet', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(dates, actual_mw[:n_plot],        label='Фактичне', color='black',     linewidth=2.0)
    ax1.plot(dates, lstm_preds_mw[:n_plot],    label='LSTM',     color='royalblue', linewidth=1.4, linestyle='--')
    ax1.plot(dates, arima_preds_mw[:n_plot],   label='ARIMA',    color='tomato',    linewidth=1.4, linestyle='-.')
    ax1.plot(dates, prophet_preds_mw[:n_plot], label='Prophet',  color='seagreen',  linewidth=1.4, linestyle=':')
    ax1.set_title('Прогноз на 72 години вперед')
    ax1.set_ylabel('Енергоспоживання (MW)')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    model_names = [m['model'] for m in metrics]
    mae_values  = [m['MAE']   for m in metrics]
    colors = ['royalblue', 'tomato', 'seagreen']
    bars = ax2.bar(model_names, mae_values, color=colors, width=0.4)
    for bar, val in zip(bars, mae_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    ax2.set_title('Порівняння точності моделей (MAE — менше = краще)')
    ax2.set_ylabel('MAE (MW)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графік збережено: forecast_comparison.png")
