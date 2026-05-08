import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def load_data(filepath, nrows=500000):
    df = pd.read_csv(filepath, nrows=nrows)
    return df


def select_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def drop_invalid_rows(df, numeric_cols):
    df = df[numeric_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def normalize_minmax(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns), scaler


def standardize_zscore(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns), scaler


def scale_robust(df):
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns), scaler


def print_stats(label, df, cols):
    print(f"\n--- {label} ---")
    print(df[cols].describe().round(4).to_string())


def plot_comparison(original, minmax, zscore, robust, cols):
    n = len(cols)
    fig, axes = plt.subplots(n, 4, figsize=(18, n * 3))
    fig.suptitle('Порівняння методів масштабування числових колонок', fontsize=14, fontweight='bold')

    titles = ['Оригінал', 'Min-Max [0,1]', 'Z-score (μ=0, σ=1)', 'Robust Scaler']
    datasets = [original, minmax, zscore, robust]

    for i, col in enumerate(cols):
        for j, (data, title) in enumerate(zip(datasets, titles)):
            ax = axes[i][j] if n > 1 else axes[j]
            ax.hist(data[col], bins=60, color=['steelblue', 'darkorange', 'seagreen', 'tomato'][j], alpha=0.8)
            if i == 0:
                ax.set_title(title, fontweight='bold')
            ax.set_ylabel(col if j == 0 else '')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nГрафік збережено: scaling_comparison.png")


if __name__ == '__main__':
    filepath = 'train.csv'

    print("Завантаження даних (500 000 рядків)...")
    df_raw = load_data(filepath, nrows=500000)
    print(f"Розмір датасету: {df_raw.shape[0]} рядків, {df_raw.shape[1]} колонок")
    print(f"Колонки: {list(df_raw.columns)}")

    numeric_cols = select_numeric_columns(df_raw)
    print(f"\nЧислові колонки: {numeric_cols}")

    target_cols = [
        'fare_amount',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count'
    ]
    target_cols = [c for c in target_cols if c in numeric_cols]

    df_clean = drop_invalid_rows(df_raw, target_cols)
    df_clean = df_clean[
        (df_clean['fare_amount'] > 0) &
        (df_clean['fare_amount'] < 500) &
        (df_clean['passenger_count'] > 0) &
        (df_clean['passenger_count'] <= 6) &
        (df_clean['pickup_longitude'].between(-75, -72)) &
        (df_clean['pickup_latitude'].between(40, 42))
    ]
    print(f"\nПісля очищення: {len(df_clean)} рядків")

    print_stats("Оригінальні дані", df_clean, target_cols)

    df_minmax, scaler_mm  = normalize_minmax(df_clean[target_cols])
    df_zscore, scaler_std = standardize_zscore(df_clean[target_cols])
    df_robust, scaler_rob = scale_robust(df_clean[target_cols])

    print_stats("Min-Max нормалізація [0, 1]", df_minmax, target_cols)
    print_stats("Z-score стандартизація (μ=0, σ=1)", df_zscore, target_cols)
    print_stats("Robust Scaler (медіана=0)", df_robust, target_cols)

    print("\n=== Параметри масштабування ===")
    print("\nMin-Max — мінімальні значення за колонками:")
    print(dict(zip(target_cols, scaler_mm.data_min_.round(4))))
    print("Min-Max — максимальні значення за колонками:")
    print(dict(zip(target_cols, scaler_mm.data_max_.round(4))))

    print("\nZ-score — середні значення (mean):")
    print(dict(zip(target_cols, scaler_std.mean_.round(4))))
    print("Z-score — стандартні відхилення (std):")
    print(dict(zip(target_cols, scaler_std.scale_.round(4))))

    plot_cols = ['fare_amount', 'pickup_longitude', 'pickup_latitude']
    plot_cols = [c for c in plot_cols if c in target_cols]

    print("\nПобудова графіку...")
    plot_comparison(
        df_clean[target_cols].reset_index(drop=True),
        df_minmax,
        df_zscore,
        df_robust,
        plot_cols
    )
