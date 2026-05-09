import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os


FILES = {
    'Спокій':                  'Acceleration with g 2026-05-09 02-59-44.zip',
    'Постукування':            'Acceleration with g 2026-05-09 03-00-56.zip',
    'Інтенсивне натискання':   'Acceleration with g 2026-05-09 03-01-49.zip',
}


def load_phyphox_zip(filepath):
    with zipfile.ZipFile(filepath, 'r') as z:
        with z.open('Raw Data.csv') as f:
            df = pd.read_csv(f)

    df.columns = ['time', 'x', 'y', 'z', 'magnitude']
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df.reset_index(drop=True)
    return df


def compute_stats(df):
    mag = df['magnitude']
    return {
        'mean':      round(mag.mean(), 4),
        'std':       round(mag.std(), 4),
        'max':       round(mag.max(), 4),
        'min':       round(mag.min(), 4),
        'amplitude': round(mag.max() - mag.min(), 4),
        'rms':       round(np.sqrt(np.mean(mag**2)), 4),
    }


def plot_time_series(surface_data):
    n = len(surface_data)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle('Часові ряди магнітуди прискорення', fontsize=13, fontweight='bold')

    colors = ['steelblue', 'darkorange', 'tomato']
    for ax, (label, df), color in zip(axes, surface_data.items(), colors):
        ax.plot(df['time'], df['magnitude'], color=color, linewidth=0.7, alpha=0.85)
        ax.axhline(df['magnitude'].mean(), color='red', linewidth=1.2,
                   linestyle='--', label=f"Середнє: {df['magnitude'].mean():.3f} м/с²")
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel('|a| (м/с²)')
        ax.set_xlabel('Час (с)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('time_series.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графік збережено: time_series.png")


def plot_amplitude_comparison(stats_dict):
    labels     = list(stats_dict.keys())
    amplitudes = [stats_dict[s]['amplitude'] for s in labels]
    stds       = [stats_dict[s]['std']       for s in labels]
    rms_vals   = [stats_dict[s]['rms']       for s in labels]

    colors = ['steelblue', 'darkorange', 'tomato']
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Порівняння вібраційних характеристик', fontsize=13, fontweight='bold')

    for ax, values, title in zip(
        axes,
        [amplitudes, stds, rms_vals],
        ['Амплітуда (peak-to-peak)', 'Стандартне відхилення', 'RMS прискорення']
    ):
        bars = ax.bar(x, values, color=colors, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('м/с²')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('amplitude_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графік збережено: amplitude_comparison.png")


def plot_distribution(surface_data):
    n = len(surface_data)
    fig, axes = plt.subplots(1, n, figsize=(15, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle('Розподіл магнітуди прискорення', fontsize=13, fontweight='bold')

    colors = ['steelblue', 'darkorange', 'tomato']
    for ax, (label, df), color in zip(axes, surface_data.items(), colors):
        ax.hist(df['magnitude'], bins=60, color=color, alpha=0.85, edgecolor='white', linewidth=0.3)
        ax.axvline(df['magnitude'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Середнє: {df['magnitude'].mean():.3f}")
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('|a| (м/с²)')
        ax.set_ylabel('Частота')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графік збережено: distribution.png")


if __name__ == '__main__':
    surface_data = {}

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"УВАГА: файл '{filename}' не знайдено — пропускаємо '{label}'")
            continue
        df = load_phyphox_zip(filename)
        surface_data[label] = df
        print(f"Завантажено [{label}]: {len(df)} рядків, тривалість: {df['time'].max():.1f} с")

    if len(surface_data) == 0:
        print("Жоден файл не знайдено. Перевір наявність ZIP файлів у папці з main.py")
        exit()

    print()
    stats = {}
    print("=== Статистика амплітудних коливань ===")
    for label, df in surface_data.items():
        s = compute_stats(df)
        stats[label] = s
        print(f"\n[{label}]")
        print(f"  Середнє значення:      {s['mean']} м/с²")
        print(f"  Стандартне відхилення: {s['std']} м/с²")
        print(f"  Максимум:              {s['max']} м/с²")
        print(f"  Мінімум:               {s['min']} м/с²")
        print(f"  Амплітуда (peak-peak): {s['amplitude']} м/с²")
        print(f"  RMS:                   {s['rms']} м/с²")

    if len(stats) > 1:
        best  = min(stats, key=lambda s: stats[s]['amplitude'])
        worst = max(stats, key=lambda s: stats[s]['amplitude'])
        print(f"\n=== Висновок ===")
        print(f"Найменші вібрації (мінімальна амплітуда): {best} ({stats[best]['amplitude']} м/с²)")
        print(f"Найбільші вібрації (максимальна амплітуда): {worst} ({stats[worst]['amplitude']} м/с²)")

    print("\nПобудова графіків...")
    plot_time_series(surface_data)
    plot_amplitude_comparison(stats)
    plot_distribution(surface_data)
