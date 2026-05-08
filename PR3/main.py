import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from functools import partial


def compute_district_stats(district_data):
    district_name, group = district_data
    mean_consumption = group['MW'].mean()
    max_consumption = group['MW'].max()
    min_consumption = group['MW'].min()
    std_consumption = group['MW'].std()
    efficiency_score = mean_consumption / max_consumption
    return {
        'district': district_name,
        'mean_MW': round(mean_consumption, 2),
        'max_MW': round(max_consumption, 2),
        'min_MW': round(min_consumption, 2),
        'std_MW': round(std_consumption, 2),
        'efficiency_score': round(efficiency_score, 4)
    }


def sequential_processing(district_groups):
    start = time.time()
    results = [compute_district_stats(item) for item in district_groups]
    elapsed = time.time() - start
    return results, elapsed


def parallel_processing(district_groups, num_processes):
    start = time.time()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(compute_district_stats, district_groups)
    elapsed = time.time() - start
    return results, elapsed


if __name__ == '__main__':
    files = {
        'AEP': 'AEP_hourly.csv',
        'DAYTON': 'DAYTON_hourly.csv',
        'DEOK': 'DEOK_hourly.csv',
        'DOM': 'DOM_hourly.csv',
        'DUQ': 'DUQ_hourly.csv',
        'EKPC': 'EKPC_hourly.csv',
        'FE': 'FE_hourly.csv',
        'NI': 'NI_hourly.csv',
        'PJME': 'PJME_hourly.csv',
        'PJMW': 'PJMW_hourly.csv',
    }

    dfs = []
    for district, filename in files.items():
        try:
            df = pd.read_csv(filename)
            df.columns = ['Datetime', 'MW']
            df['district'] = district
            dfs.append(df)
        except FileNotFoundError:
            pass

    data = pd.concat(dfs, ignore_index=True)
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    print(f"Завантажено рядків: {len(data)}")
    print(f"Унікальних районів: {data['district'].nunique()}")
    print(f"Районів: {list(data['district'].unique())}")
    print(f"Часовий діапазон: {data['Datetime'].min()} — {data['Datetime'].max()}")
    print()

    district_groups = list(data.groupby('district'))

    num_cores = mp.cpu_count()
    print(f"Доступно ядер CPU: {num_cores}")
    print()

    seq_results, seq_time = sequential_processing(district_groups)
    print(f"Послідовна обробка завершена за: {seq_time:.4f} секунд")

    par_results, par_time = parallel_processing(district_groups, num_cores)
    print(f"Паралельна обробка завершена за:  {par_time:.4f} секунд")
    print()

    speedup = seq_time / par_time
    efficiency = speedup / num_cores * 100
    print(f"Прискорення (Speedup):   {speedup:.2f}x")
    print(f"Ефективність паралелізму: {efficiency:.1f}%")
    print()

    results_df = pd.DataFrame(par_results).sort_values('efficiency_score', ascending=False)
    print("=== Оцінка ефективності використання енергії по районах ===")
    print(results_df.to_string(index=False))
    print()

    best = results_df.iloc[0]
    worst = results_df.iloc[-1]
    print(f"Найефективніший район:    {best['district']} (score={best['efficiency_score']})")
    print(f"Найменш ефективний район: {worst['district']} (score={worst['efficiency_score']})")
