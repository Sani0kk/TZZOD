import pandas as pd

df = pd.read_csv('sales_data_sample.csv', encoding='latin-1')

print("=== Перші 5 рядків датасету ===")
print(df.head())
print(f"\nРозмір датасету: {df.shape[0]} рядків, {df.shape[1]} колонок")
print(f"\nТип колонки ORDERDATE до перетворення: {df['ORDERDATE'].dtype}")

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

print(f"Тип колонки ORDERDATE після перетворення: {df['ORDERDATE'].dtype}")
print("\nПриклад дат після перетворення:")
print(df['ORDERDATE'].head())

selected_cities = ['Paris', 'Madrid']

df_filtered = df[df['CITY'].isin(selected_cities)]

print(f"\n=== Фільтрація: міста {selected_cities} ===")
print(f"Кількість рядків після фільтрації: {len(df_filtered)}")
print(f"Унікальні міста у відфільтрованих даних: {df_filtered['CITY'].unique()}")

df_filtered = df_filtered.copy()
df_filtered['month'] = df_filtered['ORDERDATE'].dt.to_period('M')

monthly_avg = df_filtered.groupby('month')['SALES'].mean()

print("\n=== Середній SALES по місяцях ===")
print(monthly_avg.to_string())

best_month = monthly_avg.idxmax()
best_value = monthly_avg.max()

print("\n=== Результат ===")
print(f"Місяць з максимальним середнім продажем: {best_month}")
print(f"Середній SALES у цьому місяці: {best_value:.2f}")
