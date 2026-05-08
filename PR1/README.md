# Практична робота №1-2
## Тема: Перетворення дат, фільтрація та групування даних

**Виконав:** Голдовський Олександр Васильович  
**Варіант:** 3  
**Дата виконання:** 09.05.2026

---

## Завдання

- Перетворити колонку `order_date` у формат `datetime`
- Відфільтрувати дані тільки по 2 вибраних містах (варіант 3: **Paris** та **Madrid**)
- Згрупувати дані по місяцях
- Знайти місяць з максимальним середнім значенням `total`

---

## Датасет

**Назва:** Sample Sales Data  
**Джерело:** [Kaggle — kyanyoga/sample-sales-data](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)  
**Файл:** `sales_data_sample.csv`  
**Опис:** Датасет містить дані про продажі компанії: номер замовлення, кількість товару, ціну, місто, країну, дату замовлення та суму продажу.

**Основні колонки:**

| Колонка | Тип | Опис |
|---|---|---|
| `ORDERDATE` | object → datetime | Дата замовлення |
| `CITY` | string | Місто клієнта |
| `SALES` | float | Сума продажу |

---

## Використані бібліотеки

```python
import pandas as pd
```

---

## Опис роботи програми

### Крок 1 — Завантаження датасету

Завантажуємо CSV файл за допомогою `pandas`. Вказуємо кодування `latin-1`, оскільки датасет містить символи, несумісні з UTF-8.

```python
df = pd.read_csv('sales_data_sample.csv', encoding='latin-1')
```

<img width="1483" height="290" alt="image" src="https://github.com/user-attachments/assets/ae17262c-1aca-407a-879a-ef793960bcbe" />

---

### Крок 2 — Перетворення `ORDERDATE` у datetime

До перетворення колонка має тип `object` (рядок). За допомогою `pd.to_datetime()` перетворюємо її у формат `datetime64`.

```python
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
```

<img width="546" height="239" alt="image" src="https://github.com/user-attachments/assets/c7494793-bb90-4578-ba1c-7596ff36ee51" />

---

### Крок 3 — Фільтрація по містах Paris та Madrid

Для варіанту 3 обрано міста **Paris** та **Madrid**. Використовуємо метод `.isin()` для відбору рядків.

```python
selected_cities = ['Paris', 'Madrid']
df_filtered = df[df['CITY'].isin(selected_cities)]
```

<img width="521" height="122" alt="image" src="https://github.com/user-attachments/assets/2174dfea-b4dc-4fa3-a314-418e8eaf15bd" />

---

### Крок 4 — Групування по місяцях

Створюємо нову колонку `month` з типом Period, потім групуємо і рахуємо середнє значення `SALES` для кожного місяця.

```python
df_filtered['month'] = df_filtered['ORDERDATE'].dt.to_period('M')
monthly_avg = df_filtered.groupby('month')['SALES'].mean()
```

<img width="339" height="551" alt="image" src="https://github.com/user-attachments/assets/c505f402-badb-4634-bb45-72a05d636058" />

---

### Крок 5 — Пошук місяця з максимальним середнім продажем

```python
best_month = monthly_avg.idxmax()
best_value = monthly_avg.max()
```

<img width="456" height="83" alt="image" src="https://github.com/user-attachments/assets/5ffac3f7-9263-436d-b166-d1b391ffc2e4" />

---

## Результат

```
=== Перші 5 рядків датасету ===
   ORDERNUMBER  QUANTITYORDERED  PRICEEACH  ORDERLINENUMBER    SALES        ORDERDATE  ... POSTALCODE  COUNTRY  TERRITORY  CONTACTLASTNAME CONTACTFIRSTNAME  DEALSIZE
0        10107               30      95.70                2  2871.00   2/24/2003 0:00  ...      10022      USA        NaN               Yu             Kwai     Small
1        10121               34      81.35                5  2765.90    5/7/2003 0:00  ...      51100   France       EMEA          Henriot             Paul     Small
2        10134               41      94.74                2  3884.34    7/1/2003 0:00  ...      75508   France       EMEA         Da Cunha           Daniel    Medium
3        10145               45      83.26                6  3746.70   8/25/2003 0:00  ...      90003      USA        NaN            Young            Julie    Medium
4        10159               49     100.00               14  5205.27  10/10/2003 0:00  ...        NaN      USA        NaN            Brown            Julie    Medium

[5 rows x 25 columns]

Розмір датасету: 2823 рядків, 25 колонок

Тип колонки ORDERDATE до перетворення: str
Тип колонки ORDERDATE після перетворення: datetime64[us]

Приклад дат після перетворення:
0   2003-02-24
1   2003-05-07
2   2003-07-01
3   2003-08-25
4   2003-10-10
Name: ORDERDATE, dtype: datetime64[us]

=== Фільтрація: міста ['Paris', 'Madrid'] ===
Кількість рядків після фільтрації: 374
Унікальні міста у відфільтрованих даних: <StringArray>
['Paris', 'Madrid']
Length: 2, dtype: str

=== Середній SALES по місяцях ===
month
2003-01    3432.458462
2003-04    3821.741000
2003-05    3592.541765
2003-06    3301.318333
2003-07    3660.697143
2003-09    3671.370769
2003-10    2887.240000
2003-11    3019.860667
2003-12    3843.808750
2004-01    3719.297105
2004-03    7665.350000
2004-04    3147.515556
2004-05    3462.448000
2004-06    3125.665625
2004-10    3170.193636
2004-11    4567.963500
2004-12    3324.253600
2005-02    3204.527073
2005-03    3327.279655
2005-04    4688.933333
2005-05    5184.477391
Freq: M

=== Результат ===
Місяць з максимальним середнім продажем: 2004-03
Середній SALES у цьому місяці: 7665.35
```

---

## Як запустити

1. Завантажити датасет з [Kaggle](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data) — файл `sales_data_sample.csv`
2. Покласти файл `sales_data_sample.csv` у ту саму папку, що і `main.py`
3. Встановити бібліотеку: `pip install pandas`
4. Запустити: `python main.py`

