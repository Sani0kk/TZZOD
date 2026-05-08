# Практична робота №5
## Тема: Нормалізація та стандартизація даних

**Виконав:** Голдовський Олександр Васильович  
**Варіант:** 3  
**Дата виконання:** 09.05.2026

---

## Завдання

Створити функції для нормалізації та стандартизації числових стовпців даних за допомогою scikit-learn.

---

## Датасет

**Назва:** New York City Taxi Fare Prediction  
**Джерело:** [Kaggle — new-york-city-taxi-fare-prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data)  
**Файл:** `train.csv`  
**Обсяг:** понад 55 мільйонів рядків (завантажується 500 000)  
**Опис:** Реальні дані про поїздки жовтих таксі Нью-Йорка. Містить числові колонки різних масштабів — вартість поїздки в доларах, координати GPS, кількість пасажирів — що робить датасет ідеальним для демонстрації методів масштабування.

**Числові колонки датасету:**

| Колонка | Одиниця | Типовий діапазон |
|---|---|---|
| `fare_amount` | USD | 2.5 — 200 |
| `pickup_longitude` | градуси | -74.05 — -73.75 |
| `pickup_latitude` | градуси | 40.63 — 40.85 |
| `dropoff_longitude` | градуси | -74.05 — -73.75 |
| `dropoff_latitude` | градуси | 40.63 — 40.85 |
| `passenger_count` | осіб | 1 — 6 |

---

## Використані бібліотеки

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
```

Встановлення:
```
pip install pandas numpy matplotlib scikit-learn
```

---

## Методи масштабування

### Min-Max нормалізація

Перетворює значення до діапазону [0, 1] за формулою:

```
x_scaled = (x - x_min) / (x_max - x_min)
```

Реалізація: `sklearn.preprocessing.MinMaxScaler`  
Застосування: нейронні мережі, алгоритми на основі відстані (KNN, SVM).

### Z-score стандартизація

Перетворює дані так, що середнє значення (μ) = 0, стандартне відхилення (σ) = 1:

```
x_scaled = (x - μ) / σ
```

Реалізація: `sklearn.preprocessing.StandardScaler`  
Застосування: лінійна регресія, логістична регресія, PCA.

### Robust Scaler

Масштабує на основі медіани та міжквартильного розмаху (IQR), стійкий до викидів:

```
x_scaled = (x - median) / IQR
```

Реалізація: `sklearn.preprocessing.RobustScaler`  
Застосування: дані з аномаліями та викидами.

---

## Опис роботи програми

### Крок 1 — Завантаження даних

Завантажується 500 000 рядків з файлу `train.csv`. Завдяки параметру `nrows` уникаємо завантаження всіх 55 млн рядків у пам'ять.

```python
df_raw = pd.read_csv(filepath, nrows=500000)
```

<img width="1359" height="132" alt="image" src="https://github.com/user-attachments/assets/303daf1e-7f11-4ed8-87e9-4ed966bda268" />

---

### Крок 2 — Очищення даних

Видаляються рядки з некоректними значеннями: від'ємна вартість, кількість пасажирів поза межами 1-6, координати за межами Нью-Йорка.

```python
df_clean = df_clean[
    (df_clean['fare_amount'] > 0) &
    (df_clean['passenger_count'] > 0) & ...
]
```

<img width="287" height="46" alt="image" src="https://github.com/user-attachments/assets/3553425c-a35b-499a-8978-6f8ac4d34584" />

---

### Крок 3 — Статистика оригінальних даних

Виводиться таблиця `describe()` для оригінальних числових колонок — видно що масштаби колонок дуже різні (fare: десятки, координати: десяткові дроби від -74 до 40).

<img width="995" height="233" alt="image" src="https://github.com/user-attachments/assets/7e424b1a-16a4-4b87-9efd-97b8a1681629" />

---

### Крок 4 — Застосування масштабування

Три функції застосовують відповідні scaler-и з scikit-learn і повертають нормалізовані датафрейми разом з навченим об'єктом scaler.

```python
df_minmax, scaler_mm  = normalize_minmax(df_clean[target_cols])
df_zscore, scaler_std = standardize_zscore(df_clean[target_cols])
df_robust, scaler_rob = scale_robust(df_clean[target_cols])
```

<img width="1004" height="722" alt="image" src="https://github.com/user-attachments/assets/975237b0-6725-4b77-8b35-c4542f72aab0" />

---

### Крок 5 — Параметри масштабування

Виводяться збережені параметри кожного scaler — мінімум/максимум для Min-Max, середнє/std для Z-score.

<img width="1614" height="342" alt="image" src="https://github.com/user-attachments/assets/99b1ed4f-01d6-4b5a-a809-0b6029fccf7c" />

---

### Крок 6 — Візуалізація

Гістограми розподілу трьох колонок (`fare_amount`, `pickup_longitude`, `pickup_latitude`) до і після кожного з трьох методів масштабування. Форма розподілу не змінюється — змінюється лише діапазон значень на осі X.

<img width="2685" height="1329" alt="image" src="https://github.com/user-attachments/assets/733aaf0e-6f0a-451c-83f9-5f7457de5acb" />

---

## Результат

```
Завантаження даних (500 000 рядків)...
Розмір датасету: 500000 рядків, 8 колонок
Колонки: ['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

Числові колонки: ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

Після очищення: 488303 рядків

--- Оригінальні дані ---
       fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count
count  488303.0000       488303.0000      488303.0000        488303.0000       488303.0000      488303.0000
mean       11.3556          -73.9751          40.7510           -73.9099           40.7182           1.6901
std         9.8503            0.0393           0.0301             3.3382            1.5027           1.3063
min         0.0100          -74.9681          40.0527         -1329.6213           -2.2697           1.0000
25%         6.0000          -73.9923          40.7365           -73.9916           40.7355           1.0000
50%         8.5000          -73.9821          40.7534           -73.9806           40.7538           1.0000
75%        12.5000          -73.9683          40.7675           -73.9652           40.7684           2.0000
max       495.0000          -72.7029          41.8003             0.0000          404.6167           6.0000

--- Min-Max нормалізація [0, 1] ---
       fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count
count  488303.0000       488303.0000      488303.0000        488303.0000       488303.0000      488303.0000
mean        0.0229            0.4384           0.3996             0.9444            0.1057           0.1380
std         0.0199            0.0173           0.0172             0.0025            0.0037           0.2613
min         0.0000            0.0000           0.0000             0.0000            0.0000           0.0000
25%         0.0121            0.4308           0.3913             0.9444            0.1057           0.0000
50%         0.0172            0.4353           0.4009             0.9444            0.1057           0.0000
75%         0.0252            0.4414           0.4090             0.9444            0.1058           0.2000
max         1.0000            1.0000           1.0000             1.0000            1.0000           1.0000

--- Z-score стандартизація (μ=0, σ=1) ---
       fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count
count  488303.0000       488303.0000      488303.0000        488303.0000       488303.0000      488303.0000
mean       -0.0000            0.0000          -0.0000            -0.0000           -0.0000          -0.0000
std         1.0000            1.0000           1.0000             1.0000            1.0000           1.0000
min        -1.1518          -25.2903         -23.1876          -376.1666          -28.6069          -0.5283
25%        -0.5437           -0.4359          -0.4804            -0.0245            0.0115          -0.5283
50%        -0.2899           -0.1765           0.0787            -0.0212            0.0237          -0.5283
75%         0.1162            0.1738           0.5468            -0.0166            0.0334           0.2372
max        49.0995           32.4033          34.8412            22.1408          242.1608           3.2994

--- Robust Scaler (медіана=0) ---
       fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count
count  488303.0000       488303.0000      488303.0000        488303.0000       488303.0000      488303.0000
mean        0.4393            0.2894          -0.0766             2.6781           -1.0830           0.6901
std         1.5154            1.6402           0.9735           126.5374           45.7100           1.3063
min        -1.3062          -41.1907         -22.6497        -47596.4054        -1308.7012           0.0000
25%        -0.3846           -0.4255          -0.5443            -0.4169           -0.5572           0.0000
50%         0.0000            0.0000           0.0000             0.0000            0.0000           0.0000
75%         0.6154            0.5745           0.4557             0.5831            0.4428           1.0000
max        74.8462           53.4361          33.8411          2804.3122        11068.0705           5.0000

=== Параметри масштабування ===

Min-Max — мінімальні значення за колонками:
{'fare_amount': np.float64(0.01), 'pickup_longitude': np.float64(-74.9681), 'pickup_latitude': np.float64(40.0527), 'dropoff_longitude': np.float64(-1329.6213), 'dropoff_latitude': np.float64(-2.2697), 'passenger_count': np.float64(1.0)}
Min-Max — максимальні значення за колонками:
{'fare_amount': np.float64(495.0), 'pickup_longitude': np.float64(-72.7029), 'pickup_latitude': np.float64(41.8003), 'dropoff_longitude': np.float64(0.0), 'dropoff_latitude': np.float64(404.6167), 'passenger_count': np.float64(6.0)}

Z-score — середні значення (mean):
{'fare_amount': np.float64(11.3556), 'pickup_longitude': np.float64(-73.9751), 'pickup_latitude': np.float64(40.751), 'dropoff_longitude': np.float64(-73.9099), 'dropoff_latitude': np.float64(40.7182), 'passenger_count': np.float64(1.6901)}
Z-score — стандартні відхилення (std):
{'fare_amount': np.float64(9.8503), 'pickup_longitude': np.float64(0.0393), 'pickup_latitude': np.float64(0.0301), 'dropoff_longitude': np.float64(3.3382), 'dropoff_latitude': np.float64(1.5027), 'passenger_count': np.float64(1.3063)}

Побудова графіку...

Графік збережено: scaling_comparison.png
```

---

## Висновок

Три методи масштабування по-різному перетворюють числові дані. Min-Max нормалізація зводить всі значення до [0, 1], що зручно для нейронних мереж, але чутливе до викидів. Z-score стандартизація центрує дані навколо нуля зі стандартним відхиленням 1 — добре підходить для лінійних моделей. Robust Scaler ігнорує викиди, оскільки використовує медіану замість середнього. На датасеті NYC Taxi з різнорідними колонками (долари, координати GPS, кількість пасажирів) масштабування є обов'язковим кроком перед навчанням будь-якої моделі машинного навчання.

---

## Як запустити

1. Зареєструватись на Kaggle та прийняти умови змагання: [NYC Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data)
2. Завантажити файл `train.csv` (5.5 GB)
3. Покласти `train.csv` у ту саму папку, що і `main.py`
4. Встановити залежності: `pip install pandas numpy matplotlib scikit-learn`
5. Запустити: `python main.py`
