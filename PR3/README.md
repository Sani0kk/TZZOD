# Практична робота №2
## Тема: Паралельний алгоритм оцінки ефективності використання енергії

**Виконав:** Голдовський Олександр Васильович  
**Варіант:** 3  
**Дата виконання:** 09.05.2026

---

## Завдання

Реалізувати паралельний алгоритм для оцінки ефективності використання енергії в різних районах за допомогою `multiprocessing`. Оцінити ефективність паралельних обчислень для великих наборів даних з енергоспоживанням.

---

## Датасет

**Назва:** PJM Hourly Energy Consumption  
**Джерело:** [Kaggle — robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)  
**Обсяг:** понад 145 000 рядків погодинних вимірювань  
**Часовий діапазон:** 2002 — 2018 роки  
**Опис:** Реальні дані про погодинне енергоспоживання (у мегаватах) від PJM Interconnection LLC — регіонального оператора електромережі США. Датасет містить окремі CSV-файли для кожного з 10 енергетичних районів (AEP, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJME, PJMW).

**Основні колонки:**

| Колонка | Тип | Опис |
|---|---|---|
| `Datetime` | string → datetime | Дата та час вимірювання |
| `MW` | float | Енергоспоживання у мегаватах |
| `district` | string | Назва енергетичного району |

---

## Використані бібліотеки

```python
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
```

---

## Опис роботи програми

### Крок 1 — Завантаження та об'єднання даних

Програма завантажує 10 окремих CSV-файлів, кожен відповідає одному енергетичному району. До кожного датафрейму додається колонка `district` з назвою району. Усі файли об'єднуються в один великий датафрейм за допомогою `pd.concat`.

```python
df.columns = ['Datetime', 'MW']
df['district'] = district
data = pd.concat(dfs, ignore_index=True)
```

<img width="769" height="89" alt="image" src="https://github.com/user-attachments/assets/8a88ba6e-2d9f-4f81-af08-ac6300157856" />

---

### Крок 2 — Функція обчислення статистики по району

Функція `compute_district_stats` приймає пару `(назва_району, датафрейм)` і повертає словник з такими показниками:

- `mean_MW` — середнє споживання
- `max_MW` — максимальне споживання
- `min_MW` — мінімальне споживання
- `std_MW` — стандартне відхилення
- `efficiency_score` — коефіцієнт ефективності (відношення середнього до максимального споживання; чим ближче до 1, тим стабільніше і ефективніше використовується енергія)

```python
efficiency_score = mean_consumption / max_consumption
```

---

### Крок 3 — Послідовна обробка

Дані кожного району обробляються по черзі в одному потоці. Час виконання вимірюється за допомогою `time.time()`.

```python
results = [compute_district_stats(item) for item in district_groups]
```

<img width="443" height="59" alt="image" src="https://github.com/user-attachments/assets/279debcf-07d7-403c-92bf-673c28f3f20c" />

---

### Крок 4 — Паралельна обробка

Дані всіх районів обробляються одночасно у кількох процесах за допомогою `mp.Pool`. Кількість процесів дорівнює кількості доступних ядер CPU.

```python
with mp.Pool(processes=num_cores) as pool:
    results = pool.map(compute_district_stats, district_groups)
```

<img width="444" height="111" alt="image" src="https://github.com/user-attachments/assets/9d832ff5-ea38-46bb-a5de-806f121c56a0" />

---

### Крок 5 — Порівняння ефективності

Розраховується прискорення (Speedup) та ефективність паралелізму:

- **Speedup** = час послідовного виконання / час паралельного виконання
- **Ефективність** = Speedup / кількість ядер × 100%

```python
speedup = seq_time / par_time
efficiency = speedup / num_cores * 100
```

<img width="292" height="62" alt="image" src="https://github.com/user-attachments/assets/b9d2a729-1202-44f3-baea-3ecefbfe7ade" />

---

### Крок 6 — Результати оцінки ефективності по районах

Підсумкова таблиця з усіма показниками по кожному району, відсортована за коефіцієнтом ефективності від найвищого до найнижчого.

<img width="574" height="342" alt="image" src="https://github.com/user-attachments/assets/03d2e95c-be67-4c38-bf0f-06f909e7c15f" />

---

## Результат

<img width="772" height="622" alt="image" src="https://github.com/user-attachments/assets/d434f33a-6c83-4397-aefd-ad08b71d05ca" />

---

## Висновок

Паралельна обробка за допомогою `multiprocessing.Pool` дозволила розподілити обчислення між усіма ядрами процесора. На великому датасеті (145 000+ рядків, 10 районів) паралельний підхід показав прискорення порівняно з послідовним виконанням. Коефіцієнт ефективності `efficiency_score` відображає стабільність енергоспоживання у кожному районі — чим він вищий, тим рівномірніше використовується енергія без різких піків.

---

## Як запустити

1. Завантажити датасет з [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) — завантажити всі CSV-файли
2. Покласти всі CSV-файли у ту саму папку, що і `main.py`
3. Встановити бібліотеки: `pip install pandas numpy`
4. Запустити: `python main.py`
