import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd

filename = "omi_health_chl_baltic_oceancolour_area_averaged_mean_19970901_P20230807.nc"
dataset = Dataset(filename, 'r')

time = dataset.variables['time'][:]
chlor_a = dataset.variables['chlor_a'][:]
dataset.close()

# Изменение массива time
start_date = datetime(1950, 1, 1)
time = np.array([start_date + timedelta(days=int(day)) for day in time])

# Проверка на маскированные значения
if ma.is_masked(chlor_a):
    chlor_a = chlor_a.filled(np.nan)

# Создание маски для пропущенных значений (предполагается, что они представлены как NaN)
mask = ~np.isnan(chlor_a)

# Применение маски к обоим массивам
chlor_a = chlor_a[mask]
time = time[mask]

# Удаление выбросов
critical_value = 2.65
window_size_for_outliers = 3
for i in range(len(chlor_a)):
    if chlor_a[i] > critical_value:
        chlor_a[i] = sum(chlor_a[i - window_size_for_outliers//2:i + window_size_for_outliers//2 + 1]) / \
                     window_size_for_outliers

# Удаление сезонности - скользящее среднее
window_size = 9
chlor_a_pandas = pd.Series(chlor_a)
windows = chlor_a_pandas.rolling(window=window_size)
chlor_a_pandas_av = windows.mean()
moving_av = chlor_a_pandas_av.tolist()

# Построение линии Тренда
time_num = mdates.date2num(time)
X = np.vstack([time_num, np.ones(len(time_num))]).T
beta = np.linalg.inv(X.T @ X) @ (X.T @ chlor_a)  # X.T @ X @ beta = X.T @ y
trend = beta[0] * time_num + beta[1]  # y = k * x + b

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(time, chlor_a, marker=',', linestyle='-', color='gray', alpha=0.5, label='chlorophyll A')
plt.plot(time, moving_av, marker=',', linestyle='-', color='green', label='moving av (without season cycle)')
plt.plot(time, trend, color='red', linewidth=2.5, label='Trend Line')

# Настройка графика
plt.title('Chlorophyll-a Concentration Over Time')
plt.ylabel('Chlorophyll-a Concentration')
plt.grid(True)
plt.legend()

# Настройка меток оси X для отображения каждого года
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Поворот меток оси X для лучшей читаемости
plt.xticks(rotation=45)

# Отображение графика
plt.show()
