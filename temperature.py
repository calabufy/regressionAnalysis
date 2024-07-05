import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from numpy import ma
import matplotlib.dates as mdates

# Пути к файлам
filename_chlor_A = "datasets/omi_health_chl_baltic_oceancolour_area_averaged_mean_19970901_P20230807.nc"
filename_temperature_anomaly = "datasets/omi_climate_sst_bal_area_averaged_anomalies_2022_P20230509_R19932014.nc"

# Загрузка данных из NetCDF файлов
dataset1 = Dataset(filename_chlor_A, 'r')
dataset2 = Dataset(filename_temperature_anomaly, 'r')

# Извлечение переменных
time1 = dataset1.variables['time'][:]
chlor_a = dataset1.variables['chlor_a'][:]
time2 = dataset2.variables['time'][:]
temp_anomaly = dataset2.variables['sst_anomaly'][:]

# Закрытие файлов
dataset1.close()
dataset2.close()

# Преобразование времени
start_date = datetime(1950, 1, 1)
time1 = np.array([start_date + timedelta(days=int(day)) for day in time1])
time2 = np.array([start_date + timedelta(days=int(day)) for day in time2])

# Оставить только год и месяц
time1 = np.array([datetime(t.year, t.month, 1) for t in time1])
time2 = np.array([datetime(t.year, t.month, 1) for t in time2])

# Создание словарей для данных
chlor_dict = {time: value for time, value in zip(time1, chlor_a)}
temp_dict = {time: value for time, value in zip(time2, temp_anomaly)}

# Проверка на маскированные значения
if ma.is_masked(chlor_a):
    chlor_a = chlor_a.filled(np.nan)

# Создание маски для пропущенных значений (предполагается, что они представлены как NaN)
mask = ~np.isnan(chlor_a)

# Применение маски к обоим массивам
chlor_a = chlor_a[mask]
time1 = time1[mask]

# Найти общие даты
common_dates = sorted(set(time1).intersection(set(time2)))

# Создание списков для общих данных
filtered_chlor_a = [chlor_dict[date] for date in common_dates]
filtered_temp_anomaly = [temp_dict[date] for date in common_dates]

# Преобразование общих данных в numpy массивы
filtered_chlor_a = np.array(filtered_chlor_a)
filtered_temp_anomaly = np.array(filtered_temp_anomaly)

# Построение графиков
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(common_dates, filtered_chlor_a, label='Chlorophyll-a')
plt.xlabel('Date')
plt.ylabel('Chlorophyll-a')
plt.legend()
plt.grid(True)

# Настройка меток оси X для отображения каждого года
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.subplot(2, 1, 2)
plt.plot(common_dates, filtered_temp_anomaly, label='Temperature Anomaly', color='orange')
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly')
plt.legend()
plt.grid(True)

# Настройка меток оси X для отображения каждого года
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()
