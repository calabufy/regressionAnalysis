import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

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
window_size_for_outliers = 5
for i in range(len(chlor_a)):
    if chlor_a[i] > critical_value:
        chlor_a[i] = sum(chlor_a[i - window_size_for_outliers // 2:i + window_size_for_outliers // 2 + 1]) / \
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

# Построение модели ARIMA
# Определение параметра d (параметр дифференциации), используем ADF-тест
# Проверка на стационарность
result = adfuller(chlor_a_pandas)
# print(result)
# result : (тестовая статстика ADF, p-value (если значение меньше уровня значимости, то ряд стационарен),
#           количество лагов, количество фактических наблюдений, использованных для теста после учета лагов,
#           критические значения для различных уровней значимости (1%, 5%, 10%), ...)
if result[1] > 0.05:  # Если временной ряд не стационарен
    chlor_a_pandas_diff = chlor_a_pandas.diff(periods=1).dropna()
    # Временной ряд необходимо привести к стационарному, тк в при нестационарном ряде его среднее значение
    # со временем может измениться, что делает моделирование менее надёжным
    result = adfuller(chlor_a_pandas_diff)
else:  # Если временной ряд стационарен
    chlor_a_pandas_diff = chlor_a_pandas_av
# print(result)

# порядки AR и MA (нахождение p и q)
plot_acf(chlor_a_pandas_diff)
plot_pacf(chlor_a_pandas_diff)

d = 1   # порядок интегрирования (I). Он равен 1, тк ряд стационарен
p = 17  # порядок авторегрессии (AR)
q = 40  # порядок скользящего среднего (MA)

model_ARIMA = ARIMA(chlor_a, order=(p, d, q)).fit()
print(model_ARIMA.summary())
residuals = model_ARIMA.resid[1:]
fig, ax = plt.subplots(1, 2)

# Построение графика остаточных значений и их плотности (проверка на белый шум)
pd.Series(residuals).plot(label="Residuals", ax=ax[0])
pd.Series(residuals).plot(label="Density", kind='kde', ax=ax[1])

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(time, chlor_a, marker=',', linestyle='-', color='gray', alpha=0.5, label='chlorophyll A')
plt.plot(time, moving_av, marker=',', linestyle='-', color='green', label='moving av (without season cycle)')
plt.plot(time, trend, color='red', linewidth=2.5, label='Trend Line')
pred = model_ARIMA.predict(start=1, end=len(chlor_a)-1)
rmse_ARIMA = np.sqrt(mean_squared_error(chlor_a[1:], pred))
rmse_trend = np.sqrt(mean_squared_error(chlor_a, trend))
print(f"RMSE for ARIMA({p}, {d}, {q}) : {rmse_ARIMA}")
print(f"RMSE for Trend line: {rmse_trend}")
plt.plot(time[1:], pred, color='blue', linestyle='--', linewidth=1, label='ARIMA Prediction')

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
