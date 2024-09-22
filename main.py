import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

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
# Разделение выборки на 2 части: обучающая и тестовая
train_chlor_a, test_chlor_a = chlor_a[:-30], chlor_a[-30:]
train_chlor_a_pandas = pd.Series(train_chlor_a)

# Определение параметра d (параметр дифференциации), используем ADF-тест
result = adfuller(train_chlor_a_pandas)
train_chlor_a_pandas_diff = train_chlor_a_pandas.diff(periods=1).dropna()
result = adfuller(train_chlor_a_pandas_diff)

# порядки AR и MA (нахождение p и q)
plot_acf(train_chlor_a_pandas_diff)
plot_pacf(train_chlor_a_pandas_diff)

# Параметры модели ARIMA (на основе AIC и RMSE)
p, d, q = 8, 1, 10

model_ARIMA = ARIMA(train_chlor_a, order=(p, d, q)).fit()
residuals = model_ARIMA.resid[1:]

fig, ax = plt.subplots(1, 3, figsize=(10, 5))

# Построение графика остаточных значений, их плотности и qqplot
pd.Series(residuals).plot(label="Residuals", ax=ax[0])
pd.Series(residuals).plot(label="Density", kind='kde', ax=ax[1])
qqplot(residuals, line='s', ax=ax[2])
plt.subplots_adjust(wspace=0.4)
ax[0].set_title("Residuals")
ax[1].set_title("Density")
ax[2].set_title("QQplot")

# Количество месяцев для прогноза на 5 лет (60 месяцев)
forecast_months = 5 * 12

# Создание временных меток для прогноза на 5 лет
future_time = pd.date_range(start=time[-1], periods=forecast_months + 1, freq='M')[1:]

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(time, chlor_a, marker=',', linestyle='-', color='gray', alpha=0.5, label='chlorophyll A')
# plt.plot(time, moving_av, marker=',', linestyle='-', color='green', label='moving av (without season cycle)')
plt.plot(time, trend, color='red', linewidth=2.5, label='Trend Line')
train_ARIMA = model_ARIMA.predict(start=1, end=len(train_chlor_a) - 1)
pred_ARIMA = model_ARIMA.predict(start=len(train_chlor_a), end=len(train_chlor_a) + len(test_chlor_a) - 1)
forecast_ARIMA = model_ARIMA.predict(start=len(chlor_a), end=len(chlor_a) + forecast_months - 1)
plt.plot(time[1:len(train_chlor_a)], train_ARIMA, color='blue', linestyle='-', linewidth=1, label='ARIMA train')
plt.plot(time[len(train_chlor_a):], pred_ARIMA, color='blue', linestyle='--', linewidth=1, label='ARIMA test')
plt.plot(future_time, forecast_ARIMA, color='orange', linestyle='--', linewidth=1, label='ARIMA forecast')

# Настройка графика
plt.title('Chlorophyll-a Concentration Over Time and ARIMA-model')
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
