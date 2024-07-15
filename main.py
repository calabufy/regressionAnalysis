import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")


# Пути к файлам
filename = "datasets/sharkweb_data.xlsx"
worksheet_name = "KOVIKSUDDE"

dataset = pd.read_excel(filename, sheet_name=worksheet_name)


# Извлечение переменных
time = dataset['Sampling date (start)']
chlor_a = dataset['Value']
air_temp = dataset['Air temperature (C)']

# Удаление выбросов
critical_value = 25
window_size_for_outliers = 5
for i in range(len(chlor_a)):
    if chlor_a[i] > critical_value:
        chlor_a[i] = sum(chlor_a[i - window_size_for_outliers // 2:i + window_size_for_outliers // 2 + 1]) / \
                     window_size_for_outliers

# Удаление сезонности - скользящее среднее
window_size = 12
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
# Проверка на стационарность
result = adfuller(train_chlor_a_pandas)
# print(result)
# result : (тестовая статстика ADF, p-value (если значение меньше уровня значимости, то ряд стационарен),
#           количество лагов, количество фактических наблюдений, использованных для теста после учета лагов,
#           критические значения для различных уровней значимости (1%, 5%, 10%), ...)
train_chlor_a_pandas_diff = train_chlor_a_pandas.diff(periods=1).dropna()
# Временной ряд необходимо привести к стационарному, тк в при нестационарном ряде его среднее значение
# со временем может измениться, что делает моделирование менее надёжным (проверка на стационарность: p-value <= 0.05)
result = adfuller(train_chlor_a_pandas_diff)
# print(result)

# порядки AR и MA (нахождение p и q)
# Построение графиков ACF и PACF
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plot_acf(train_chlor_a_pandas_diff, ax=axes[0])
axes[0].set_title('ACF of Differenced Series')

plot_pacf(train_chlor_a_pandas_diff, ax=axes[1])
axes[1].set_title('PACF of Differenced Series')

'''d = 1   # порядок интегрирования (I). Он равен 1, тк ряд приведён к стационарному через одну дифференциацию

best_pdq_rmse = None
best_pdq_AIC = None
min_rmse = float('inf')
min_AIC = float('inf')

for p in range(1, 11):
    for q in range(1, 11):
        print(f"ARIMA({p}, {d}, {q})")
        try:
            model_ARIMA = ARIMA(train_chlor_a, order=(p, d, q)).fit()
            pred = model_ARIMA.predict(start=len(train_chlor_a), end=len(train_chlor_a) + len(test_chlor_a) - 1)
            rmse_ARIMA = np.sqrt(mean_squared_error(test_chlor_a, pred))
            if rmse_ARIMA < min_rmse:
                best_pdq_rmse = (p, d, q)
                min_rmse = rmse_ARIMA
            if model_ARIMA.aic < min_AIC:
                best_pdq_AIC = (p, d, q)
                min_AIC = model_ARIMA.aic
        except:
            continue

print(f"RMSE: Best pdq: {best_pdq_rmse}, Min RMSE: {min_rmse}")
print(f"AIC: Best pdq: {best_pdq_AIC}, Min AIC: {min_AIC}")'''

p, d, q = 8, 1, 10  # best_pdq from RMSE and AIC

model_ARIMA = ARIMA(train_chlor_a, order=(p, d, q)).fit()
residuals = model_ARIMA.resid[1:]

correlation = np.corrcoef(chlor_a, air_temp)[0, 1]
print(f"Коэффициент корреляции Пирсона: {correlation}")  # проверка на линейную зависимость

# Значение коэффициента Пирсона ~ 0.124, что указывает на слабую линейную положительную зависимость

corr, p_value = spearmanr(chlor_a, air_temp)
print(f"Коэффициент корреляции Спирмена: {corr}")  # проверка на монотонную зависимость
print(f"p-value: {p_value}")

# Значение коэффициента Спирмена ~ 0.096, что указывает на слабую монотонную зависимость
# P-value выше уровня значимости, поэтому нельзя отвергать нулевую гипотезу, однако значения
# коэффицентов указывает на то, что при больших данных зависимость может быть случайной
        
fig, ax = plt.subplots(1, 3, figsize=(10, 5))

# Построение графика остаточных значений, их плотности и qqplot
pd.Series(residuals).plot(label="Residuals", ax=ax[0])
pd.Series(residuals).plot(label="Density", kind='kde', ax=ax[1])
qqplot(residuals, line='s', ax=ax[2])
plt.subplots_adjust(wspace=0.4)
ax[0].set_title("Residuals")
ax[1].set_title("Density")
ax[2].set_title("QQplot")

# Построение графика
plt.figure(figsize=(12, 6))
#plt.subplot(2, 1, 1)
plt.plot(time, chlor_a, marker=',', linestyle='-', color='gray', alpha=0.5, label='chlorophyll A')
plt.plot(time, moving_av, marker=',', linestyle='-', color='green', label='moving av (without season cycle)')
plt.plot(time, trend, color='red', linewidth=2.5, label='Trend Line')
train_ARIMA = model_ARIMA.predict(start=1, end=len(train_chlor_a)-1)
pred_ARIMA = model_ARIMA.predict(start=len(train_chlor_a), end=len(train_chlor_a) + len(test_chlor_a) - 1)
rmse_ARIMA = np.sqrt(mean_squared_error(test_chlor_a, pred_ARIMA))
rmse_trend = np.sqrt(mean_squared_error(chlor_a, trend))
print(f"RMSE for ARIMA({p}, {d}, {q}) : {rmse_ARIMA}")
print(f"RMSE for Trend line: {rmse_trend}")
# plt.plot(time[1:len(train_chlor_a)], train_ARIMA, color='blue', linestyle='-', linewidth=1, label='ARIMA train')
# plt.plot(time[len(train_chlor_a):], pred_ARIMA, color='blue', linestyle='--', linewidth=1, label='ARIMA test')

# Настройка графика
plt.title('Chlorophyll-a Concentration Over Time')
plt.ylabel('Chlorophyll-a Concentration (ug/l)')
plt.grid(True)
plt.legend()

# Настройка меток оси X для отображения каждого года
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Поворот меток оси X для лучшей читаемости
plt.xticks(rotation=45)

'''plt.subplot(2, 1, 2)
plt.plot(time, air_temp, label='Temperature Anomaly', color='orange')
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly')
plt.legend()
plt.grid(True)

# Настройка меток оси X для отображения каждого года
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xticks(rotation=30)'''

# Отображение графика
plt.show()
