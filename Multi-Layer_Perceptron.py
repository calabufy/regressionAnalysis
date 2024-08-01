from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from netCDF4 import Dataset
from numpy import ma
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates


# Функция для создания признаков и целевой переменной
def create_features(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


filename = "omi_health_chl_baltic_oceancolour_area_averaged_mean_19970901_P20230807.nc"
dataset = Dataset(filename, 'r')

time = dataset.variables['time'][:]
chlor_a = dataset.variables['chlor_a'][:]
dataset.close()

# Проверка на маскированные значения
if ma.is_masked(chlor_a):
    chlor_a = chlor_a.filled(np.nan)

# Создание маски для пропущенных значений (предполагается, что они представлены как NaN)
mask = ~np.isnan(chlor_a)

# Применение маски к обоим массивам
chlor_a = chlor_a[mask]
time = time[mask]

window_size = 9
chlor_a_pandas = pd.Series(chlor_a)
windows = chlor_a_pandas.rolling(window=window_size)
moving_av = windows.mean().dropna().tolist()

# Создание признаков и целевой переменной
X, y = create_features(moving_av, window_size)

# Разделение данных на обучающую и тестовую выборки
train_size = -30
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Определение модели
model = Sequential([
    Dense(window_size, input_shape=(window_size,)),  # Входной слой с 9 нейронами
    Dense(16, activation='relu'),  # Полносвязный слой с 16 скрытыми нейронами
    Dense(1)  # Выходной слой с 1 нейроном
])
# activation - активационная функция


# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
# optimizer - способ обновления весов модели
# loss - функция потерь

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, batch_size=150)
# epohs - количество эпох обучения
# batch_size - количество примеров на одной эпохе

# Прогнозирование
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE for MLP: {rmse}")

start_date = datetime(1950, 1, 1)
time = np.array([start_date + timedelta(days=int(day)) for day in time])

plt.figure(figsize=(10, 5))
plt.plot(time, chlor_a, marker=',', linestyle='-', color='gray', alpha=0.5, label='chlorophyll A')
plt.plot(time[window_size - 1:], moving_av, marker=',', linestyle='-',
         color='green', label='moving av (without season cycle)')
plt.plot(time[train_size:], predictions, marker=',', linestyle='--', color='blue', label='Predicted Values')
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

plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
