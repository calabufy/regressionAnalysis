import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy.ma as ma
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error


class Perceptron:
    def __init__(self, input_size, learning_rate=0.001):
        self.weights = np.zeros(input_size + 1)  # Включаем смещение
        self.learning_rate = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return z

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error


def create_features(data, window_size):
    """Функция создания признаков и целевой переменной"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])  # X - набор признаков от data[i] длинной window_size
        y.append(data[i + window_size])  # y - значение, которое необходимо предсказать по набору признаков X
    return np.array(X), np.array(y)


if __name__ == "__main__":
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
    X, y = create_features(chlor_a, window_size)

    # Разделение данных на обучающую и тестовую выборки
    train_size = -30
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    perceptron = Perceptron(input_size=window_size)
    perceptron.fit(X_train, y_train, epochs=100)

    pred = np.array([perceptron.predict(x) for x in X_test])
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"RMSE for Perceptron: {rmse}")

    start_date = datetime(1950, 1, 1)
    time = np.array([start_date + timedelta(days=int(day)) for day in time])

    plt.figure(figsize=(10, 5))
    plt.plot(time, chlor_a, marker=',', linestyle='-', color='gray', alpha=0.5, label='chlorophyll A')
    plt.plot(time[window_size - 1:], moving_av, marker=',', linestyle='-', color='green',
             label='moving av (without season cycle)')
    plt.plot(time[train_size:], pred, marker=',', linestyle='--', color='blue', label='Predicted Values')
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
