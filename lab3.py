import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
data = pd.read_csv('winequalityN.csv')

# Преобразование категориального признака (тип вина) в числовой формат
data['type'] = data['type'].map({'red': 0, 'white': 1})

# Обработка пропущенных значений (замена на медианные значения)
data = data.fillna(data.median())

# Разделение данных на три набора: красное вино, белое вино, общие данные
red_wine = data[data['type'] == 0]
white_wine = data[data['type'] == 1]
all_wine = data

# Функция для нормализации данных
def normalize(df):
    scaler = MinMaxScaler()
    df_norm = scaler.fit_transform(df)
    return pd.DataFrame(df_norm, columns=df.columns)

# Функция для выполнения эксперимента
def run_experiment(data, n_splits=10):
    results = []
    for i in range(n_splits):
        # Разбиение на обучающую и тестовую выборки (70:30)
        X = data.drop('quality', axis=1)  # Признаки
        y = data['quality']  # Целевая переменная
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        # Нормализация данных
        X_train_norm = normalize(X_train)
        X_test_norm = normalize(X_test)
        
        # Построение модели LASSO
        model = Lasso(alpha=0.1, random_state=42)
        model.fit(X_train_norm, y_train)
        
        # Прогнозирование на тестовой выборке
        y_pred = model.predict(X_test_norm)
        
        # Оценка качества (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        results.append(mae)
        print(f"Разбиение {i+1}: MAE = {mae:.4f}")
    
    return results

# Выполнение эксперимента для каждого набора данных
print("Красное вино:")
red_results = run_experiment(red_wine)

print("\nБелое вино:")
white_results = run_experiment(white_wine)

print("\nОбщие данные:")
all_results = run_experiment(all_wine)

# Вывод результатов
print("\nРезультаты тестирования:")
print(f"Красное вино: Средний MAE = {np.mean(red_results):.4f}")
print(f"Белое вино: Средний MAE = {np.mean(white_results):.4f}")
print(f"Общие данные: Средний MAE = {np.mean(all_results):.4f}")