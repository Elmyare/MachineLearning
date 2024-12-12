import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv('data1.csv')
print(data)
X = data.iloc[:, :-1].values  # Признаки
y = data.iloc[:, -1].values   # Классы

# Функция ядра (квадратичное)
def kernel(z):
    return (1 - z**2)**2 if abs(z) <= 1 else 0

# Самостоятельная реализация метода парзеновского окна
def parzen_window(X_train, y_train, X_test, h):
    predictions = []
    for x_test in X_test:
        # Вычисление расстояний
        distances = np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))  # Евклидово расстояние
        weights = np.array([kernel(d / h) for d in distances])  # Применение ядра
        
        # Взвешенное голосование
        class_votes = {}
        for cls in np.unique(y_train):
            class_votes[cls] = np.sum(weights[y_train == cls])  # Сумма весов для каждого класса
        
        # Выбор класса с максимальной суммой весов
        predicted_class = max(class_votes, key=class_votes.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

# Функция для разбиения и тестирования
def run_experiment(X, y, n_splits=10, test_size=1/3, h=5000.0):
    results = []
    for i in range(n_splits):
        # Разбиение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        # print(X_train)
        # print(X_test)
        # print(y_train)
        # print(y_test)
        # Классификация
        y_pred = parzen_window(X_train, y_train, X_test, h)
        
        # Оценка точности
        accuracy = np.mean(y_test == y_pred)  # Ручная оценка точности
        results.append(accuracy)
    
    return results

# Параметры
n_splits = 10
h = 5000.0  # Фиксированное значение h

# Выполнение эксперимента
results = run_experiment(X, y, n_splits, h=h)

# Вывод результатов
print("Результаты тестирования:")
for i, acc in enumerate(results):
    print(f"Разбиение {i+1}: Точность = {acc:.4f}")

# Средняя точность
print(f"Средняя точность: {np.mean(results):.4f}")