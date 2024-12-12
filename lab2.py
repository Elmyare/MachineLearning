import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('heart_data.csv')

# Обработка пропущенных значений (замена '?' на медианные значения)
for col in data.columns:
    if data[col].dtype == 'object':  # Если столбец содержит строки
        data[col] = data[col].replace('?', np.nan)
        data[col] = data[col].fillna(data[col].mode()[0])  # Замена на наиболее частое значение
    else:
        data[col] = data[col].replace('?', np.nan)
        data[col] = data[col].fillna(data[col].median())  # Замена на медиану

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop('goal', axis=1)  # Признаки
y = data['goal']  # Целевая переменная

# Функция для выполнения эксперимента
def run_experiment(X, y, n_splits=10):
    results = []
    for i in range(n_splits):
        # Разбиение на обучающую и тестовую выборки (70:30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        # Построение решающего дерева
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        # Прогнозирование на тестовой выборке
        y_pred = clf.predict(X_test)
        
        # Оценка точности
        accuracy = accuracy_score(y_test, y_pred)
        results.append(accuracy)
        print(f"Разбиение {i+1}: Точность = {accuracy:.4f}")
    
    return results

# Выполнение эксперимента
results = run_experiment(X, y)

# Вывод результатов
print("\nРезультаты тестирования:")
for i, acc in enumerate(results):
    print(f"Разбиение {i+1}: Точность = {acc:.4f}")

# Средняя точность
print(f"Средняя точность: {np.mean(results):.4f}")