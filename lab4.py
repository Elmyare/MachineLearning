import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# Загрузка данных
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

# Преобразование данных в формат, подходящий для нейронной сети
x_train = pad_sequences(x_train, maxlen=100)  # Ограничиваем длину текста до 100 слов
x_test = pad_sequences(x_test, maxlen=100)

# Преобразование меток в категориальный формат
y_train = to_categorical(y_train, 46)  # 46 классов
y_test = to_categorical(y_test, 46)

# Создание модели
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(100,)))  # Первый слой
model.add(Dropout(0.5))  # Dropout для предотвращения переобучения
model.add(Dense(256, activation='relu'))  # Второй слой
model.add(Dropout(0.5))  # Dropout
model.add(Dense(128, activation='relu'))  # Третий слой
model.add(Dropout(0.5))  # Dropout
model.add(Dense(46, activation='softmax'))  # Выходной слой

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# Оценка модели на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовой выборке: {test_acc:.4f}")

# Визуализация процесса обучения
plt.figure(figsize=(12, 4))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('График потерь')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

# График точности
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('График точности')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.show()