#  обучающий цикл: https://www.youtube.com/playlist?list=PLOSf9rRg-fvJt5adxPbefB394QBtW9smZ
#  автор: Дмитрий Коробченко 2021г.

import random
import numpy as np

INPUT_DIM = 4  # количество входящих параметров
OUT_DIM = 3  # количество вариантов ответа (вероятности правильных ответов)
H_DIM = 10  # количество нейронов в промежуточном слое


# Функция - активатор
def relu(t):
    return np.maximum(t, 0)


def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)


def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full


def relu_derivative(t):
    return (t >= 0).astype(float)


from sklearn import datasets  # имя библиотеки: scikit-learn (pip install scikit-learn)

iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
print(f"Загрузили подготовленные данные для обучения (из библиотеки scikit-learn.iris): {len(dataset)} шт.")

W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

W1 = (W1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1 / H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1 / H_DIM)

ALPHA = 0.002  # Шаг градиентного спуска (скорость обучения) (меньше - дольше) (0.0002)
NUM_EPOCHS = 100  # Количество "эпох" для обучения (больше - дольше, точнее) (400)
BATCH_SIZE = 5  # размер "пакета" (50)

loss_arr = []

for ep in range(NUM_EPOCHS):
    # print(f"---------------------------------------------------------------------------------------------- Эпоха: {ep}")
    random.shuffle(dataset)  # перемешаем данные, для лучшего обучения

    for i in range(len(dataset) // BATCH_SIZE):

        batch_x, batch_y = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)  # набор данных (порция BATCH_SIZE наборов данных)
        y = np.array(batch_y)  # Правильный ответ.  Внимание "y" - это не ВЕКТОР, а ИНДЕКС правильного класса (порция BATCH_SIZE правильных ответов)

        # итерация алгоритма обучения:

        # 1. Forward (прямое распространение)
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)  # вероятности, предсказанные нашей моделью
        E = np.sum(sparse_cross_entropy_batch(z, y))  # ошибка (разреженная кросс-энтропия)

        # 2. Backward (обратное распространение)
        y_full = to_full_batch(y, OUT_DIM)  # полный вектор правильного ответа
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2  # "@" это матричное умножение, ".T" - это транспонированное значение
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_derivative(t1)  # relu_derivative(t1) это производная функции активации в точке t1
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # 3. Update (обновление весов) - делаем смещение в сторону антиградиента (поэтому знак -)
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)

        # print(f"Набор данных: {i} (порция: {BATCH_SIZE}) ошибка: {E}")
        # print(f"W1:\n{W1}\n\nb1:\n{b1}\n\nW2:\n{W2}\n\nb2:\n{b2}")

# здесь мы обучили нашу сеть, дальше с ней можно работать - сохранить её, и тд.


# -----------------------------------------------------------------------------------------------------------------------
# теперь попробуем проверить её на наших тестовых данных - посчитаем процент распознавания всего тестового массива

def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax_batch(t2)
    return z


def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc, correct


print(f"---------------------------------------------------------------------------------------------- ИТОГИ:")
accuracy, correct = calc_accuracy()
print(f"Точность: {accuracy} (корректно распознано {correct} из {len(dataset)}) Последняя ошибка: {E} Пакетный режим! BATCH_SIZE={BATCH_SIZE}")

# выведем график ошибок по итерациям (хорошо, если график будет снижаться, падать)
import matplotlib.pyplot as plt  #  https://habr.com/ru/post/468295/
plt.plot(loss_arr)
plt.show()
