import math

import numpy as np


LEARNING_RATE = 0.001
INITIAL_WEIGHT = 0.0
INITIAL_BIAS = 0.0


def read_positive_int(prompt):
    while True:
        value = input(prompt).strip()
        try:
            number = int(value)
        except ValueError:
            print("Введите целое число.")
            continue
        if number > 0:
            return number
        print("Введите число больше нуля.")


def read_nonnegative_int(prompt):
    while True:
        value = input(prompt).strip()
        try:
            number = int(value)
        except ValueError:
            print("Введите целое число.")
            continue
        if number >= 0:
            return number
        print("Введите неотрицательное число.")


def read_point(point_number, dim):
    count = dim + 1
    while True:
        raw_values = input(
            f"Точка {point_number}. Введите {dim} координат и значение y через пробел: "
        ).strip()
        try:
            values = [float(value.replace(",", ".")) for value in raw_values.split()]
        except ValueError:
            print("Используйте только конечные числа, разделённые пробелами.")
            continue
        if len(values) != count:
            print(f"Нужно ввести ровно {count} чисел.")
            continue
        if not all(math.isfinite(value) for value in values):
            print("Используйте только конечные числа.")
            continue
        return values


def read_data(dimension, sample_count):
    print("Каждая строка содержит x1 ... xn y для одной экспериментальной точки.")
    points = []
    for point_number in range(1, sample_count + 1):
        points.append(read_point(point_number, dimension))
    return points


def train(points, dimension, epochs):
    weights = np.full(dimension, INITIAL_WEIGHT, dtype=float)
    bias = float(INITIAL_BIAS)
    for _ in range(epochs):
        gradient_weights = np.zeros(dimension, dtype=float)
        gradient_bias = 0.0
        for point in points:
            y_hat = bias
            for feature_index in range(dimension):
                y_hat += weights[feature_index] * point[feature_index]
            error = y_hat - point[dimension]
            gradient_bias += 2.0 * error
            for feature_index in range(dimension):
                gradient_weights[feature_index] += 2.0 * error * point[feature_index]
        for feature_index in range(dimension):
            weights[feature_index] -= LEARNING_RATE * gradient_weights[feature_index]
        bias -= LEARNING_RATE * gradient_bias
        if not np.all(np.isfinite(weights)) or not math.isfinite(bias):
            raise FloatingPointError("Параметры стали неконечными")
    loss = 0.0
    for point in points:
        y_hat = bias
        for feature_index in range(dimension):
            y_hat += weights[feature_index] * point[feature_index]
        error = y_hat - point[dimension]
        loss += error * error
    if not math.isfinite(loss):
        raise FloatingPointError("Значение функции потерь стало неконечным")
    return weights, bias, loss


def print_result(weights, bias, loss):
    print("\nРезультат обучения")
    print("Вектор весов:")
    for index, value in enumerate(weights, start=1):
        print(f"w{index} = {value:.10f}")
    print(f"b = {bias:.10f}")
    print(f"Сумма квадратов ошибок = {loss:.10f}")


def run():
    print("Многомерная линейная регрессия: реализация через циклы")
    print(f"Скорость обучения: {LEARNING_RATE}")
    print(f"Начальные веса: {INITIAL_WEIGHT}, начальный b: {INITIAL_BIAS}")
    dimension = read_positive_int("Введите размерность пространства n: ")
    sample_count = read_positive_int("Введите число экспериментальных точек m: ")
    points = read_data(dimension, sample_count)
    epochs = read_nonnegative_int("Введите число эпох: ")
    try:
        weights, bias, loss = train(points, dimension, epochs)
    except FloatingPointError:
        print("Вычисления расходятся. Используйте данные меньшего масштаба или меньшее число эпох.")
        return
    print_result(weights, bias, loss)


if __name__ == "__main__":
    run()
