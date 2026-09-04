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


def read_point(point_number, dimension):
    expected_count = dimension + 1
    while True:
        raw_values = input(
            f"Точка {point_number}. Введите {dimension} координат и значение y через пробел: "
        ).strip()
        try:
            values = [float(value.replace(",", ".")) for value in raw_values.split()]
        except ValueError:
            print("Используйте только конечные числа, разделённые пробелами.")
            continue
        if len(values) != expected_count:
            print(f"Нужно ввести ровно {expected_count} чисел.")
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
    return np.asarray(points, dtype=float).T


def train(data, epochs):
    features = data[:-1, :]
    targets = data[-1, :]
    weights = np.full(features.shape[0], INITIAL_WEIGHT, dtype=float)
    bias = float(INITIAL_BIAS)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for _ in range(epochs):
                predictions = weights @ features + bias
                errors = predictions - targets
                gradient_weights = 2.0 * (features @ errors)
                gradient_bias = 2.0 * np.sum(errors)
                weights -= LEARNING_RATE * gradient_weights
                bias -= LEARNING_RATE * gradient_bias
                if not np.all(np.isfinite(weights)) or not math.isfinite(bias):
                    raise FloatingPointError
            final_errors = weights @ features + bias - targets
            loss = float(np.sum(final_errors ** 2))
    except FloatingPointError as error:
        raise FloatingPointError("Параметры стали неконечными") from error
    return weights, bias, loss


def print_result(weights, bias, loss):
    print("\nРезультат обучения")
    print("Вектор весов:")
    for index, value in enumerate(weights, start=1):
        print(f"w{index} = {value:.10f}")
    print(f"b = {bias:.10f}")
    print(f"Сумма квадратов ошибок = {loss:.10f}")


def run():
    print("Многомерная линейная регрессия: матричная реализация NumPy")
    print(f"Скорость обучения: {LEARNING_RATE}")
    print(f"Начальные веса: {INITIAL_WEIGHT}, начальный b: {INITIAL_BIAS}")
    dimension = read_positive_int("Введите размерность пространства n: ")
    sample_count = read_positive_int("Введите число экспериментальных точек m: ")
    data = read_data(dimension, sample_count)
    epochs = read_nonnegative_int("Введите число эпох: ")
    try:
        weights, bias, loss = train(data, epochs)
    except FloatingPointError:
        print("Вычисления расходятся. Используйте данные меньшего масштаба или меньшее число эпох.")
        return
    print_result(weights, bias, loss)


if __name__ == "__main__":
    run()
