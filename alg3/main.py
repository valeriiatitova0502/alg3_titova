import numpy as np
import time
import matplotlib.pyplot as plt
import random

# Создание матрицы размером n на n
def create_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        matrix.append(row)
    return matrix

# Заполнение матрицы целочисленными числами от 0 до 10

def fill_integer_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = random.randint(0, 10)

# Заполнение матрицы случайными вещественными числами от 0 до 1
def fill_float_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = random.uniform(0, 1)

#транспонирование матрицы
def transpose_matrix(matrix):
    n = len(matrix)
    transposed = create_matrix(n)
    for i in range(n):
        for j in range(n):
            transposed[j][i] = matrix[i][j]
    return transposed

# Функция умножения двух матриц
def multiply_matrices(a, b):
    # Проверка на соответствие размеров матриц для умножения
    if len(a[0]) != len(b):
        return "Матрицы невозможно умножить"
    # Создание матрицы результата
    result = create_matrix(len(a))
    # Заполнение матрицы результата
    for i in range(len(a)):
        for j in range(len(b[0])):
            sum = 0
            for k in range(len(b)):
                sum += a[i][k] * b[k][j]
            result[i][j] = sum
    return result

# Вывод матрицы
# Вывод матрицы
def print_matrix(matrix):
    for row in matrix:
        for value in row:
            if isinstance(value, float):
                print("{:.2f}".format(value), end=" ")
            else:
                print(value, end=" ")
        print()


# Функция проверки результата умножения двух матриц
def check_multiply_matrices(a, b):
    if len(a[0]) != len(b):
        return "Матрицы невозможно умножить"
    product_custom = multiply_matrices(a, b)
    result = ""
    for row in product_custom:
        for value in row:
            if isinstance(value, float):
                result += "{:.2f} ".format(value)
            else:
                result += str(value) + " "
        result += '\n'
    return result


# Создание и заполнение матрицы размером 3 на 3

matrix_3x3 = create_matrix(3)
fill_integer_matrix(matrix_3x3)
transposed_3x3 = transpose_matrix(matrix_3x3)
numpy_transposed_3x3 = np.transpose(np.array(matrix_3x3))

# Создание и заполнение матрицы размером 5 на 5
matrix_5x5 = create_matrix(5)
fill_integer_matrix(matrix_5x5)
transposed_5x5 = transpose_matrix(matrix_5x5)
numpy_transposed_5x5 = np.transpose(np.array(matrix_5x5))

# Создание и заполнение матрицы размером 7 на 7
matrix_7x7 = create_matrix(7)
fill_integer_matrix(matrix_7x7)
transposed_7x7 = transpose_matrix(matrix_7x7)
numpy_transposed_7x7 = np.transpose(np.array(matrix_7x7))

# Создание и заполнение матрицы размером 3 на 3 с нецелочисленными числами

float_matrix_3x3 = create_matrix(3)
fill_float_matrix(float_matrix_3x3)
transposed_float_3x3 = transpose_matrix(float_matrix_3x3)
numpy_float_transposed_3x3 = np.transpose(np.array(float_matrix_3x3))

# Создание и заполнение матрицы размером 5 на 5 с нецелочисленными числами

float_matrix_5x5 = create_matrix(5)
fill_float_matrix(float_matrix_5x5)
transposed_float_5x5 = transpose_matrix(float_matrix_5x5)
numpy_float_transposed_5x5 = np.transpose(np.array(float_matrix_5x5))

# Создание и заполнение матрицы размером 7 на 7 с нецелочисленными числами

float_matrix_7x7 = create_matrix(7)
fill_float_matrix(float_matrix_7x7)
transposed_float_7x7 = transpose_matrix(float_matrix_7x7)
numpy_float_transposed_7x7 = np.transpose(np.array(float_matrix_7x7))

# Вывод матриц
print("Вывод матриц без применения встроенных методов:")

print("Матрица 3 на 3:")
print_matrix(matrix_3x3)

print("\nТранспонированная матрица 3 на 3:")
print_matrix(transposed_3x3)
print("\nПроверка:")
print_matrix(numpy_transposed_3x3)

print("\nУмножение транспонированной и исходной матрицы 3x3:")
matrix_product_3x3 = multiply_matrices(transposed_3x3,  matrix_3x3)
print_matrix(matrix_product_3x3)
result = check_multiply_matrices(transposed_3x3, matrix_3x3)
print("\nРезультат проверки умножения матриц, 3x3 матрица:\n", result)

print("\nМатрица 5 на 5:")
print_matrix(matrix_5x5)

print("\nТранспонированная матрица 5 на 5:")
print_matrix(transposed_5x5)
print("\nПроверка:")
print_matrix(numpy_transposed_5x5)

print("\nУмножение исходной и транспонированной матрицы 5x5:")
matrix_product_5x5 = multiply_matrices(transposed_5x5, matrix_5x5)
print_matrix(matrix_product_5x5)
result = check_multiply_matrices(transposed_5x5, matrix_5x5)
print("\nРезультат проверки умножения матриц, 5x5 матрица: \n", result)

print("\nМатрица 7 на 7:")
print_matrix(matrix_7x7)

print("\nТранспонированная матрица 7 на 7:\n")
print_matrix(transposed_7x7)
print("\nПроверка:")
print_matrix(numpy_transposed_7x7)

print("\nУмножение исходной и транспонированной матрицы 7x7:")
matrix_product_7x7 = multiply_matrices(transposed_7x7, matrix_7x7)
print_matrix(matrix_product_7x7)
result = check_multiply_matrices(transposed_7x7, matrix_7x7)
print("\nРезультат проверки умножения матриц, 7x7 матрица: \n", result)

print("Матрица 3 на 3 с нецелочисленными числами:")
print_matrix(float_matrix_3x3)

print("\nТранспонированная матрица 3 на 3 с нецелочисленными числами:")
print_matrix(transposed_float_3x3)
print("\nПроверка:")
print_matrix(numpy_float_transposed_3x3)

print("\nУмножение транспонированной и исходной матрицы с нецелочисленными числами 3x3:")
float_matrix_product_3x3 = multiply_matrices(transposed_float_3x3, float_matrix_3x3)
print_matrix(float_matrix_product_3x3)
result = check_multiply_matrices(transposed_float_3x3, float_matrix_3x3)
print("\nРезультат проверки умножения матриц, 3x3 матрица: \n", result)

print("Матрица 5 на 5 с нецелочисленными числами:")
print_matrix(float_matrix_5x5)

print("\nТранспонированная матрица 5 на 5 с нецелочисленными числами:")
print_matrix(transposed_float_5x5)
print("\nПроверка:")
print_matrix(numpy_float_transposed_5x5)

print("\nУмножение транспонированной и исходной матрицы с нецелочисленными числами 5x5:")
float_matrix_product_5x5 = multiply_matrices(transposed_float_5x5, float_matrix_5x5)
print_matrix(float_matrix_product_5x5)
result = check_multiply_matrices(transposed_float_5x5, float_matrix_5x5)
print("\nРезультат проверки умножения матриц, 5x5 матрица: \n", result)

print("Матрица 7 на 7 с нецелочисленными числами:")
print_matrix(float_matrix_7x7)

print("\nТранспонированная матрица 7 на 7 с нецелочисленными числами:")
print_matrix(transposed_float_7x7)
print("\nПроверка:")
print_matrix(numpy_float_transposed_7x7)

print("\nУмножение транспонированной и исходной матрицы с нецелочисленными числами 7x7:")
float_matrix_product_7x7 = multiply_matrices(transposed_float_7x7, float_matrix_7x7)
print_matrix(float_matrix_product_7x7)
result = check_multiply_matrices(transposed_float_7x7, float_matrix_7x7)
print("\nРезультат проверки умножения матриц, 7x7 матрица: \n", result)

# Проверка с помощью встроенных функций
# Функция для создания рандомной матрицы заданной размерности и типа данных

def create_random_matrix(rows, columns, data_type):
    if data_type == "int":
        return np.random.randint(0, 10, size=(rows, columns))  # Генерирует целые числа от 0 до 9
    elif data_type == "float":
        return np.random.rand(rows, columns)  # Генерирует числа с плавающей точкой от 0 до 1

# Функция для умножения матрицы на её транспонированную версию
def multiply_with_transpose(matrix):
    transposed_matrix = matrix.T  # Транспонируем матрицу
    result = np.dot(transposed_matrix, matrix)  # Умножаем транспонированную матрицу на исходную
    return result

# Размерности матриц
sizes = [3, 5, 7]

for size in sizes:
    integer_matrix = create_random_matrix(size, size, "int")
    float_matrix = create_random_matrix(size, size, "float")
    result_integer = multiply_with_transpose(integer_matrix)
    result_float = multiply_with_transpose(float_matrix)

# Определение функций для измерения времени работы процессов
def time_custom_with_and_without_creation(size, data_type):
    # Включающее создание матрицы
    start_with_creation = time.time()
    matrix = create_matrix(size)
    if data_type == "int":
        fill_integer_matrix(matrix)
    else:
        fill_float_matrix(matrix)
    transposed = transpose_matrix(matrix)
    multiply_matrices(matrix, transposed)
    time_with_creation = time.time() - start_with_creation

    # Не включающее создание матрицы
    start_without_creation = time.time()
    transposed = transpose_matrix(matrix)
    multiply_matrices(matrix, transposed)
    time_without_creation = time.time() - start_without_creation

    return time_with_creation, time_without_creation


# numpy функция
def time_numpy_with_and_without_creation(size, data_type):
    # Включающее создание матрицы
    start_with_creation = time.time()
    matrix = create_random_matrix(size, size, data_type)
    transposed_matrix = matrix.T
    np.dot(transposed_matrix, matrix)
    time_with_creation = time.time() - start_with_creation

    # Не включающее создание матрицы
    start_without_creation = time.time()
    transposed_matrix = matrix.T
    np.dot(transposed_matrix, matrix)
    time_without_creation = time.time() - start_without_creation

    return time_with_creation, time_without_creation

sizes_custom = range(50, 301, 1)  # Меняем на ваши значения
sizes_numpy = range(5, 1001, 1)   # Меняем на ваши значения
time_custom_int = np.array([time_custom_with_and_without_creation(s, 'int') for s in sizes_custom])
time_custom_float = np.array([time_custom_with_and_without_creation(s, 'float') for s in sizes_custom])

time_numpy_int = np.array([time_numpy_with_and_without_creation(s, 'int') for s in sizes_numpy])
time_numpy_float = np.array([time_numpy_with_and_without_creation(s, 'float') for s in sizes_numpy])

# Функции для измерения времени работы процессов
def time_custom_with_and_without_creation(size, data_type):
    # Включающее создание матрицы
    start_with_creation = time.time()
    matrix = create_matrix(size)
    if data_type == "int":
        fill_integer_matrix(matrix)
    else:
        fill_float_matrix(matrix)
    multiply_matrices(transpose_matrix(matrix), matrix)
    time_with_creation = time.time() - start_with_creation

    # Не включающее создание матрицы
    start_without_creation = time.time()
    multiply_matrices(transpose_matrix(matrix), matrix)
    time_without_creation = time.time() - start_without_creation

    return time_with_creation, time_without_creation

# numpy функция
def time_numpy_with_and_without_creation(size, data_type):
    # Включающее создание матрицы
    start_with_creation = time.time()
    matrix = create_random_matrix(size, size, data_type)
    np.dot(matrix.T, matrix)
    time_with_creation = time.time() - start_with_creation

    # Не включающее создание матрицы
    start_without_creation = time.time()
    transposed_matrix = matrix.T
    np.dot(transposed_matrix, matrix)
    time_without_creation = time.time() - start_without_creation

    return time_with_creation, time_without_creation

charts = [
    ("Зависимость времени транспонирования нецелочисленной матрицы (ручной метод)", sizes_custom, time_custom_float),
    ("Зависимость времени транспонирования целочисленной матрицы (ручной метод)", sizes_custom, time_custom_int),
    ("Зависимость времени транспонирования нецелочисленной матрицы (встроенный метод)", sizes_numpy, time_numpy_float[:, ::-1]),
    ("Зависимость времени транспонирования целочисленной матрицы (встроенный метод)", sizes_numpy, time_numpy_int[:, ::-1]),
    ("Время умножения транспонированной на исходную (нецелочисленная, ручной метод)", sizes_custom, time_custom_float),
    ("Время умножения транспонированной на исходную (целочисленная, ручной метод)", sizes_custom, time_custom_int),
    ("Время умножения транспонированной на исходную (нецелочисленная, встроенный метод)", sizes_numpy, time_numpy_float),
    ("Время умножения транспонированной на исходную (целочисленная, встроенный метод)", sizes_numpy, time_numpy_int),
]

for i, chart in enumerate(charts, start=1):
    plt.figure()
    # Множество результаты времени выполнения на 1000 для получения миллисекунд
    time_with_creation_ms = chart[2][:, 0] * 1000
    time_without_creation_ms = chart[2][:, 1] * 1000
    plt.plot(chart[1], time_with_creation_ms, label='С учетом формирования матрицы')
    plt.plot(chart[1], time_without_creation_ms, label='Без учета формирования матрицы')
    plt.title(chart[0])
    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (мс)")
    plt.xticks(np.arange(min(chart[1]), max(chart[1]) + 1, 100))
    plt.yticks()
    plt.legend()
    plt.grid(True)  # Добавить сетку
    plt.show()