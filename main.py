from linear_regression_loops import run as run_loops
from linear_regression_matrix import run as run_matrix


def main():
    while True:
        print("\nВыберите реализацию")
        print("1 — матричная NumPy")
        print("2 — через циклы")
        print("0 — выход")
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            run_matrix()
        elif choice == "2":
            run_loops()
        elif choice == "0":
            return
        else:
            print("Введите 1, 2 или 0.")


if __name__ == "__main__":
    main()
