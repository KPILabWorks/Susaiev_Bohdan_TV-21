import pandas as pd 
import pyarrow.parquet as pq
import pyarrow as pa
import os

# Функція для кешування даних у форматі Parquet
def cache_data_as_parquet(data, file_path, compression='snappy'):
    # Перетворення pandas DataFrame в Arrow Table для запису в Parquet
    table = pa.Table.from_pandas(data)
    # Запис даних у файл Parquet з вказаною компресією
    pq.write_table(table, file_path, compression=compression)
    print(f"Дані збережено в {file_path} з використанням компресії {compression}.")

# Функція для завантаження кешованих даних з файлу Parquet
def load_cached_data(file_path):
    if os.path.exists(file_path):  # Перевірка наявності файлу
        table = pq.read_table(file_path)  # Читання таблиці з файлу
        return table.to_pandas()  # Повернення даних як pandas DataFrame
    else:
        print(f"Не знайдено кешованих даних за шляхом {file_path}.")
        return None

# Імітація функції обробки даних (створення прикладу даних)
def process_data():
    # Генерація даних для споживання енергії за допомогою pandas DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2025-03-01', periods=5, freq='H'),
        'energy_consumption': [100, 150, 200, 250, 300]
    })
    return data

# Функція для порівняння форматів Parquet і CSV
def compare_formats():
    # Обробка даних
    data = process_data()

    # Кешування даних у форматі Parquet
    parquet_file = 'energy_data.parquet'
    cache_data_as_parquet(data, parquet_file)
    
    # Завантаження даних з Parquet
    cached_parquet_data = load_cached_data(parquet_file)
    print("Дані завантажено з Parquet:")
    print(cached_parquet_data)

    # Кешування даних у форматі CSV як альтернативу
    csv_file = 'energy_data.csv'
    data.to_csv(csv_file, index=False)  # Запис даних у формат CSV
    print(f"Дані збережено у форматі CSV за шляхом {csv_file}")

    # Завантаження даних з CSV
    cached_csv_data = pd.read_csv(csv_file)
    print("Дані завантажено з CSV:")
    print(cached_csv_data)

# Виклик функції для порівняння форматів
compare_formats()
