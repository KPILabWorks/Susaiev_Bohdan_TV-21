import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# === CONFIGURATION ===
input_file = "C:\\Users\\Bohdan\\Downloads\\large_data.json"  # Абсолютний шлях з подвійними слешами
chunk_size = 10000
output_file = "processed_data.parquet"

# === CUSTOM TRANSFORMATIONS ===
def clean_email(email):
    if pd.isna(email):
        return email
    return email.lower().strip()

def categorize_industry_vectorized(industry_series):
    conditions = [
        industry_series.str.contains("tech", case=False, na=False),
        industry_series.str.contains("finance", case=False, na=False)
    ]
    choices = ['Technology', 'Finance']
    return np.select(conditions, choices, default='Other')

# === PROCESSING FUNCTION ===
def process_chunk(chunk):
    # Normalize nested 'address' and 'company'
    if 'address' in chunk.columns:
        address_df = pd.json_normalize(chunk['address'])
        address_df.columns = [f'address_{col}' for col in address_df.columns]
        chunk = chunk.drop('address', axis=1)
        chunk = pd.concat([chunk, address_df], axis=1)

    if 'company' in chunk.columns:
        company_df = pd.json_normalize(chunk['company'])
        company_df.columns = [f'company_{col}' for col in company_df.columns]
        chunk = chunk.drop('company', axis=1)
        chunk = pd.concat([chunk, company_df], axis=1)

    # Transformations
    chunk['name'] = chunk['name'].str.upper()
    chunk['email'] = chunk['email'].apply(clean_email)
    chunk['company_industry'] = categorize_industry_vectorized(chunk['company_industry'])
    chunk['status'] = chunk['profile_complete'].map({True: 'Complete', False: 'Incomplete'})

    # Optimize dtypes
    dtype_map = {
        'id': 'int32',
        'profile_complete': 'bool',
        'company_industry': 'category',
        'status': 'category',
    }
    for col, dtype in dtype_map.items():
        if col in chunk.columns:
            chunk[col] = chunk[col].astype(dtype)

    # Convert datetime
    if 'created_at' in chunk.columns:
        chunk['created_at'] = pd.to_datetime(chunk['created_at'], errors='coerce')

    return chunk

# === MAIN EXECUTION ===
def process_large_json(input_file, chunk_size):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found.")

    print(f"Reading {input_file} in chunks using Pandas read_json with chunksize={chunk_size}...")

    processed_chunks = []
    # Для звичайного JSON (масиву) chunk_size не працює,
    # тому тут краще читати повністю, або конвертувати у JSON Lines.
    # Якщо файл великий, рекомендую конвертувати у JSON Lines.
    # Тут для демонстрації читаємо повністю.
    df_full = pd.read_json(input_file)
    
    # Обробляємо дані по чанках вручну
    total_records = len(df_full)
    for start in tqdm(range(0, total_records, chunk_size), desc="Processing chunks"):
        chunk = df_full.iloc[start:start+chunk_size].copy()
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)

    final_df = pd.concat(processed_chunks, ignore_index=True)

    # Final optimization
    str_cols = ['name', 'email', 'address_city', 'address_street', 'company_name']
    for col in str_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype('string')

    print(f"Saving to {output_file}...")
    final_df.to_parquet(output_file, engine='pyarrow', index=False)

    final_df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"\nProcessed data saved to: {os.path.abspath(output_file)}")

    print("Done.")
    final_df.info(memory_usage='deep')
    return final_df

# === RUN ===
try:
    df = process_large_json(input_file, chunk_size)
except Exception as e:
    print(f"Error occurred: {str(e)}")
