"""
user_features_unified.py

Objetivo:
 - Ler CSVs de treino (./data/files/treino/treino_parte*.csv).
 - Aplicar as funções antigas (create_diff_time_attributes_non_logged, etc.)
 - Em vez de salvar .parquet intermediários, fazemos tudo em memória
 - Gerar UM arquivo final: user_data_enriched.parquet
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime

################### Funções antigas adaptadas ###################

def converter_timestamp(timestamp):
    """Converte milissegundos (string) em '%d/%m/%Y %H:%M'."""
    return datetime.fromtimestamp(int(timestamp)/1000).strftime('%d/%m/%Y %H:%M')

def processar_timestamps(linha):
    """Ex.: "1658773550000,1658773555000" => "01/07/2022 10:00;01/07/2022 10:05"."""
    if not isinstance(linha, str):
        return ""
    parts = linha.split(',')
    datas_formatadas = []
    for ts in parts:
        ts = ts.strip()
        if ts:
            datas_formatadas.append(converter_timestamp(ts))
    return ';'.join(datas_formatadas)

def create_diff_time_attributes_non_logged(df, data_column_name, timestamp_column_name):
    """
    Calcula diffs entre leituras consecutivas e salva na coluna 'diff_seconds'.
    """
    print(f"create_diff_time_attributes_non_logged: df.shape={df.shape}")

    # Converter timestamp -> string de datas
    df[data_column_name] = df[timestamp_column_name].apply(processar_timestamps)

    # Calcular diffs
    diffs = []
    for _, row in df.iterrows():
        datas_str = row[data_column_name]
        if not datas_str:
            diffs.append(0.0)
            continue
        datas_list = datas_str.split(';')
        datas_sorted = sorted(datas_list)
        total_diff = 0.0
        for i in range(1, len(datas_sorted)):
            dt1 = datetime.strptime(datas_sorted[i-1], '%d/%m/%Y %H:%M')
            dt2 = datetime.strptime(datas_sorted[i],   '%d/%m/%Y %H:%M')
            total_diff += (dt2 - dt1).total_seconds()
        diffs.append(total_diff)
    df["diff_seconds"] = diffs

def create_attributes_non_logged(df, column_name, type_column):
    """
    Cria colunas de engajamento (soma ou média) no DataFrame em memória.
    ex.: 'eng_column_name'.
    """
    print(f"create_attributes_non_logged: col={column_name}, type={type_column}, df.shape={df.shape}")
    results = []
    for _, row in df.iterrows():
        col_str = row.get(column_name, "")
        if not isinstance(col_str, str) or not col_str:
            results.append(0.0)
            continue
        parts = col_str.split(',')
        if type_column == 'int':
            arr = list(map(int, parts))
            val = sum(arr)
        else:
            arr = list(map(float, parts))
            val = float(np.mean(arr)) if arr else 0.0
        results.append(val)
    df[f"eng_{column_name}"] = results

################### Fim das funções antigas adaptadas ###################

def main():
    # 1) Ler todos CSV: ./data/files/treino/treino_parte*.csv
    base_path = "./data/files/treino/*.csv"
    all_dfs = []
    for full_path in glob.glob(base_path):
        df_part = pd.read_csv(full_path)
        match = re.search(r'parte(\d+)', full_path)
        print(f"Lendo {full_path}, shape={df_part.shape}, parte={match.group(0) if match else '??'}")
        all_dfs.append(df_part)

    if not all_dfs:
        print("Nenhum CSV de treino encontrado em data/files/treino/")
        return

    # 2) Definir colunas
    columns_to_create_count_attr = ['numberOfClicksHistory', 'timeOnPageHistory', 'pageVisitsCountHistory']
    columns_to_create_mean_attr  = ['scrollPercentageHistory']

    # 3) Aplicar funções em cada DF
    for i, df_local in enumerate(all_dfs):
        print(f"\n--- Processando df {i}, shape={df_local.shape} ---")

        # Ex.: diff time
        create_diff_time_attributes_non_logged(
            df_local,
            data_column_name="dataHistory",
            timestamp_column_name="timestampHistory"
        )

        # Ex.: mean
        for col in columns_to_create_mean_attr:
            create_attributes_non_logged(df_local, col, 'float')

        # Ex.: count
        for col in columns_to_create_count_attr:
            create_attributes_non_logged(df_local, col, 'int')

    # 4) Concatenar tudo
    df_user_enriched = pd.concat(all_dfs, ignore_index=True)
    print(f"Tamanho final do df_user_enriched: {df_user_enriched.shape}")

    # 5) Salvar em UM arquivo
    output_file = "./data/user_data_enriched.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_user_enriched.to_parquet(output_file, index=False)
    print(f"Gerado: {output_file}")
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()
