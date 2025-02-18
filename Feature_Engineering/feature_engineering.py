import re
from turtle import right
from unittest import result
import pandas as pd
import numpy as np
import glob

from scipy.stats import linregress
from sqlalchemy import column
from datetime import timedelta
from Scripts.functions_to_help import *

def create_attributes_count_mean(df, user_type, column_name, type_column, parte_treino):
    
    df_anonimo = df[df['userType'] == user_type]
        
    df_anonimo['history'] = df_anonimo['history'].str.split(',')
    df_anonimo[column_name] = df_anonimo[column_name].str.split(',')

    df_history_explode = df_anonimo[['history', column_name]].apply(pd.Series.explode)
    
    if type_column == 'int':
        df_history_explode[column_name] = df_history_explode[column_name].astype(int)

        table = df_history_explode.groupby('history').agg(total=(column_name, 'sum')).reset_index()
    elif type_column == 'float':
        df_history_explode[column_name] = df_history_explode[column_name].astype(float)
        
        table = df_history_explode.groupby('history').agg(total=(column_name, 'mean')).reset_index()
            
    if type_column == 'int':
        attribute_type = 'count'
    elif type_column == 'float':
        attribute_type = 'count'
        
    table.to_parquet('./Dados/files/treino/Atributos/table_{0}_{1}_by_page_treino_{2}_{3}.parquet'.format(attribute_type, column_name, parte_treino, str(user_type).replace(r"-", "")))

def create_diff_time_attributes(df, user_type, data_column_name, timestamp_column_name,parte_treino):
        
    df_anonimo = df[df['userType'] == user_type]
    
    df_anonimo[data_column_name] = df_anonimo[timestamp_column_name].apply(processar_timestamps)

    df_anonimo['history'] = df_anonimo['history'].str.split(',')
    df_anonimo[data_column_name] = df_anonimo[data_column_name].str.split(',')

    df_history_explode = df_anonimo[['history', data_column_name]].apply(pd.Series.explode)

    df_history_explode[data_column_name] = pd.to_datetime(df_history_explode[data_column_name], format = "%d/%m/%Y %H:%M")

    df_history_explode.sort_values(by = ['history', data_column_name], inplace = True)

    df_history_explode['diff_seconds'] = df_history_explode.groupby('history')[data_column_name].diff().dt.total_seconds()

    tables_media_diff = df_history_explode.groupby('history')['diff_seconds'].sum().reset_index()

    table = tables_media_diff.groupby('history')['diff_seconds'].mean().reset_index()
    
    table.to_parquet('./Dados/files/treino/Atributos/table_mean_diff_{0}_by_page_treino_{1}_{2}.parquet'.format(data_column_name, parte_treino, str(user_type).replace(r"-", "")))


'''
1 - 00:00
1 - 00:01
1 - 00:02
2 - 00:00
2 - 05:00

id diff_accesso_medio
1 - 1 min
2 - 300 min
'''

def create_count_userId_by_history(df, user_type, parte_treino):
    
    
    df_filtrado = df[df['userType'] == user_type]
    
    
    df_filtrado['history'] = df_filtrado['history'].str.split(',')
    
    df_history_explode = df_filtrado.explode('history')
    
    table = df_history_explode.groupby('history').agg(count_userId_by_page=('userId', 'nunique')).reset_index()
    
    table.to_parquet('./Dados/files/treino/Atributos/table_count_userId_by_page_treino_{0}_{1}.parquet'.format(parte_treino, str(user_type).replace(r"-", "")))
    

def create_flag_was_modified():
    
    BASE_PATH_TABLE_COUNT_USERID = "./Dados/files/treino/Atributos/table_count_userId_*.parquet"
    BASE_PATH_ITENS = "./Dados/itens/itens/*.csv"
    
    
    df_tables_count_id = {}

    for full_path in glob.glob(BASE_PATH_TABLE_COUNT_USERID):
        df = pd.read_parquet(full_path)
        
        match = re.search(r'parte(\d+)', full_path)
        df_tables_count_id[match.group(0)] = df
    
    df_itens = {}
    
    for full_path in glob.glob(BASE_PATH_ITENS):
        df = pd.read_csv(full_path)
        
        match = re.search(r'parte(\d+)', full_path)
        df_itens[match.group(0)] = df
    
    
    list_itens_parts = list(df_itens.keys())

    list_tables_parts = list(df_tables_count_id.keys())
    
    
    
    for table_part in list_tables_parts:
        list_df_left_join = []
        for itens_part in list_itens_parts:
            df_itens[itens_part]['is_modified'] = np.where(df_itens[itens_part]['issued'] == df_itens[itens_part]['modified'], 0, 1)
            list_df_left_join.append(df_tables_count_id[table_part].merge(df_itens[itens_part][['page', 'is_modified']], left_on = 'history', right_on = 'page', how = 'left'))
        
        df_left_join = pd.concat(list_df_left_join, ignore_index=True)
        df_left_join_grouped = df_left_join.groupby('history', as_index=False).first()
    
    
        df_left_join_grouped[['history', 'is_modified']].to_parquet('./Dados/files/treino/Atributos/table_flag_is_modified_by_page_treino_{0}_Nonlogged.parquet'.format(table_part))

def create_count_days_by_page(df, user_type, parte_treino):
    
    df_filtrado = df[df['userType'] == user_type]
    
    df_filtrado['dataHistory'] = df_filtrado['timestampHistory'].apply(processar_timestamps)

    df_filtrado['history'] = df_filtrado['history'].str.split(',')

    df_filtrado['dataHistory'] = df_filtrado['dataHistory'].str.split(',')

    df_explode = df_filtrado[['history', 'dataHistory']].apply(pd.Series.explode)

    df_explode['dataHistory'] = pd.to_datetime(df_explode['dataHistory'], format = "%d/%m/%Y %H:%M")

    df_explode['dataHistory_day'] = df_explode['dataHistory'].dt.date


    table = df_explode.groupby('history').agg(qnt_dias=('dataHistory_day', 'nunique')).reset_index()

    table.to_parquet('./Dados/files/treino/Atributos/table_count_days_by_page_treino_{0}_{1}.parquet'.format(parte_treino, str(user_type).replace(r"-", "")))
    

def create_flag_was_modified():
    
    BASE_PATH_TABLE_COUNT_USERID = "./Dados/files/treino/Atributos/table_count_userId_*.parquet"
    BASE_PATH_ITENS = "./Dados/itens/itens/*.csv"
    
    
    df_tables_count_id = {}

    for full_path in glob.glob(BASE_PATH_TABLE_COUNT_USERID):
        df = pd.read_parquet(full_path)
        
        match = re.search(r'parte(\d+)', full_path)
        df_tables_count_id[match.group(0)] = df
    
    df_itens = {}
    
    for full_path in glob.glob(BASE_PATH_ITENS):
        df = pd.read_csv(full_path)
        
        match = re.search(r'parte(\d+)', full_path)
        df_itens[match.group(0)] = df
    
    
    list_itens_parts = list(df_itens.keys())

    list_tables_parts = list(df_tables_count_id.keys())
    
    
    
    for table_part in list_tables_parts:
        list_df_left_join = []
        for itens_part in list_itens_parts:
            df_itens[itens_part]['is_modified'] = np.where(df_itens[itens_part]['issued'] == df_itens[itens_part]['modified'], 0, 1)
            list_df_left_join.append(df_tables_count_id[table_part].merge(df_itens[itens_part][['page', 'is_modified']], left_on = 'history', right_on = 'page', how = 'left'))
        
        df_left_join = pd.concat(list_df_left_join, ignore_index=True)
        df_left_join_grouped = df_left_join.groupby('history', as_index=False).first()
    
    
        df_left_join_grouped[['history', 'is_modified']].to_parquet('./Dados/files/treino/Atributos/table_flag_is_modified_by_page_treino_{0}_Nonlogged.parquet'.format(table_part))


def create_count_in_period(df, user_type, table_part):
    
    BASE_PATH_ITENS = "./Dados/itens/itens/*.csv"
    
    df_filtrado = df[df['userType'] == user_type]
    
    df_filtrado['dataHistory'] = df_filtrado['timestampHistory'].apply(processar_timestamps)

    df_filtrado['history'] = df_filtrado['history'].str.split(',')

    df_filtrado['dataHistory'] = df_filtrado['dataHistory'].str.split(',')

    df_explode = df_filtrado[['history', 'dataHistory']].apply(pd.Series.explode)

    df_explode['dataHistory'] = pd.to_datetime(df_explode['dataHistory'], format = "%d/%m/%Y %H:%M")
    
    df_itens = {}
    
    for full_path in glob.glob(BASE_PATH_ITENS):
        df = pd.read_csv(full_path)
        
        df['issued'] = pd.to_datetime(df['issued'])
        
        match = re.search(r'parte(\d+)', full_path)
        df_itens[match.group(0)] = df
    
    
    list_itens_parts = list(df_itens.keys())
    list_df_left_join = list()
    
    for itens_part in list_itens_parts:
        
        
        list_df_left_join.append(df_explode.merge(df_itens[itens_part][['page', 'issued']], left_on = 'history', right_on = 'page', how = 'left'))
    
    
    
    df_left_join = pd.concat(list_df_left_join, ignore_index=True)
    
    
    df_filtered = df_left_join[(df_left_join['dataHistory'] >= df_left_join['issued']) &
                                (df_left_join['dataHistory'] <= df_left_join['issued'] + timedelta(days=1))]
    
    
    table = df_filtered.groupby('history').size().reset_index(name='count_in_first_day')
    table.to_parquet('./Dados/files/treino/Atributos/table_count_page_users_in_first_day_treino_{0}_Nonlogged.parquet'.format(table_part))

import os
import pandas as pd
from collections import defaultdict

def merge_files_by_same_part(folder_path, key_column, output_folder):

    # Dicionário para agrupar arquivos por número de parte
    part_files = defaultdict(list)

    # Percorre todos os arquivos na pasta
    for filename in os.listdir(folder_path):
        # Verifica se o arquivo contém "parteX" onde X é de 1 a 5
        for i in range(1, 6):
            if f"parte{i}" in filename:
                # Adiciona o arquivo ao grupo correspondente
                part_files[i].append(filename)
                break

    # Verifica se há arquivos agrupados
    if not part_files:
        print("Nenhum arquivo com 'parteX' encontrado.")
        return

    # Cria a pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)

    # Processa cada grupo de arquivos com a mesma parte
    for part_number, files in part_files.items():
        # Lista para armazenar os DataFrames do grupo
        dfs = []

        # Lê todos os arquivos do grupo
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            try:
                # Lê o arquivo (supondo que seja CSV)
                df = pd.read_parquet(file_path)
                print(filename)
                padrao = r"table_count_(.*?)_treino_parte(.*?)_Nonlogged.parquet"

                # Procurar o padrão na string
                column_count_name = re.search(padrao, filename)
                
                try:
                    column_count_name = column_count_name.group(1)
                    
                    df.rename(columns={'total': column_count_name}, inplace = True)
                    dfs.append(df)
                except:
                    
                    dfs.append(df)
            except Exception as e:
                print(f"Erro ao ler o arquivo {filename}: {e}")

        # Faz o merge dos DataFrames do grupo pela chave comum
        if dfs:
            merged_df = dfs[0]
            for df in dfs[1:]:
                
                
                
                merged_df = pd.merge(merged_df, df, on=key_column, how='inner')

            # Salva o arquivo final do grupo
            output_file = os.path.join(output_folder, f"treino_atributos_{part_number}_Nonlogged.parquet")
            merged_df.to_parquet(output_file, index=False)
            print(f"Arquivo final salvo em: {output_file}")
        else:
            print(f"Nenhum DataFrame válido para parte {part_number}.")

# Exemplo de uso
folder_path = './Dados/files/treino/Atributos/'
key_column = 'history'  # Substitua pelo nome da coluna chave
output_folder = './Dados/files/treino/Treino/'

merge_files_by_same_part(folder_path, key_column, output_folder)

text = "table_count_numberOfClicksHistory_by_page_treino_parte1_Nonlogged.parquet"
padrao = r"table_count_(.*?)_treino_parte1_Nonlogged.parquet"

# Procurar o padrão na string
resultado = re.search(padrao, text)
resultado.group(1)


df_treino_atributos = pd.read_parquet('./Dados/files/treino/Treino/treino_atributos_1_Nonlogged.parquet')