import re
from turtle import right
import pandas as pd
import numpy as np
import glob

from scipy.stats import linregress
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
    
    
    


def calcular_inclinacao(grupo, date_column, count_column):
    # Converter dias para números (dias desde o primeiro dia)
    dias = (grupo[date_column] - grupo[date_column].min()).dt.days
    quantidades = grupo[count_column]
    
    # Calcular a regressão linear
    slope, _, _, _, _ = linregress(dias, quantidades)
    return slope

def create_rate_by_day(df, user_type, parte_treino):
    
    df_filtrado = df[df['userType'] == user_type]
    
    df_filtrado['dataHistory'] = df_filtrado['timestampHistory'].apply(processar_timestamps)

    df_filtrado['history'] = df_filtrado['history'].str.split(',')

    df_filtrado['dataHistory'] = df_filtrado['dataHistory'].str.split(',')

    df_explode = df_filtrado[['history', 'dataHistory']].apply(pd.Series.explode)

    df_explode['dataHistory'] = pd.to_datetime(df_explode['dataHistory'], format = "%d/%m/%Y %H:%M")

    df_explode['dataHistory_day'] = df_explode['dataHistory'].dt.date
    
    
    table_page_by_day = df_explode.groupby(['history', 'dataHistory_day']).size().reset_index(name='qnt_dias')
    
    table_page_by_day.sort_values(by = ['history', 'dataHistory_day'])
    
    table_crescimento = table_page_by_day.groupby('history').apply(calcular_inclinacao).reset_index()
    
    table_crescimento.columns = ['history', 'taxa_crescimento_users']
    
    return [table_crescimento, table_page_by_day]