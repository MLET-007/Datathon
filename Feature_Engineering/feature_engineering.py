import pandas as pd
import numpy as np

from Scripts.functions_to_help import *

def create_attributes_non_logged(df, column_name, type_column, parte_treino, n_splits = 10):
    
    
    df_anonimo = df[df['userType'] == 'Non-Logged']
    
    
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
        
    table.to_parquet('./Dados/files/treino/Atributos/table_{0}_{1}_by_page_treino_{2}.parquet'.format(attribute_type, column_name, parte_treino))

    

def create_diff_time_attributes_non_logged(df, data_column_name, timestamp_column_name,parte_treino,  n_splits = 10):
        
    df_anonimo = df[df['userType'] == 'Non-Logged']
    
    df_anonimo[data_column_name] = df_anonimo[timestamp_column_name].apply(processar_timestamps)

    df_anonimo['history'] = df_anonimo['history'].str.split(',')
    df_anonimo[data_column_name] = df_anonimo[data_column_name].str.split(',')


    df_history_explode = df_anonimo[['history', data_column_name]].apply(pd.Series.explode)


    df_history_explode[data_column_name] = pd.to_datetime(df_history_explode[data_column_name], format = "%d/%m/%Y %H:%M")

    df_history_explode.sort_values(by = ['history', data_column_name], inplace = True)

    df_history_explode['diff_seconds'] = df_history_explode.groupby('history')[data_column_name].diff().dt.total_seconds()

    tables_media_diff = df_history_explode.groupby('history')['diff_seconds'].sum().reset_index()

    table = tables_media_diff.groupby('history')['diff_seconds'].mean().reset_index()
    
    table.to_parquet('./Dados/files/treino/Atributos/table_mean_diff_{0}_by_page_treino_{1}.parquet'.format(data_column_name, parte_treino))

    #return(table)

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