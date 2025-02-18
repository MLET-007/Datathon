import pandas as pd
import numpy as np
import os
import glob
import re

from datetime import datetime


from Scripts.functions_to_help import * 
from Feature_Engineering.feature_engineering import *


# verificando items
df_itens_1 = pd.read_csv("./Dados/itens/itens/itens-parte1.csv")



df_itens_1.info()


df_itens_1['issued'] = pd.to_datetime(df_itens_1['issued'])

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 7 columns):
 #   Column    Non-Null Count   Dtype 
---  ------    --------------   ----- 
 0   page      100000 non-null  object
 1   url       100000 non-null  object
 2   issued    100000 non-null  object
 3   modified  100000 non-null  object
 4   title     100000 non-null  object
 5   body      100000 non-null  object
 6   caption   100000 non-null  object
dtypes: object(7)
memory usage: 5.3+ MB

'''

# quantas noticias um usuario logado acessa em media em um periodo x de tempo?
dt_train_1 = pd.read_csv("./Dados/files/treino/treino_parte1.csv")




# criando set df_train para testar a logica dos atributos
dt_train_1.info()
dt_train_1


'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 10 columns):
 #   Column                   Non-Null Count  Dtype 
---  ------                   --------------  ----- 
 0   userId                   100 non-null    object
 1   userType                 100 non-null    object
 2   historySize              100 non-null    int64 
 3   history                  100 non-null    object
 4   timestampHistory         100 non-null    object
 5   numberOfClicksHistory    100 non-null    object *
 6   timeOnPageHistory        100 non-null    object *
 7   scrollPercentageHistory  100 non-null    object 
 8   pageVisitsCountHistory   100 non-null    object *
 9   timestampHistory_new     100 non-null    object
dtypes: int64(1), object(9)
'''

BASE_PATH = "./Dados/files/treino/*.csv"

# criando atributos
columns_to_create_count_attr = ['numberOfClicksHistory', 'timeOnPageHistory', 'pageVisitsCountHistory']

columns_to_create_mean_attr = ['scrollPercentageHistory']

df_train_files = {}

for full_path in glob.glob(BASE_PATH):
    df = pd.read_csv(full_path)
    
    match = re.search(r'parte(\d+)', full_path)
    df_train_files[match.group(0)] = df


train_parts = list(df_train_files.keys())

for part in train_parts:

    # criando atributos diff mean time access
    #create_diff_time_attributes(df_train_files[part], user_type='Non-Logged', data_column_name="dataHistory", timestamp_column_name="timestampHistory", parte_treino = part)
    
    #create_count_userId_by_history(df_train_files[part], user_type='Non-Logged', parte_treino=part)
    
    ##create_count_in_period(df_train_files[part], user_type = 'Non-Logged', table_part = part)
    
    #create_count_days_by_page(df_train_files[part], user_type='Non-Logged', parte_treino=part)
    '''
    for column_name in columns_to_create_mean_attr:
        # criando atributos de media
        create_attributes_count_mean(df_train_files[part], user_type='Non-Logged', column_name=column_name, type_column='float', parte_treino = part)
    
    for column_name in columns_to_create_count_attr:
        
        # criando atributos de contagem
        create_attributes_count_mean(df_train_files[part], user_type='Non-Logged', column_name=column_name, type_column='int', parte_treino = part)
    '''  

# criando flag foi modificado
create_flag_was_modified()

