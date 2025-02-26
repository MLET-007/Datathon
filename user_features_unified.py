"""
unified_datathon.py

Objetivo:
 - Manter todas as funções do pipeline antigo em um só arquivo.

"""

import os
import re
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

###########################
# Funções auxiliares
###########################

def converter_timestamp(timestamp):
    """ Converte milissegundos em '%d/%m/%Y %H:%M'. """
    return datetime.fromtimestamp(int(timestamp)/1000).strftime('%d/%m/%Y %H:%M')

def processar_timestamps(linha):
    """Ex.: '1658773550000,1658773555000' -> '01/07/2022 10:00;01/07/2022 10:05'."""
    if not isinstance(linha, str):
        return ""
    parts = linha.split(',')
    datas_formatadas = [converter_timestamp(ts.strip()) for ts in parts if ts.strip()]
    return ';'.join(datas_formatadas)

def contar_valores(string):
    """Conta quantos valores separados por vírgula existem em 'string'."""
    if not isinstance(string, str):
        return 0
    return len(string.split(','))

###########################
# Funções principais do pipeline
###########################

def create_attributes_count_mean(df, user_type, column_name, type_column, dir):
    """
    Gera ex. 'table_count_timeOnPageHistory_by_page_treino_final_{userType}.parquet'.
    Se type_column='int' => soma, 'float' => média.
    """
    df_anonimo = df[df['userType'] == user_type].copy()

    # Converter strings para listas
    df_anonimo['history'] = df_anonimo['history'].astype(str).str.split(',')
    df_anonimo[column_name] = df_anonimo[column_name].astype(str).str.split(',')

    # Resetar índice para evitar duplicados
    df_anonimo.reset_index(drop=True, inplace=True)

    # Explodir
    df_anonimo = df_anonimo.explode('history', ignore_index=True)
    df_anonimo = df_anonimo.explode(column_name, ignore_index=True)

    # Converter tipo e agrupar
    if type_column == 'int':
        df_anonimo[column_name] = df_anonimo[column_name].astype(int, errors='ignore')
        table = df_anonimo.groupby('history')[column_name].sum().reset_index(name='total')
    else:
        df_anonimo[column_name] = df_anonimo[column_name].astype(float, errors='ignore')
        table = df_anonimo.groupby('history')[column_name].mean().reset_index(name='total')

    attribute_type = 'count'  # conforme seu script
    os.makedirs(dir, exist_ok=True)

    fname = f"{dir}/table_{attribute_type}_{column_name}_by_page_treino_final_{str(user_type).replace('-', '')}.parquet"
    table.to_parquet(fname, index=False)
    print(f"[create_attributes_count_mean] Salvo: {fname}")

def create_diff_time_attributes(df, user_type, data_column_name, timestamp_column_name, dir):
    """
    Gera table_mean_diff_{data_column_name}_by_page_treino_final_{userType}.parquet
    """
    # Filtrar
    df_anonimo = df[df['userType'] == user_type].copy()

    # Converter timestamp -> datas formatadas
    df_anonimo[data_column_name] = df_anonimo[timestamp_column_name].apply(processar_timestamps)

    # Transformar colunas em listas
    df_anonimo['history'] = df_anonimo['history'].astype(str).str.split(',')
    df_anonimo[data_column_name] = df_anonimo[data_column_name].astype(str).str.split(',')

    # Resetar índice para evitar duplicatas
    df_anonimo.reset_index(drop=True, inplace=True)

    # Explodir colunas
    df_anonimo = df_anonimo.explode('history', ignore_index=True)
    df_anonimo = df_anonimo.explode(data_column_name, ignore_index=True)

    # Converter a coluna data_column_name em datetime
    df_anonimo[data_column_name] = pd.to_datetime(df_anonimo[data_column_name], format="%d/%m/%Y %H:%M", errors='coerce')

    # Ordenar e calcular diffs
    df_anonimo.sort_values(by=['history', data_column_name], inplace=True)
    df_anonimo['diff_seconds'] = df_anonimo.groupby('history')[data_column_name].diff().dt.total_seconds()

    # Agrupar soma e tirar a média
    tables_media_diff = df_anonimo.groupby('history')['diff_seconds'].sum().reset_index()
    table = tables_media_diff.groupby('history')['diff_seconds'].mean().reset_index()

    # Salvar .parquet
    os.makedirs(dir, exist_ok=True)
    fname = f"{dir}/table_mean_diff_{data_column_name}_by_page_treino_final_{str(user_type).replace('-', '')}.parquet"
    table.to_parquet(fname, index=False)
    print(f"[create_diff_time_attributes] Salvo: {fname}")

def create_count_userId_by_history(df, user_type, dir):
    """
    Gera table_count_userId_by_page_treino_final_{userType}.parquet
    """
    df_filtrado = df[df['userType'] == user_type].copy()
    df_filtrado['history'] = df_filtrado['history'].astype(str).str.split(',')

    # Resetar índice
    df_filtrado.reset_index(drop=True, inplace=True)

    # Explodir
    df_filtrado = df_filtrado.explode('history', ignore_index=True)

    table = df_filtrado.groupby('history')['userId'].nunique().reset_index(name='count_userId_by_page')

    os.makedirs(dir, exist_ok=True)
    fname = f"{dir}/table_count_userId_by_page_treino_final_{str(user_type).replace('-', '')}.parquet"
    table.to_parquet(fname, index=False)
    print(f"[create_count_userId_by_history] Salvo: {fname}")

def create_flag_was_modified(df_itens, parquet_userid, dir):
    """
    Ex.: gera table_flag_is_modified_by_page_treino_final_Nonlogged.parquet
    Necessita df_itens c/ col 'issued','modified','page'.
    Necessita 'parquet_userid' (dfUser?).
    """
    df_itens['is_modified'] = np.where(df_itens['issued'] == df_itens['modified'], 0, 1)

    df_left_join = parquet_userid.merge(df_itens[['page','is_modified']], left_on='history', right_on='page', how='left')
    df_left_join_grouped = df_left_join.groupby('history', as_index=False).first()

    os.makedirs(dir, exist_ok=True)
    fname = f"{dir}/table_flag_is_modified_by_page_treino_final_Nonlogged.parquet"
    df_left_join_grouped[['history','is_modified']].to_parquet(fname, index=False)
    print(f"[create_flag_was_modified] Salvo: {fname}")

def create_count_days_by_page(df, user_type, dir):
    """
    Gera table_count_days_by_page_treino_final_{userType}.parquet
    """
    df_filtrado = df[df['userType'] == user_type].copy()
    df_filtrado['dataHistory'] = df_filtrado['timestampHistory'].apply(processar_timestamps)

    df_filtrado['history'] = df_filtrado['history'].astype(str).str.split(',')
    df_filtrado['dataHistory'] = df_filtrado['dataHistory'].astype(str).str.split(',')

    # Resetar e explode
    df_filtrado.reset_index(drop=True, inplace=True)
    df_filtrado = df_filtrado.explode('history', ignore_index=True)
    df_filtrado = df_filtrado.explode('dataHistory', ignore_index=True)

    df_filtrado['dataHistory'] = pd.to_datetime(df_filtrado['dataHistory'], format="%d/%m/%Y %H:%M", errors='coerce')
    df_filtrado['dataHistory_day'] = df_filtrado['dataHistory'].dt.date

    table = df_filtrado.groupby('history')['dataHistory_day'].nunique().reset_index(name='qnt_dias')

    os.makedirs(dir, exist_ok=True)
    fname = f"{dir}/table_count_days_by_page_treino_final_{str(user_type).replace('-', '')}.parquet"
    table.to_parquet(fname, index=False)
    print(f"[create_count_days_by_page] Salvo: {fname}")

def create_count_in_period(df, df2, user_type, dir, flag_minutes):
    """
    Gera ex. table_count_page_users_in_15_minutes_treino_final_NonLogged.parquet
         ou table_count_page_users_in_first_day_treino_final_NonLogged.parquet
    """
    df_filtrado = df[df['userType'] == user_type].copy()
    df_filtrado['dataHistory'] = df_filtrado['timestampHistory'].apply(processar_timestamps)

    df_filtrado['history'] = df_filtrado['history'].astype(str).str.split(',')
    df_filtrado['dataHistory'] = df_filtrado['dataHistory'].astype(str).str.split(',')

    df_filtrado.reset_index(drop=True, inplace=True)
    df_filtrado = df_filtrado.explode('history', ignore_index=True)
    df_filtrado = df_filtrado.explode('dataHistory', ignore_index=True)

    df_filtrado['dataHistory'] = pd.to_datetime(df_filtrado['dataHistory'], format="%d/%m/%Y %H:%M", errors='coerce')

    df2['issued'] = pd.to_datetime(df2['issued'], errors='coerce')
    df2['issued_sem_fuso'] = df2['issued'].dt.tz_localize(None)

    df_left_join = df_filtrado.merge(df2[['page','issued_sem_fuso']], left_on='history', right_on='page', how='left')

    if flag_minutes:
        df_filtered = df_left_join[
            (df_left_join['dataHistory'] >= df_left_join['issued_sem_fuso']) &
            (df_left_join['dataHistory'] <= df_left_join['issued_sem_fuso'] + timedelta(minutes=15))
        ]
        table = df_filtered.groupby('history').size().reset_index(name='count_in_15_minutes')
        table.sort_values(by='count_in_15_minutes', ascending=False, inplace=True)

        os.makedirs(dir, exist_ok=True)
        fname = f"{dir}/table_count_page_users_in_15_minutes_treino_final_{str(user_type).replace('-', '')}.parquet"
        table.to_parquet(fname, index=False)
        print(f"[create_count_in_period] 15 minutes => Salvo: {fname}")
    else:
        df_filtered = df_left_join[
            (df_left_join['dataHistory'] >= df_left_join['issued_sem_fuso']) &
            (df_left_join['dataHistory'] <= df_left_join['issued_sem_fuso'] + timedelta(days=1))
        ]
        table = df_filtered.groupby('history').size().reset_index(name='count_in_first_day')
        table.sort_values(by='count_in_first_day', ascending=False, inplace=True)

        os.makedirs(dir, exist_ok=True)
        fname = f"{dir}/table_count_page_users_in_first_day_treino_final_{str(user_type).replace('-', '')}.parquet"
        table.to_parquet(fname, index=False)
        print(f"[create_count_in_period] 1 day => Salvo: {fname}")

###########################
# Funções de merge
###########################

def merge_files_by_same_part(folder_path, key_column, output_folder):
    """
    Exemplo de func p/ agrupar .parquet contendo 'final' no nome e gerar
    'treino_atributos_final_Nonlogged.parquet'.
    """
    part_files = []
    for filename in os.listdir(folder_path):
        if "final" in filename:
            part_files.append(filename)
            break

    if not part_files:
        print("Nenhum arquivo com 'final' encontrado.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for files in part_files:
        dfs = []
        for fn in files:
            file_path = os.path.join(folder_path, fn)
            try:
                df = pd.read_parquet(file_path)
                print("[merge_files_by_same_part] Lendo:", fn)
                pat = r"table_count_(.*?)_treino_final_Nonlogged.parquet"
                ccname = re.search(pat, fn)
                if ccname:
                    colname = ccname.group(1)
                    df.rename(columns={'total': colname}, inplace=True)
                dfs.append(df)
            except Exception as e:
                print(f"Erro ao ler {fn}: {e}")

        if dfs:
            merged_df = dfs[0]
            for df2 in dfs[1:]:
                merged_df = pd.merge(merged_df, df2, on=key_column, how='inner')

            out_path = os.path.join(output_folder, "treino_atributos_final_Nonlogged.parquet")
            merged_df.to_parquet(out_path, index=False)
            print(f"Arquivo final salvo em {out_path}")
        else:
            print("Nenhum DataFrame válido para parte final.")


###########################
# Exemplo de pipeline main()
###########################

def main():
    """
    Exemplo de como usar as funções com a estrutura de pastas estabelecida:
      data/files/treino/*.csv -> df_treino_final
      data/itens/*.csv -> df_itens_final
      data/resultados/ -> saídas .parquet
    """
    dir_arq_resultado = "data/resultados"  
    os.makedirs(dir_arq_resultado, exist_ok=True)

    # 1) Ler CSV de treino
    all_csv_treino = glob.glob("data/files/treino/*.csv")
    if not all_csv_treino:
        print("Nenhum CSV em data/files/treino/")
        return

    df_treino_list = []
    for csvf in all_csv_treino:
        temp = pd.read_csv(csvf)
        df_treino_list.append(temp)
    df_treino_final = pd.concat(df_treino_list, ignore_index=True)
    print("df_treino_final shape=", df_treino_final.shape)

    # 2) Ler CSV de itens
    all_csv_itens = glob.glob("data/itens/*.csv")
    if not all_csv_itens:
        print("Nenhum CSV em data/itens/")
        return

    df_itens_list = []
    for csvf in all_csv_itens:
        temp = pd.read_csv(csvf)
        df_itens_list.append(temp)
    df_itens_final = pd.concat(df_itens_list, ignore_index=True)
    print("df_itens_final shape=", df_itens_final.shape)

    # 3) Exemplo: chamar create_diff_time_attributes
    create_diff_time_attributes(
        df_treino_final,
        user_type="Non-Logged",
        data_column_name="dataHistory",
        timestamp_column_name="timestampHistory",
        dir=dir_arq_resultado
    )

    # 4) Exemplo: create_count_userId_by_history
    create_count_userId_by_history(
        df_treino_final,
        user_type="Non-Logged",
        dir=dir_arq_resultado
    )

    # E assim por diante, chamando as outras. Ajuste conforme necessidade:
    # create_count_in_period, create_count_days_by_page, etc.

    print("Processamento concluído.")


if __name__ == "__main__":
    main()
