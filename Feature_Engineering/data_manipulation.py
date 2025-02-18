import os
import pandas as pd
from collections import defaultdict

# unificando datasets de treino

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