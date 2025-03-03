
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import joblib
import logging

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Saída no console
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")
MODEL_DIR = os.path.join(BASE_DIR, "/app/model")
os.makedirs(MODEL_DIR, exist_ok=True)

def tratamento_treino(df_treino, df_vali, df_itens, user_type):
    """Pré-processa os dados de treino para um tipo de usuário específico."""
    logger.info("Iniciando tratamento dos dados de treino para user_type: %s", user_type)
    
    logger.info("Removendo coluna 'timestampHistory_new'")
    df_treino = df_treino.drop(columns=['timestampHistory_new'], axis=1)
    
    logger.info("Filtrando dados para user_type: %s", user_type)
    df_treino = df_treino[df_treino['userType'] == user_type]
    df_vali = df_vali[df_vali['userType'] == user_type]
    
    logger.info("Filtrando usuários presentes no conjunto de validação")
    df_treino2 = df_treino[df_treino['userId'].isin(df_vali['userId'])]
    logger.info("Usuarios filtrados: %d", len(df_treino2))
    
    cols_to_explode = ["history", "timestampHistory", "numberOfClicksHistory",
                       "timeOnPageHistory", "scrollPercentageHistory", "pageVisitsCountHistory"]
    logger.info("Explodindo colunas: %s", cols_to_explode)
    for col in cols_to_explode:
        df_treino2[col] = df_treino2[col].str.split(", ")
    df_treino_3 = df_treino2.explode(cols_to_explode, ignore_index=True)
    logger.info("Dados explodidos: %d linhas", len(df_treino_3))
    
    logger.info("Realizando merge com df_itens")
    df_treino_4 = df_treino_3.merge(df_itens, left_on='history', right_on='page', how='left')
    
    logger.info("Calculando frequências de classificação e filtrando valores com mais de 1 ocorrência")
    frequencias = df_treino_4['classificacao'].value_counts()
    valores_para_manter = frequencias[frequencias > 1].index
    df_treino_4 = df_treino_4[df_treino_4['classificacao'].isin(valores_para_manter)]
    
    logger.info("Removendo linhas com 'classificacao' nula")
    df_treino_4 = df_treino_4.dropna(subset=['classificacao'])
    logger.info("Dados após limpeza: %d linhas", len(df_treino_4))
    
    logger.info("Realizando split estratificado para reduzir a 2%% dos dados")
    _, df_treino_5 = train_test_split(df_treino_4, test_size=0.02, stratify=df_treino_4['classificacao'], random_state=42)
    logger.info("Dados finais após split: %d linhas", len(df_treino_5))
    
    return df_treino_5

def treinar_logged(df_treino_logged):
    """Treina um modelo SVD para usuários logados."""
    logger.info("Iniciando treinamento do modelo SVD para usuários logados")
    
    logger.info("Removendo colunas desnecessárias do DataFrame")
    df_total_logged = df_treino_logged.drop(columns=['timestampHistory', 'numberOfClicksHistory', 'pageVisitsCountHistory',
                                                     'page', 'issued', 'classificacao', 'agrupamento', 'top_5_palavras',
                                                     'historySize', 'userType'])
    
    logger.info("Convertendo colunas 'timeOnPageHistory' e 'scrollPercentageHistory' para numérico")
    df_total_logged["timeOnPageHistory"] = pd.to_numeric(df_total_logged["timeOnPageHistory"], errors='coerce')
    df_total_logged["scrollPercentageHistory"] = pd.to_numeric(df_total_logged["scrollPercentageHistory"], errors='coerce')
    
    max_time = df_total_logged["timeOnPageHistory"].max()
    logger.info("Normalizando tempo com base no máximo: %s", max_time)
    df_total_logged["normalized_time"] = df_total_logged["timeOnPageHistory"] / max_time
    df_total_logged["normalized_scroll"] = df_total_logged["scrollPercentageHistory"] / 100
    
    alpha = 0.5
    beta = 0.5
    logger.info("Calculando interaction_weight com alpha=%s e beta=%s", alpha, beta)
    df_total_logged["interaction_weight"] = (df_total_logged["normalized_time"] * alpha) + (df_total_logged["normalized_scroll"] * beta)
    
    logger.info("Criando matriz de interação (pivot)")
    interaction_matrix = df_total_logged.pivot(index="userId", columns="history", values="interaction_weight").fillna(0)
    logger.info("Matriz de interação criada: %d usuários x %d notícias", interaction_matrix.shape[0], interaction_matrix.shape[1])
    
    logger.info("Convertendo para matriz esparsa")
    sparse_matrix = csr_matrix(interaction_matrix.values)
    
    logger.info("Aplicando TruncatedSVD com 20 componentes")
    svd = TruncatedSVD(n_components=20)
    user_factors = svd.fit_transform(sparse_matrix)
    logger.info("Fatores latentes dos usuários gerados: %d x %d", user_factors.shape[0], user_factors.shape[1])
    
    logger.info("Normalizando fatores com StandardScaler")
    scaler = StandardScaler()
    user_factors = scaler.fit_transform(user_factors)
    
    # Salvar modelos
    logger.info("Salvando modelos em %s", MODEL_DIR)
    joblib.dump(svd, os.path.join(MODEL_DIR, "svd_logged.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_logged.pkl"))
    interaction_matrix.to_pickle(os.path.join(MODEL_DIR, "interaction_matrix.pkl"))
    logger.info("Modelos salvos com sucesso: svd_logged.pkl, scaler_logged.pkl, interaction_matrix.pkl")
    
    return svd, scaler, user_factors, interaction_matrix

def treinar_modelo_logged(df_treino, df_validacao, df_itens):
    """Orquestra o treinamento do modelo para usuários logados."""
    logger.info("Iniciando o treinamento completo para usuários logados")
    
    try:
        logger.info("Processando dados de treino")
        df_treino_logged = tratamento_treino(df_treino, df_validacao, df_itens, "Logged")
        
        logger.info("Treinando modelo SVD")
        svd, scaler, user_factors, interaction_matrix = treinar_logged(df_treino_logged)
        
        logger.info("Treinamento concluído com sucesso")
        return svd, scaler, user_factors, interaction_matrix
    except Exception as e:
        logger.error("Erro durante o treinamento do modelo para logados: %s", str(e))
        raise


## descomentar para executar direto    
# if __name__ == "__main__":
#     logger.info("Iniciando leitura do treino .parquet")
#     df_treino = pd.read_parquet(os.path.join(PARQUET_DIR, "treino_final.parquet"))
#     logger.info("Iniciando leitura do validacao .parquet")
#     df_validacao = pd.read_csv(os.path.join(BASE_DIR, "files_csv", "validacao.csv"))
#     logger.info("Iniciando leitura de validacao .parquet")
#     df_itens = pd.read_parquet(os.path.join(PARQUET_DIR, "itens_finalv2.parquet"))
#     logger.info("Iniciando treinamento do modelo logado")
#     treinar_modelo_logged(df_treino, df_validacao, df_itens)   
#     logger.info("Finalizado treinamento") 