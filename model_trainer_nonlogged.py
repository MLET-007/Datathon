
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
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
MODEL_DIR = os.path.join(BASE_DIR, "model")
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

def treinar_non_logged_svd(df_treino_nonlogged):
    """Treina um modelo SVD baseado em conteúdo para usuários não logados."""
    logger.info("Iniciando treinamento do modelo SVD para usuários não logados")
    
    logger.info("Convertendo 'issued' para datetime")
    df_treino_nonlogged['issued'] = pd.to_datetime(df_treino_nonlogged['issued'])
    
    ultimo_mes = df_treino_nonlogged['issued'].max() - pd.DateOffset(days=30)
    logger.info("Filtrando notícias recentes (últimos 30 dias a partir de %s)", ultimo_mes)
    df_recente = df_treino_nonlogged[df_treino_nonlogged['issued'] >= ultimo_mes]
    logger.info("Notícias recentes filtradas: %d", len(df_recente))
    
    logger.info("Convertendo 'numberOfClicksHistory' para numérico")
    df_treino_nonlogged["numberOfClicksHistory"] = pd.to_numeric(df_treino_nonlogged["numberOfClicksHistory"], errors='coerce')
    
    logger.info("Selecionando as 100 notícias mais populares por cliques")
    df_popular = df_treino_nonlogged.nlargest(100, 'numberOfClicksHistory')
    logger.info("Notícias populares selecionadas: %d", len(df_popular))
    
    logger.info("Concatenando notícias recentes e populares e removendo duplicatas")
    df_relevante = pd.concat([df_recente, df_popular]).drop_duplicates(subset=[col for col in df_recente.columns if col != 'top_5_palavras'])
    logger.info("Dataset relevante criado: %d notícias", len(df_relevante))
    
    logger.info("Criando coluna 'conteudo' com top_5_palavras, classificacao e agrupamento")
    df_relevante['conteudo'] = df_relevante['top_5_palavras'] + " " + df_relevante['classificacao'] + " " + df_relevante['agrupamento']
    df_relevante['conteudo'] = df_relevante['conteudo'].astype(str)
    
    logger.info("Vetorizando conteúdo com TF-IDF")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_relevante['conteudo'])
    tfidf_matrix = csr_matrix(tfidf_matrix)
    logger.info("Matriz TF-IDF gerada: %d x %d", tfidf_matrix.shape[0], tfidf_matrix.shape[1])
    
    logger.info("Aplicando TruncatedSVD com 100 componentes")
    svd = TruncatedSVD(n_components=100)
    item_factors = svd.fit_transform(tfidf_matrix)
    logger.info("Fatores latentes das notícias gerados: %d x %d", item_factors.shape[0], item_factors.shape[1])
    
    indices_noticias = df_relevante['history'].values
    logger.info("Índices de notícias mapeados: %d", len(indices_noticias))
    
    # Salvar modelos
    logger.info("Salvando modelos em %s", MODEL_DIR)
    joblib.dump(svd, os.path.join(MODEL_DIR, "svd_nonlogged.pkl"))
    np.save(os.path.join(MODEL_DIR, "item_factors.npy"), item_factors)
    np.save(os.path.join(MODEL_DIR, "indices_noticias.npy"), indices_noticias)
    logger.info("Modelos salvos com sucesso: svd_nonlogged.pkl, item_factors.npy, indices_noticias.npy")
    
    return svd, item_factors, indices_noticias

def treinar_modelo_nonlogged(df_treino, df_validacao, df_itens):
    """Orquestra o treinamento do modelo para usuários não logados."""
    logger.info("Iniciando o treinamento completo para usuários não logados")
    
    try:
        logger.info("Processando dados de treino")
        df_treino_nonlogged = tratamento_treino(df_treino, df_validacao, df_itens, "Non-Logged")
        
        logger.info("Treinando modelo SVD baseado em conteúdo")
        svd, item_factors, indices_noticias = treinar_non_logged_svd(df_treino_nonlogged)
        
        logger.info("Treinamento concluído com sucesso")
        return svd, item_factors, indices_noticias
    except Exception as e:
        logger.error("Erro durante o treinamento do modelo para não logados: %s", str(e))
        raise
    
##descomentar para executar direto 
if __name__ == "__main__":
    logger.info("Iniciando leitura do treino .parquet")
    df_treino = pd.read_parquet(os.path.join(PARQUET_DIR, "treino_final.parquet"))
    logger.info("Iniciando leitura do validacao .parquet")
    df_validacao = pd.read_csv(os.path.join(BASE_DIR, "files_csv", "validacao.csv"))
    logger.info("Iniciando leitura de validacao .parquet")
    df_itens = pd.read_parquet(os.path.join(PARQUET_DIR, "itens_finalv2.parquet"))
    logger.info("Iniciando treinamento do modelo não logado")
    treinar_modelo_nonlogged(df_treino, df_validacao, df_itens)    
    logger.info("Finalizado treinamento") 
