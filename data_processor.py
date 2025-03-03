
import glob
import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words("portuguese")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")
os.makedirs(PARQUET_DIR, exist_ok=True)

def carregar_e_unificar_itens(dir_itens):
    arquivos_csv = glob.glob(os.path.join(dir_itens, "itens*.csv"))
    df_noticias = pd.concat([pd.read_csv(arquivo) for arquivo in arquivos_csv], ignore_index=True)
    df_noticias.to_parquet(os.path.join(PARQUET_DIR, "itens_final.parquet"), index=False)
    return df_noticias

def categorizar_itens(df):
    abreviacoes_estados = {
        'sao-paulo': 'sp', 'rio-de-janeiro': 'rj', 'minas-gerais': 'mg', 'bahia': 'ba',
        'espirito-santo': 'es', 'parana': 'pr', 'santa-catarina': 'sc', 'goias': 'go',
        'pernambuco': 'pe', 'ceara': 'ce', 'mato-grosso': 'mt', 'mato-grosso-do-sul': 'ms',
        'pariba': 'pb', 'sergipe': 'se', 'alagoas': 'al', 'amazonas': 'am', 'acre': 'ac',
        'rondonia': 'ro', 'maranhao': 'ma', 'piaui': 'pi', 'rio-grande-do-sul': 'rs',
        'rio-grande-do-norte': 'rn', 'tocantins': 'to', 'distrito-federal': 'df'
    }
    estados_br = list(abreviacoes_estados.values())
    jornais_programas = ['jornal-nacional', 'jornal-hoje', 'bom-dia-brasil', 'jornal-da-globo',
                         'hora1', 'fantastico', 'profissao-reporter', 'globo-reporter',
                         'fato-ou-fake', 'resumo-do-dia', 'agenda-do-dia', 'globonews']
    temas = ['politica', 'economia', 'meio-ambiente', 'saude', 'ciencia', 'tecnologia']
    eventos = ['carnaval', 'dia-das-mulheres', 'consciencia-negra', 'rock-in-rio']
    curiosidades = ['adnet-na-cpi', 'que-meme-e-esse']
    especiais_blogs = ['especial-publicitario', 'especiais', 'blogs-e-colunas']
    entretenimento = ['musica', 'pop-arte', 'comida-di-buteco']

    df['classificacao'] = df['url'].str.extract(r'(?<=\.com\/)([^\/]+)')
    df['classificacao'] = df['classificacao'].replace(abreviacoes_estados)

    def categorizar(valor):
        if valor in estados_br: return 'Estados'
        elif valor in jornais_programas: return 'Jornal/Programa'
        elif valor in temas: return 'Temas'
        elif valor in eventos: return 'Eventos'
        elif valor in curiosidades: return 'Curiosidades'
        elif valor in especiais_blogs: return 'Especiais Blogs'
        elif valor in entretenimento: return 'Entreterimento'
        else: return 'Outro'

    df['agrupamento'] = df['classificacao'].apply(categorizar)
    return df

def vetorizar_itens(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['body'])
    words = tfidf_vectorizer.get_feature_names_out()
    top_words = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-5:][::-1]
        top_words.append([words[idx] for idx in top_indices])
    df['top_5_palavras'] = top_words
    df.to_parquet(os.path.join(PARQUET_DIR, "itens_finalv2.parquet"), index=False)
    return df

def carregar_e_unificar_treino(dir_treino):
    arquivos_csv = glob.glob(os.path.join(dir_treino, "treino*.csv"))
    df_treino = pd.concat([pd.read_csv(arquivo) for arquivo in arquivos_csv], ignore_index=True)
    df_treino.to_parquet(os.path.join(PARQUET_DIR, "treino_final.parquet"), index=False)
    return df_treino

def processar_dados(dir_itens, dir_treino):
    df_itens = carregar_e_unificar_itens(dir_itens)
    df_itens = categorizar_itens(df_itens)
    df_itens = vetorizar_itens(df_itens)
    df_treino = carregar_e_unificar_treino(dir_treino)
    return df_itens, df_treino