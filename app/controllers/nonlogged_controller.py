
import os
import pandas as pd
from ..recommender import carregar_modelos_nonlogged, recomendar_noticias_non_logged_svd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")

def get_nonlogged_recommendations(noticia_id: str = None, top_n: int = 5):
    # Carregar modelos
    svd, item_factors, indices_noticias = carregar_modelos_nonlogged()
    
    # Carregar itens para mapear IDs para detalhes
    df_itens = pd.read_parquet(os.path.join(PARQUET_DIR, "itens_finalv2.parquet"))
    
    # Se não fornecer noticia_id, usar a primeira notícia como exemplo
    if not noticia_id or noticia_id not in indices_noticias:
        noticia_id = indices_noticias[0]
    
    # Gerar recomendações
    recommended_ids = recomendar_noticias_non_logged_svd(noticia_id, svd, item_factors, indices_noticias, top_n)
    # Mapear IDs para detalhes
    recommendations = []
    for rec_id in recommended_ids:
        item = df_itens[df_itens['page'] == rec_id].iloc[0]
        recommendations.append({
            "id": rec_id,
            "url": item['url'],
            "title": item['title'],
            "conteudo": item['body']
        })
    
    return recommendations