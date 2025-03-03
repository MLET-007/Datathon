
import os
import pandas as pd
from ..recommender import carregar_modelos_logged, predict_logged

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")

def get_logged_recommendations(user_id: str, top_n: int = 5):
    # Carregar modelos
    svd, scaler, interaction_matrix = carregar_modelos_logged()
    
    # Carregar itens para mapear IDs para detalhes
    df_itens = pd.read_parquet(os.path.join(PARQUET_DIR, "itens_finalv2.parquet"))
    
    # Gerar recomendações
    recommended_ids = predict_logged(user_id, svd, scaler, interaction_matrix, top_n)
    
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