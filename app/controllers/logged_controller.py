
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


# fe92fb8f410d4471184fd3e1d6b70a1ab43bd97e27ccc08f0cb320489595261c
# ff91b6c9615dc9f18893be73c0b4d7dd335ca09c0d8c7326b2e81e468b412003
# fceed15715d42b934667ac733c5322bec5a99718f231e432d521fb7ad02f780e
# fff72f1e3d25c5027177850a217129bf088902d4ee1103f1aa3bb48c68d5ca0a