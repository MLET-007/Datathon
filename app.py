import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="API de Recomendação - G1")

# Carregar modelo
MODEL_PATH = os.getenv("MODEL_PATH", "models/content_model.pkl")
with open(MODEL_PATH, "rb") as f:
    content_model = pickle.load(f)

vectorizer = content_model["vectorizer"]
tfidf_matrix = content_model["tfidf_matrix"]
pages_array = content_model["pages"]
df_items = content_model["df_items"]  # Contém "page", "url", "title"
df_user = content_model.get("df_user", None)

# Indexar df_items para acesso rápido
df_items.set_index("page", inplace=True)

class RecommendationRequest(BaseModel):
    userId: int
    history: list[int] = []

@app.get("/")
def home():
    return {"message": "API de recomendação de notícias- G1!"}

@app.post("/recommend")
def recommend(data: RecommendationRequest):
    user_id = data.userId
    history = data.history

    # Função auxiliar para buscar título e URL
    def get_page_info(page_id):
        if page_id in df_items.index:
            row = df_items.loc[page_id]
            return {"page": page_id, "url": row["url"], "title": row["title"]}
        return {"page": page_id, "url": None, "title": None}

    # Fallback => top 5 mais recentes
    if not history:
        df_sorted = df_items.sort_values("issued", ascending=False)
        top_pages = df_sorted.head(5).index.tolist()
        recommendations = [get_page_info(p) for p in top_pages]
        return {"userId": user_id, "recommendations": recommendations}

    # Criar user_profile => média TF-IDF das páginas do history
    user_indices = [np.where(pages_array == pid)[0][0] for pid in history if len(np.where(pages_array == pid)[0]) > 0]

    if not user_indices:
        df_sorted = df_items.sort_values("issued", ascending=False)
        top_pages = df_sorted.head(5).index.tolist()
        recommendations = [get_page_info(p) for p in top_pages]
        return {"userId": user_id, "recommendations": recommendations}

    user_vectors = tfidf_matrix[user_indices]
    user_profile = user_vectors.mean(axis=0)

    sims = cosine_similarity(user_profile, tfidf_matrix)[0]
    ranked_idxs = np.argsort(sims)[::-1]

    read_set = set(history)
    top_pages = [pages_array[idx] for idx in ranked_idxs if pages_array[idx] not in read_set][:5]

    recommendations = [get_page_info(p) for p in top_pages]
    return {"userId": user_id, "recommendations": recommendations}
