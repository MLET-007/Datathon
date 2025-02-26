"""
app.py

Flask API:
- Carrega content_model.pkl
- /recommend (POST) => fallback se history vazio, senão rank TF-IDF
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/content_model.pkl")
with open(MODEL_PATH, "rb") as f:
    content_model = pickle.load(f)

vectorizer = content_model["vectorizer"]
tfidf_matrix = content_model["tfidf_matrix"]
pages_array = content_model["pages"]
df_items = content_model["df_items"]
df_user = content_model.get("df_user", None) 

@app.route("/")
def home():
    return "API de Recomendação - Online!"

    """
    ///recommend - post
    
        {
        "userId": 124,
        "history": [999, 1000, 1010]
        }

    """
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = data.get("userId", None)
    history = data.get("history", [])

    # Fallback => top N recentes
    if not history:
        df_sorted = df_items.sort_values("issued", ascending=False)
        top_pages = df_sorted["page"].head(5).tolist()
        return jsonify({"userId": user_id, "recommendations": top_pages})

    # Montar user_profile => média TF-IDF das news do history
    user_indices = []
    for pid in history:
        idxs = np.where(pages_array == pid)[0]
        if len(idxs) > 0:
            user_indices.append(idxs[0])

    if not user_indices:
        df_sorted = df_items.sort_values("issued", ascending=False)
        top_pages = df_sorted["page"].head(5).tolist()
        return jsonify({"userId": user_id, "recommendations": top_pages})

    user_vectors = tfidf_matrix[user_indices]
    user_profile = user_vectors.mean(axis=0)

    sims = cosine_similarity(user_profile, tfidf_matrix)[0]
    ranked_idxs = np.argsort(sims)[::-1]
    read_set = set(history)
    recs = []
    for idx in ranked_idxs:
        p = pages_array[idx]
        if p not in read_set:
            recs.append(p)
        if len(recs) == 5:
            break

    return jsonify({"userId": user_id, "recommendations": recs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
