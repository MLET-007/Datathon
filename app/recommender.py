
import os
import pandas as pd
import numpy as np
import joblib

# Usar o diretório de trabalho atual (/app/) diretamente, já que estamos no container
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../model")  

def predict_logged(user_id, svd, scaler, interaction_matrix, top_n=5):
    user_index = interaction_matrix.index.get_loc(user_id)
    user_factors = scaler.transform(svd.transform(interaction_matrix.iloc[user_index].values.reshape(1, -1)))
    scores = user_factors.dot(svd.components_)
    top_indices = np.argsort(scores[0])[::-1][:top_n]
    recommended_news = interaction_matrix.columns[top_indices].tolist()
    return recommended_news

def recomendar_noticias_non_logged_svd(noticia_id, svd, item_factors, indices_noticias, top_n=5):
    if noticia_id not in indices_noticias:
        return "Notícia não encontrada"
    noticia_index = np.where(indices_noticias == noticia_id)[0][0]
    noticia_vector = item_factors[noticia_index].reshape(1, -1)
    scores = item_factors.dot(noticia_vector.T).flatten()
    top_indices = np.argsort(scores)[::-1]
    top_indices = [i for i in top_indices if i != noticia_index][:top_n]
    recommended_news = indices_noticias[top_indices].tolist()
    return recommended_news

def carregar_modelos_logged():
    svd = joblib.load(os.path.join(MODEL_DIR, "svd_logged.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_logged.pkl"))
    interaction_matrix = pd.read_pickle(os.path.join(MODEL_DIR, "interaction_matrix.pkl"))
    return svd, scaler, interaction_matrix

def carregar_modelos_nonlogged():
    svd = joblib.load(os.path.join(MODEL_DIR, "svd_nonlogged.pkl"))
    item_factors = np.load(os.path.join(MODEL_DIR, "item_factors.npy"))
    indices_noticias = np.load(os.path.join(MODEL_DIR, "indices_noticias.npy"))
    return svd, item_factors, indices_noticias