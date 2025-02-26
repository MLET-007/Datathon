"""
train_model.py

1) Ler CSVs de itens (data/itens/*.csv)
2) Gerar TF-IDF (title+body)
3) (Opcional) Ler user_data_enriched.parquet para pipeline híbrido
4) Salvar content_model.pkl
"""

import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    data_folder = "data"
    pattern = os.path.join(data_folder, "itens", "*.csv")
    chunk_size = 50000
    all_chunks = []

    # 1) Ler CSVs de itens
    files = glob.glob(pattern)
    if not files:
        print(f"ERRO: Nenhum CSV de itens em {pattern}")
        return

    for file_path in files:
        print(f"Lendo {file_path} em chunks={chunk_size}")
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk["Text"] = chunk["title"].fillna("") + " " + chunk["body"].fillna("")
            all_chunks.append(chunk)

    if not all_chunks:
        print("Nenhum dado de itens encontrado.")
        return

    df_items = pd.concat(all_chunks, ignore_index=True)
    print(f"df_items shape={df_items.shape}")

    # 2) TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df_items["Text"])

    # 3) (Opcional) Ler user_data_enriched
    user_data_file = "./data/user_data_enriched.parquet"
    df_user = None
    if os.path.exists(user_data_file):
        df_user = pd.read_parquet(user_data_file)
        print(f"Lido user_data_enriched: shape={df_user.shape}")

    # 4) Salvar content_model.pkl
    os.makedirs("models", exist_ok=True)
    model_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "pages": df_items["page"].values,  # Ajuste se a col. ID é "page"
        "df_items": df_items[["page","issued","modified","title","body"]].copy(),
        "df_user": df_user
    }
    with open("models/content_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("Salvo em models/content_model.pkl")

if __name__ == "__main__":
    main()
