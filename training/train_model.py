import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    data_folder = "data"
    pattern = os.path.join(data_folder, "itens", "*.csv")
    parquet_output = os.path.join(data_folder, "itens_cache.parquet")
    chunk_size = 50000
    all_chunks = []

    # Verificar se já existe um .parquet salvo para evitar reler CSVs
    if os.path.exists(parquet_output):
        print(f"Carregando {parquet_output} em vez de CSVs")
        df_items = pd.read_parquet(parquet_output)
    else:
        # Ler CSVs de itens
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
        df_items.to_parquet(parquet_output)
        print(f"Salvo cache em {parquet_output}")

    print(f"df_items shape={df_items.shape}")

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df_items["Text"])

    # Carregar histórico de usuários, se existir
    user_data_file = "./data/user_data_enriched.parquet"
    df_user = pd.read_parquet(user_data_file) if os.path.exists(user_data_file) else None

    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    model_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "pages": df_items["page"].values,
        "df_items": df_items[["page", "url", "title", "issued"]].copy(),
        "df_user": df_user
    }
    with open("models/content_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("Salvo em models/content_model.pkl")

if __name__ == "__main__":
    main()
