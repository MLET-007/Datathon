
import os
from data_processor import processar_dados
from model_trainer_logged import treinar_modelo_logged
from model_trainer_nonlogged import treinar_modelo_nonlogged
from app.recommender import carregar_modelos_logged, carregar_modelos_nonlogged, predict_logged, recomendar_noticias_non_logged_svd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "files_csv")
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")

def main():
    # Processar dados
    # df_itens, df_treino = processar_dados(CSV_DIR, CSV_DIR)
    # df_itens_exp = df_itens.drop(columns=['url', 'modified', 'title', 'body', 'caption'])
    # df_itens_exp['issued'] = pd.to_datetime(df_itens_exp['issued']).dt.date
    # df_validacao = pd.read_csv(os.path.join(CSV_DIR, "validacao.csv"))

    # Treinar modelos
    svd_logged, scaler_logged, user_factors, interaction_matrix = treinar_modelo_logged(df_treino, df_validacao, df_itens_exp)
    svd_nonlogged, item_factors, indices_noticias = treinar_modelo_nonlogged(df_treino, df_validacao, df_itens_exp)

    # Carregar modelos e fazer recomendações
    svd_logged, scaler_logged, interaction_matrix = carregar_modelos_logged()
    svd_nonlogged, item_factors, indices_noticias = carregar_modelos_nonlogged()

    # Exemplo de recomendação para logados
    user_id_exemplo = interaction_matrix.index[0]
    recomendacoes_logged = predict_logged(user_id_exemplo, svd_logged, scaler_logged, interaction_matrix)
    print(f"Recomendações para usuário logado {user_id_exemplo}: {recomendacoes_logged}")

    # Exemplo de recomendação para não logados
    noticia_exemplo = indices_noticias[0]
    recomendacoes_nonlogged = recomendar_noticias_non_logged_svd(noticia_exemplo, svd_nonlogged, item_factors, indices_noticias)
    print(f"Recomendações para notícia {noticia_exemplo}: {recomendacoes_nonlogged}")

# if __name__ == "__main__":
#     main()