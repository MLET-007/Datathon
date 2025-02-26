# Datathon = Recomendação de notícias G1 - Globo 

Este projeto demonstra uma solução **content-based** para o desafio do Datathon,
incluindo:
- Leitura de **itens** (notícias) em chunks em arquivos - CSV.
- Geração de **TF-IDF** e salvamento de um modelo (`content_model.pkl`).
- **API Flask** com fallback para anônimos e recomendação content-based para usuários logados.
- **Docker** para containerização.
- **Poetry** para gerenciamento de dependências.

## Estrutura
