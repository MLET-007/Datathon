# Usa imagem base do Python 3.10
FROM python:3.10

# Define diretório de trabalho
WORKDIR /app

# Instala curl (se necessário)
RUN apt-get update && apt-get install -y curl

# Instala Poetry via script oficial
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ajusta variáveis de ambiente para Poetry
ENV POETRY_HOME="/root/.local"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copia arquivos de dependência ANTES de rodar poetry install
COPY pyproject.toml poetry.lock ./

# Instala dependências
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

# Copia o restante do projeto
COPY . .

# Expondo porta (caso seja FastAPI ou algo do gênero)
EXPOSE 8000

# Comando default (ajuste se sua app rodar de outra forma)
CMD ["python", "app.py"]
