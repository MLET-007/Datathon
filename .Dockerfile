FROM python:3.10-slim

WORKDIR /app

# Instalar Poetry
RUN pip install --no-cache-dir poetry==1.6.1

# Copia pyproject / poetry.lock primeiro (cache)
COPY pyproject.toml poetry.lock /app/

# Instalar dependências
RUN poetry install --no-root --no-interaction --no-ansi

# Copiar todo o projeto
COPY . /app

# Expor a porta da API (Flask)
EXPOSE 5000

# Comando final: roda Flask usando Gunicorn
CMD ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
