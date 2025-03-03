
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
# RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev

COPY . .

RUN pip install poetry

RUN poetry config installer.max-workers 10
RUN poetry install --no-interaction --no-ansi

EXPOSE 8009
