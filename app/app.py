
from fastapi import FastAPI
from app.routers.logged_router import logged_router
from app.routers.nonlogged_router import nonlogged_router

app = FastAPI(title="Datathon G1 Recomendações de Notícias - API")

@app.get("/")
def read_root():
    return {"message": "Bem-vindos ao Datathon G1 Recomendações de Notícias - API"}

app.include_router(logged_router, prefix="/user", tags=["Usuários Logados"])
app.include_router(nonlogged_router, prefix="/news", tags=["Usuários Não Logados"])