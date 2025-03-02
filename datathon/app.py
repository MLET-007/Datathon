from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from datathon.routers import apis


app = FastAPI()
app.include_router(apis.router)
# app.mount('/mkdocs', StaticFiles(directory='site', html=True), name='mkdocs')
