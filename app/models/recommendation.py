
from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    id: str
    url: str
    title: str
    conteudo: str

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]