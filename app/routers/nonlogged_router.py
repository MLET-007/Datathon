
from fastapi import APIRouter
from ..controllers.nonlogged_controller import get_nonlogged_recommendations
from ..models.recommendation import RecommendationResponse

nonlogged_router = APIRouter()

@nonlogged_router.get("/recommendations", response_model=RecommendationResponse)
def recommend_nonlogged(noticia_id: str = None, top_n: int = 5):
    recommendations = get_nonlogged_recommendations(noticia_id, top_n)
    return {"recommendations": recommendations}