
from fastapi import APIRouter, HTTPException
from ..controllers.logged_controller import get_logged_recommendations
from ..models.recommendation import RecommendationResponse

logged_router = APIRouter()

@logged_router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommend_logged(user_id: str, top_n: int = 5):
    try:
        recommendations = get_logged_recommendations(user_id, top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao gerar recomendações: {str(e)}")