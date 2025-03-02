from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from datathon.controllers.predict_controller import PredictController

router = APIRouter()

class User(BaseModel):
    user_id: str | None = None

@router.post('/predict/news/')
def predict_news(user: User):
    try:
        user_id = user.user_id
        predict_controller = PredictController()
        return predict_controller.predict(user_id)
    except Exception as e:
        return {"error": str(e)}



