from pydantic import BaseModel


class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    class_index: int
    mask_coverage: float