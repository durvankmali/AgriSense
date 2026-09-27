from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from app.schemas import PredictionResponse
from src.inference.pipeline import AgriSensePipeline


app = FastAPI(
    title="AgriSense API",
    description="Crop disease classification, localization, and explainability API.",
    version="1.0.0",
)


# Load the ML pipeline once when the API starts.
pipeline = AgriSensePipeline()


@app.get("/")
def root():
    return {
        "message": "AgriSense API is running.",
        "status": "ok",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):
    """
    Run AgriSense inference on an uploaded plant image.
    """

    image_bytes = await file.read()

    temporary_path = (
        Path("temp_uploaded_image")
        / file.filename
    )

    temporary_path.parent.mkdir(
        exist_ok=True
    )

    temporary_path.write_bytes(image_bytes)

    result = pipeline.predict(
        temporary_path
    )

    temporary_path.unlink(
        missing_ok=True
    )

    return {
        "disease": result["disease"],
        "confidence": result["confidence"],
        "class_index": result["class_index"],
        "mask_coverage": result["mask_coverage"],
    }