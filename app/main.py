import base64
import io

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from app.schemas import PredictionResponse
from src.inference.pipeline import AgriSensePipeline


app = FastAPI(
    title="AgriSense API",
    description=(
        "Crop disease classification, localization, "
        "and explainability API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML pipeline once when the API starts.
pipeline = AgriSensePipeline()


def array_to_base64_png(array):
    """
    Convert a normalized 2D NumPy array into
    a grayscale PNG encoded as Base64.
    """

    array = (array * 255).clip(0, 255).astype("uint8")

    image = Image.fromarray(
        array,
        mode="L",
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


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
async def predict(
    file: UploadFile = File(...),
):
    """
    Run AgriSense inference on an uploaded plant image.
    """

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = pipeline.predict_from_image(
        image
    )

    segmentation_mask = (
        result["segmentation_mask"]
        .squeeze()
    )

    gradcam = result["gradcam"]

    return {
        "disease": result["disease"],
        "confidence": result["confidence"],
        "class_index": result["class_index"],
        "mask_coverage": result["mask_coverage"],
        "segmentation_mask": array_to_base64_png(
            segmentation_mask
        ),
        "gradcam": array_to_base64_png(
            gradcam
        ),
    }