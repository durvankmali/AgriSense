import base64
import io

import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.schemas import PredictionResponse
from src.inference.pipeline import AgriSensePipeline


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


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
        "https://agrisense-rcwy.onrender.com",
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

    array = (
        (array * 255)
        .clip(0, 255)
        .astype("uint8")
    )

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
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "device": str(pipeline.device),
        "num_classes": len(pipeline.class_names),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
):
    print("[PREDICT] request received", flush=True)
    """
    Run AgriSense inference on an uploaded plant image.
    """

    # Basic content-type validation.
    if file.content_type not in {
        "image/jpeg",
        "image/png",
    }:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported.",
        )

    image_bytes = await file.read()

    print(
        f"[PREDICT] file received | size={len(image_bytes)} bytes",
        flush=True,
    )

    # File-size protection.
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Maximum size is 10 MB.",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # Verify that the uploaded bytes are actually
    # a readable image.
    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.verify()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be processed.",
        )

    try:
        result = pipeline.predict_from_image(
            image
        )

    except Exception as exc:
        print(f"[PREDICT] pipeline error: {exc}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
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
        "uncertain": result["uncertain"],
        "mask_coverage": result["mask_coverage"],
        "segmentation_mask": array_to_base64_png(
            segmentation_mask
        ),
        "gradcam": array_to_base64_png(
            gradcam
        ),
    }