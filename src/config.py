from pathlib import Path
import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"


# ============================================================
# Dataset
# ============================================================

PLANTSEG_DIR = RAW_DATA_DIR / "PlantSeg"
IMAGE_DIR = PLANTSEG_DIR / "images"
ANNOTATION_DIR = PLANTSEG_DIR / "annotations"

CLEAN_METADATA_PATH = (
    PROCESSED_DATA_DIR / "agrisense_metadata_clean.csv"
)


# ============================================================
# Model checkpoints
# ============================================================

CLASSIFIER_CHECKPOINT = (
    CHECKPOINT_DIR / "resnet18_extended_finetuned_latest.pth"
)

SEGMENTATION_CHECKPOINT = (
    CHECKPOINT_DIR / "unet_segmentation_best.pth"
)


# ============================================================
# Runtime
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Image configuration
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 115

SEGMENTATION_THRESHOLD = 0.4


# ============================================================
# Disease classes
# ============================================================

import pandas as pd


def load_disease_classes():
    """
    Load disease class names in the same sorted order
    used during model training.
    """

    df = pd.read_csv(CLEAN_METADATA_PATH)

    disease_classes = sorted(
        df["Disease"].dropna().unique()
    )

    return disease_classes