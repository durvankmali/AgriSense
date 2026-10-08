# AgriSense — Crop Health Intelligence Platform

AgriSense is an end-to-end computer vision platform for crop disease identification and disease-region localization from plant images.

The project combines a fine-tuned ResNet-18 classifier, a lightweight U-Net segmentation model, Grad-CAM explainability, and a FastAPI inference backend with a web-based frontend.

---

## Project Overview

### Problem

Plant disease identification from images is challenging because real-world images can contain different resolutions, aspect ratios, backgrounds, lighting conditions, plant species, and disease appearances.

AgriSense addresses this by building a complete image-based pipeline that:

1. Identifies the disease from a plant image.
2. Localizes the predicted disease region.
3. Provides model confidence.
4. Generates a Grad-CAM explanation showing image regions influencing the classification.
5. Exposes the complete inference pipeline through an API and web interface.

---

## Key Features

* Multi-class plant disease classification across **115 disease classes**
* Fine-tuned **ResNet-18** classifier
* Lightweight **U-Net** for disease-region segmentation
* **Grad-CAM** visual explanations
* Confidence-based uncertainty indication
* Aspect-ratio-preserving image preprocessing
* Duplicate and conflicting-label detection
* Cross-split leakage control
* FastAPI backend
* Interactive browser frontend
* Dockerized deployment
* Git/GitHub version-controlled workflow

---

## System Architecture

```text
                         Uploaded Plant Image
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             Image Preprocessing         Image Preprocessing
                    |                           |
                    v                           v
             ResNet-18 Classifier       U-Net Segmentation
                    |                           |
                    v                           v
             Disease Prediction          Disease Region
             + Confidence                + Mask Coverage
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                         Grad-CAM Explanation
                                  |
                                  v
                    Explainable Prediction Result
```

The production inference pipeline returns:

* predicted disease
* confidence score
* class index
* uncertainty flag
* segmentation mask
* mask coverage
* Grad-CAM visualization

---

## Dataset

AgriSense uses the **PlantSeg** dataset.

### Original Dataset

| Property                | Value |
| ----------------------- | ----: |
| Images                  | 7,774 |
| Plants                  |    34 |
| Diseases                |   115 |
| Missing metadata values |     0 |

The original dataset contained duplicate image contents and conflicting disease labels, so data-quality checks were performed before model development.

### Clean Dataset

The final modeling dataset contains:

| Property                       | Value |
| ------------------------------ | ----: |
| Records                        | 7,399 |
| Unique image contents          | 7,399 |
| Training                       | 5,084 |
| Validation                     |   811 |
| Test                           | 1,504 |
| Duplicate image contents       |     0 |
| Cross-split duplicate contents |     0 |

### Data Quality Processing

The cleaning workflow included:

* duplicate-content detection using image hashes
* conflicting-label detection
* removal of records associated with conflicting image contents
* cross-split leakage checks
* metadata and filesystem consistency checks
* image integrity checks
* image-size and aspect-ratio analysis
* plant-label validation

A total of **98 conflicting image contents**, corresponding to **205 records**, were removed.

The final dataset contains no duplicate image contents and no cross-split duplicate contents.

---

## Image Preprocessing

The final classifier and production inference pipeline use:

* RGB conversion
* resize while preserving aspect ratio
* padding to **224 × 224**
* black padding
* BILINEAR interpolation
* ImageNet normalization

Training additionally uses:

* random horizontal flipping
* random rotation up to 10 degrees

This approach avoids indiscriminate cropping of heterogeneous plant images.

---

# Disease Classification

## Model

**ResNet-18** was used as the classification model and fine-tuned for the 115 disease classes.

### Training Configuration

* Trainable layers: `layer4` and final classification layer
* Optimizer: Adam
* Learning rate: `0.0001`
* Scheduler: ReduceLROnPlateau
* Early stopping
* Batch size: 16
* Training epochs: 10

The final checkpoint is:

```text
models/checkpoints/resnet18_extended_finetuned_latest.pth
```

## Model Development

The classification development process included multiple experiments.

| Model / Experiment         |   Accuracy |   Macro F1 | Decision           |
| -------------------------- | ---------: | ---------: | ------------------ |
| Baseline ResNet-18         |     44.02% |     33.99% | Starting benchmark |
| Class-weighted loss        |     42.09% |     33.68% | Rejected           |
| Initial fine-tuning        |     59.44% |     46.50% | Improved           |
| Final extended fine-tuning | **62.70%** | **51.48%** | Final model        |

The baseline was used as a reference point rather than as the final selected model.

The class-weighted experiment was evaluated separately and performed worse than the other approaches, so it was not used in the final classifier.

## Final Test Performance

The authoritative test evaluation after aligning production preprocessing with the training and evaluation pipeline:

| Metric         | Final Model |
| -------------- | ----------: |
| Accuracy       |  **62.70%** |
| Macro F1       |  **51.48%** |
| Weighted F1    |  **61.24%** |
| Top-5 Accuracy |  **88.50%** |

### Improvement Over Baseline

| Metric      | Baseline | Final Model | Improvement |
| ----------- | -------: | ----------: | ----------: |
| Accuracy    |   44.02% |      62.70% |   +18.68 pp |
| Macro F1    |   33.99% |      51.48% |   +17.49 pp |
| Weighted F1 |   42.37% |      61.24% |   +18.87 pp |

The final model substantially improved over the baseline while retaining the original multi-class problem formulation.

---

# Class Imbalance Experiment

A class-weighted training experiment was evaluated to examine whether weighting classes differently would improve performance on the imbalanced dataset.

Results:

* Accuracy: 42.09%
* Macro F1: 33.68%
* Weighted F1: 41.62%

Because this performed worse than the other training approaches, class weighting was not used in the final classifier.

---

# Error Analysis

The final test set contained **561 incorrect predictions**.

Of these:

* Same-plant errors: 153 (27.27%)
* Cross-plant errors: 408 (72.73%)

This indicates that a large proportion of errors occur across different plant categories rather than only between diseases affecting the same plant.

The relationship between class support and F1 score was moderately positive:

* Pearson correlation: 0.468
* Spearman correlation: 0.488

This analysis was used to understand model behavior beyond overall accuracy.

---

# Disease-Region Segmentation

AgriSense uses a lightweight **U-Net** to estimate the disease region in an image.

## Model

* Encoder channels: 3 → 32 → 64 → 128
* Bottleneck: 256
* Decoder with skip connections
* Output: single-channel disease probability map
* Parameters: 1,928,417
* Loss: BCE + Dice
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 16
* Training epochs: 5

Checkpoint:

```text
models/checkpoints/unet_segmentation_best.pth
```

## Final Test Performance

At the calibrated threshold of **0.4**:

| Metric         |      Score |
| -------------- | ---------: |
| Dice           | **0.5233** |
| IoU            | **0.3971** |
| Pixel Accuracy | **0.8198** |

The threshold was selected using validation-set Dice performance.

The segmentation model provides useful disease-region localization, although small and thin lesions can be missed and some predictions extend beyond the annotated region.

**Mask coverage should not be interpreted as disease severity.**

---

# Explainability with Grad-CAM

Grad-CAM is applied to the final convolutional block of ResNet-18.

The generated heatmap provides a coarse visual indication of image regions that influenced the predicted class.

A qualitative sample-level check showed that Grad-CAM can overlap meaningfully with annotated disease regions, but Grad-CAM is treated as an explanation mechanism rather than proof that the classifier has learned the true disease boundary.

---

# Confidence and Uncertainty

The production classifier uses a confidence threshold of **0.65**.

Predictions below this threshold are marked:

```text
uncertain = true
```

### Validation Analysis

* Mean confidence: 66.13%
* Validation accuracy: 62.15%
* Predictions accepted at 0.65: 54.62%
* Accuracy among accepted validation predictions: 81.04%

The uncertainty flag is intended to communicate lower-confidence predictions. It is not a guarantee that a low-confidence prediction is incorrect.

---

# End-to-End Inference

```text
Upload Image
     |
     v
Validate File
     |
     v
Preprocess to 224x224
     |
     +--------------------+
     |                    |
     v                    v
ResNet-18              U-Net
     |                    |
     v                    v
Disease + Confidence   Disease Mask
     |                    |
     +----------+---------+
                |
                v
            Grad-CAM
                |
                v
        Prediction Result
```

---

# Backend

The application backend is implemented using **FastAPI**.

## Main Endpoints

```text
GET  /health
POST /predict
GET  /docs
```

`/health` reports backend and model-loading status.

`/predict` accepts a plant image and returns the complete prediction result.

The FastAPI application also validates uploaded files, including supported image types, empty uploads, corrupted images, and file-size limits.

### Example Response Structure

```json
{
  "disease": "apple black rot",
  "confidence": 0.9739,
  "class_index": 0,
  "uncertain": false,
  "mask_coverage": 0.0141,
  "segmentation_mask": "...",
  "gradcam": "..."
}
```

---

# Frontend

The frontend is a responsive browser-based interface containing:

* image upload and drag-and-drop support
* image preview
* analysis/loading state
* prediction result
* confidence display
* disease-region visualization
* Grad-CAM visualization
* interpretation of the prediction
* uncertainty indication
* error handling

Frontend structure:

```text
frontend/
├── index.html
├── style.css
└── script.js
```

---

# Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/durvankmali/AgriSense.git
cd AgriSense
```

## 2. Create and Activate a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

## 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

The project is configured for CPU inference when CUDA is unavailable.

## 4. Install PyTorch

PyTorch and torchvision are installed separately because the project uses the CPU-only PyTorch builds.

For CPU-only installation:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
```

## 5. Start the FastAPI Server

From the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## 6. Start the Frontend

The frontend can be served locally using VS Code Live Server or another static HTTP server.

For VS Code Live Server, the frontend will typically be available at:

```text
http://127.0.0.1:5500
```

---

# Local and Production API Configuration

The API endpoint used by the frontend is configured in:

```text
frontend/script.js
```

## Local Development

When running the FastAPI backend locally, use:

```javascript
const API_URL = "http://127.0.0.1:8000/predict";
```

The local FastAPI server must be running:

```bash
uvicorn app.main:app --reload
```

## Public Deployment

For the deployed Render frontend, use:

```javascript
const API_URL = "https://agrisense-api-9bzp.onrender.com/predict";
```

### Important

The **Render API URL should remain active in the committed production version** of `frontend/script.js`.

For local development, temporarily replace it with the localhost URL.

A convenient configuration is:

```javascript
const API_URL = "https://agrisense-api-9bzp.onrender.com/predict";

// LOCAL DEVELOPMENT:
// const API_URL = "http://127.0.0.1:8000/predict";
```

When working locally:

1. Comment the Render URL.
2. Uncomment the localhost URL.
3. Run the FastAPI backend locally.
4. Run the frontend through Live Server.

Before committing and pushing changes:

1. Comment the localhost URL.
2. Uncomment the Render URL.
3. Verify the production URL is the active API endpoint.
4. Then commit and push.

This prevents the deployed frontend from attempting to call a user's local computer.

---

# Docker

The project includes a CPU-based Docker configuration.

## Build

```bash
docker build -t agrisense-api .
```

## Run

```bash
docker run --rm -p 8000:8000 agrisense-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The Docker image installs the CPU-only PyTorch and torchvision builds separately from the remaining Python requirements.

---

# Deployment

The project was prepared for deployment using Render:

* FastAPI backend deployed as a Docker Web Service
* frontend deployed as a Render Static Site
* model checkpoints managed with Git LFS
* health endpoint configured for service monitoring

The public deployment was useful for validating the complete deployment workflow.

## Deployment Limitation

The free backend environment has limited memory. The full inference pipeline includes classification, segmentation, and Grad-CAM, which creates a significant memory footprint.

The public backend successfully returned predictions, but repeated inference can exceed the available free-service resources and cause the service process to restart.

For demonstrations, the **local deployment is the authoritative environment** because it provides stable execution of the complete pipeline.

---

# Limitations

1. Classification performance is not sufficient for autonomous agricultural diagnosis.
2. Some disease classes have limited samples.
3. Class imbalance remains present in the dataset.
4. Small and thin disease regions can be missed by segmentation.
5. Grad-CAM provides coarse visual explanations rather than precise disease boundaries.
6. Segmentation mask coverage is not a validated disease-severity measurement.
7. The PlantSeg dataset does not provide geolocation, so location-specific recommendations are not currently part of the implemented system.
8. Environmental/contextual information is not yet integrated into the prediction pipeline.
9. Public free-tier deployment is constrained by memory and service lifecycle limits.

---

# Future Scope

Potential extensions include:

* improved fine-tuning and class-specific training strategies
* stronger segmentation architectures
* richer disease localization metrics
* calibrated confidence estimates
* environmental and contextual data integration
* weather and location-aware risk analysis when appropriate data is available
* agricultural advisory integration using trusted external sources
* SQL-based prediction and analytics storage
* dashboard-based historical analysis
* improved cloud infrastructure for production inference
* model monitoring and performance tracking

These are future extensions and are not represented as currently implemented features.

---

# Development Workflow

The project was developed using a version-controlled workflow with Git and GitHub.

Major development stages included:

1. Dataset exploration
2. Data-quality analysis
3. Duplicate and leakage detection
4. Dataset cleaning
5. Preprocessing design
6. Baseline classification
7. Fine-tuning
8. Classification evaluation
9. Error analysis
10. Segmentation
11. Grad-CAM explainability
12. FastAPI integration
13. Frontend integration
14. Dockerization
15. Deployment testing
16. Resource optimization

Model checkpoints are stored separately from the source-code logic and tracked using Git LFS.

---

# Git Workflow

Typical development workflow:

```bash
git status
git add .
git commit -m "Describe the change"
git push
git status
```

Before pushing production frontend changes, verify that the active API URL in:

```text
frontend/script.js
```

is the Render API URL.

---

# Final Verification Summary

| Component                    | Status                             |
| ---------------------------- | ---------------------------------- |
| Clean dataset                | Complete                           |
| Duplicate/leakage checks     | Complete                           |
| ResNet-18 classifier         | Complete                           |
| U-Net segmentation           | Complete                           |
| Grad-CAM                     | Complete                           |
| Confidence/uncertainty logic | Complete                           |
| FastAPI backend              | Complete                           |
| Frontend                     | Complete                           |
| Docker                       | Complete                           |
| Local end-to-end inference   | Verified                           |
| Public deployment experiment | Verified with resource limitations |

---

# License and Dataset

The project uses the PlantSeg dataset. Refer to the original dataset source and license information before redistribution of dataset images or annotations.

---

## Conclusion

AgriSense demonstrates an end-to-end applied computer vision workflow, covering data quality analysis, dataset cleaning, leakage control, model development, evaluation, segmentation, explainability, API integration, frontend development, Dockerization, and deployment testing.

The project combines disease classification with disease-region segmentation and Grad-CAM explainability to provide a more informative result than classification alone.
