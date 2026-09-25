from PIL import Image

import torch

from src.config import (
    CLASSIFIER_CHECKPOINT,
    DEVICE,
    SEGMENTATION_CHECKPOINT,
    SEGMENTATION_THRESHOLD,
    load_disease_classes,
)

from src.models.classifier import (
    load_classifier,
    predict_classifier,
)

from src.models.segmentation import (
    load_segmentation_model,
    predict_segmentation,
)

from src.preprocessing.image_preprocessing import (
    preprocess_image,
)

from src.explainability.gradcam import (
    GradCAM,
)


class AgriSensePipeline:
    """
    End-to-end AgriSense inference pipeline.

    Combines:
    - Disease classification
    - Disease-region segmentation
    - Grad-CAM explainability
    """

    def __init__(
        self,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        device=DEVICE,
    ):
        self.device = device

        # Load disease classes
        self.class_names = load_disease_classes()

        # Load classifier
        self.classifier, self.classifier_checkpoint = (
            load_classifier(
                classifier_checkpoint,
                num_classes=len(self.class_names),
                device=device,
            )
        )

        # Load segmentation model
        self.segmenter, self.segmentation_checkpoint = (
            load_segmentation_model(
                segmentation_checkpoint,
                device=device,
            )
        )

        # Grad-CAM target layer
        self.gradcam = GradCAM(
            model=self.classifier,
            target_layer=self.classifier.layer4[-1],
        )

    def predict(self, image_path):
        """
        Run the complete AgriSense inference pipeline.

        Parameters
        ----------
        image_path:
            Path to the input plant image.

        Returns
        -------
        dict
            Classification, segmentation, and explainability results.
        """

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # --------------------------------------------------
        # 1. Disease classification
        # --------------------------------------------------

        classification = predict_classifier(
            model=self.classifier,
            image_tensor=image_tensor,
            class_names=self.class_names,
            device=self.device,
        )

        predicted_index = classification["class_index"]

        # --------------------------------------------------
        # 2. Disease segmentation
        # --------------------------------------------------

        segmentation = predict_segmentation(
            model=self.segmenter,
            image_tensor=image_tensor,
            device=self.device,
            threshold=SEGMENTATION_THRESHOLD,
        )

        # --------------------------------------------------
        # 3. Grad-CAM
        # --------------------------------------------------

        cam, _ = self.gradcam.generate(
            image_tensor=image_tensor,
            class_index=predicted_index,
        )

        # --------------------------------------------------
        # 4. Combined result
        # --------------------------------------------------

        return {
            "disease": classification["disease"],
            "confidence": classification["confidence"],
            "class_index": classification["class_index"],
            "mask_coverage": segmentation["coverage"],
            "segmentation_mask": (
                segmentation["mask"]
                .detach()
                .cpu()
                .numpy()
            ),
            "segmentation_probability": (
                segmentation["probability_map"]
                .detach()
                .cpu()
                .numpy()
            ),
            "gradcam": cam,
        }

    def close(self):
        """
        Remove Grad-CAM hooks.
        """

        self.gradcam.close()