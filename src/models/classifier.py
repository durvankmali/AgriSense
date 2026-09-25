import torch
import torch.nn as nn
from torchvision import models


def build_classifier(
    num_classes: int = 115,
    device: torch.device | None = None,
):
    """
    Build the final AgriSense ResNet-18 classifier.

    Architecture:
    - ImageNet-pretrained ResNet-18
    - Frozen backbone
    - Fine-tuned Layer 4
    - Custom classification head
    """

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Freeze the complete backbone
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune Layer 4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the original ImageNet classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    if device is not None:
        model = model.to(device)

    return model


def load_classifier(
    checkpoint_path,
    num_classes: int = 115,
    device: torch.device | None = None,
):
    """
    Build the final classifier and load trained weights.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = build_classifier(
        num_classes=num_classes,
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def predict_classifier(
    model,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
):
    """
    Generate a disease prediction from a preprocessed image tensor.
    """

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    predicted_disease = class_names[predicted_index]

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "class_index": predicted_index,
    }