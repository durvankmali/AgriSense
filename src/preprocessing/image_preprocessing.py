from PIL import Image
import torch
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def resize_and_pad(image: Image.Image, size: int = IMAGE_SIZE):
    """
    Resize an image while preserving its aspect ratio,
    then pad it to a square of the requested size.
    """

    image = image.convert("RGB")

    width, height = image.size

    scale = min(
        size / width,
        size / height,
    )

    new_width = round(width * scale)
    new_height = round(height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    padded_image = Image.new(
        "RGB",
        (size, size),
        (0, 0, 0),
    )

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    padded_image.paste(
        image,
        (left, top),
    )

    return padded_image


def preprocess_image(image: Image.Image):
    """
    Apply the preprocessing used by the AgriSense classifier.
    """

    image = resize_and_pad(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    tensor = transform(image)

    return tensor.unsqueeze(0)