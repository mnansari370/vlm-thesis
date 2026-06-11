"""Minimal PIL-level transform; real preprocessing happens in the LLaVA processor."""

from typing import Callable, Optional

from PIL import Image


def build_image_transform(image_size: int = 336, is_train: bool = False) -> Optional[Callable]:
    """
    Build image transform for VQA v2 samples.

    Important design note for this thesis codebase:
    ----------------------------------------------
    In the current LLaVA-style pipeline, image resizing / normalization should
    be handled by the Hugging Face processor inside the model wrapper, not here.

    Therefore this function intentionally returns a minimal PIL-level transform.
    It keeps the dataset/model responsibilities clean:

    - dataset returns raw PIL images
    - collator keeps them as a list
    - model wrapper / processor performs model-specific preprocessing

    The image_size argument is retained for API/config consistency and future
    flexibility, but is not actively used here.
    """

    def _transform(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    return _transform