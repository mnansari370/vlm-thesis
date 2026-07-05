from typing import Callable, Optional, Tuple

from PIL import Image

# CLIP ViT-L/14 image mean used as background for expand2square — matches
# LLaVA-1.5 original preprocessing (llava/mm_utils.py) exactly.
# Values: tuple(int(x * 255) for x in [0.48145466, 0.4578275, 0.40821073])
_CLIP_MEAN_RGB: Tuple[int, int, int] = (122, 116, 104)


def _expand2square(
    img: Image.Image,
    background_color: Tuple[int, int, int] = _CLIP_MEAN_RGB,
) -> Image.Image:
    """Pad image to square with background_color, centered.

    Matches LLaVA-1.5 original preprocessing exactly. Applied before HF
    CLIPImageProcessor so the processor sees a square image and the subsequent
    resize is a no-op crop (square -> square), preserving all image content.
    """
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    result = Image.new("RGB", (side, side), background_color)
    result.paste(img, ((side - w) // 2, (side - h) // 2))
    return result


def build_image_transform(
    image_size: int = 336,
    is_train: bool = False,
    image_aspect_ratio: str = "center_crop",
) -> Optional[Callable]:
    """
    Build image transform for VQA v2 samples.

    image_aspect_ratio:
      "pad"         — expand2square with CLIP mean background (LLaVA-1.5 original,
                      matches FasterVLM). Apply this for paper-quality results.
      "center_crop" — no PIL-level transform; HF CLIPImageProcessor does its own
                      center crop (legacy default).

    In both cases the HF processor inside the model wrapper performs the final
    resize, normalization, and tensor conversion — this transform only handles
    the aspect-ratio correction at the PIL level.
    """
    use_pad = image_aspect_ratio == "pad"

    def _transform(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if use_pad:
            image = _expand2square(image)
        return image

    return _transform
