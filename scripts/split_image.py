from pathlib import Path
from typing import TypeAlias
import math

from PIL import Image

Fraction2D: TypeAlias = tuple[float, float]


def split_image(
    source: str | Path,
    dest: str | Path,
    size: Fraction2D,
    overlap: Fraction2D = (0.0, 0.0),
    pad: bool = False,
) -> list[Path]:
    """
        Split an image into overlapping patches.

    `size` and `overlap` are fractions of the complete image dimensions.
    Tuple entries are ordered as `(horizontal, vertical)`.

    For example:
        size=(0.25, 0.5)
        overlap=(0.05, 0.1)

    creates patches covering 25% of the image width and 50% of its height,
    with overlaps of 5% of the patch width and 10% of th patch height.

    When `pad=True`, patches extending beyond the right or bottom edge are
    padded to the requested patch size. Transparent images receive transparent
    black padding; other images receive opaque black padding.

    Returns:
        Paths of all generated PNG patches.
    """
    source = Path(source)
    dest = Path(dest)

    if len(size) != 2 or len(overlap) != 2:
        raise ValueError("size and overlap must contain exactly two values")

    size_x, size_y = size
    overlap_x, overlap_y = overlap

    if not 0 < size_x <= 1 or not 0 < size_y <= 1:
        raise ValueError("size fractions must be greater than 0 and at most 1")

    if not 0 <= overlap_x < 1 or not 0 <= overlap_y < 1:
        raise ValueError("overlap fractions must be at least 0 and smaller than 1")

    dest.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as opened_image:
        has_alpha = (
            opened_image.mode in {"RGBA", "LA"} or "transparency" in opened_image.info
        )
        image = opened_image.convert("RGBA" if has_alpha else "RGB")

    image_width, image_height = image.size

    patch_width = max(1, round(image_width * size_x))
    patch_height = max(1, round(image_height * size_y))

    # Overlap is relative to the patch size.
    overlap_width = round(patch_width * overlap_x)
    overlap_height = round(patch_height * overlap_y)

    step_x = patch_width - overlap_width
    step_y = patch_height - overlap_height

    if step_x <= 0 or step_y <= 0:
        raise ValueError("overlap must leave a positive step in both dimensions")

    padding_color = (0, 0, 0, 0) if has_alpha else (0, 0, 0)
    patch_count = 0

    overlap_fp = "" if overlap==(0.0,0.0) else f"_o{overlap_width}x{overlap_height}"
    pd = math.ceil(max(math.log10(image_height) ,math.log10(image_width)))

    for y in range(0, image_height, step_y):
        for x in range(0, image_width, step_x):
            right = min(x + patch_width, image_width)
            bottom = min(y + patch_height, image_height)

            patch = image.crop((x, y, right, bottom))

            if pad and patch.size != (patch_width, patch_height):
                padded_patch = Image.new(
                    image.mode,
                    (patch_width, patch_height),
                    padding_color,
                )
                padded_patch.paste(patch, (0, 0))
                patch = padded_patch

            output_path = dest / (f"{source.stem}_{x:0{pd}d}_{y:0{pd}d}{overlap_fp}.png")
            patch.save(output_path)
            patch_count += 1

    return patch_count
