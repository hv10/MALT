"""!IMPORTANT!: Initial version created with GPT-5.6, manually checked for correctness."""
from __future__ import annotations

import re
from collections.abc import Hashable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter


def _parse_patch_info(
    value: object,
    pattern: re.Pattern[str],
) -> tuple[int, int, int, int]:
    match = pattern.search(str(value))
    if match is None:
        raise ValueError(
            f"Path {value!r} does not match the supplied filename pattern"
        )

    if match.lastindex is None or match.lastindex < 2:
        raise ValueError(
            "fpattern must contain at least two capture groups: "
            "patch x-index and patch y-index"
        )

    def optional_int(group: int) -> int:
        if match.lastindex is None or match.lastindex < group:
            return 0

        matched_value = match.group(group)
        return (
            0
            if matched_value is None or matched_value == ""
            else int(matched_value)
        )

    patch_x = int(match.group(1))
    patch_y = int(match.group(2))
    overlap_x = optional_int(3)
    overlap_y = optional_int(4)

    if patch_x < 0 or patch_y < 0:
        raise ValueError(
            f"Patch indices must be non-negative, got "
            f"({patch_x}, {patch_y}) for {value!r}"
        )

    if overlap_x < 0 or overlap_y < 0:
        raise ValueError(
            f"Overlaps must be non-negative, got "
            f"({overlap_x}, {overlap_y}) for {value!r}"
        )

    return patch_x, patch_y, overlap_x, overlap_y


def _resolve_patch_path(
    value: object,
    image_folder: Path,
) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else image_folder / path


def _find_first_image(
    fpths: pd.Series,
    image_folder: Path,
) -> tuple[int, int]:
    for fpth in fpths:
        candidate = _resolve_patch_path(fpth, image_folder)
        if not candidate.is_file():
            continue

        try:
            with Image.open(candidate) as image:
                image.load()
                return image.size
        except OSError:
            continue

    raise FileNotFoundError(
        "None of the paths in df['fpth'] could be opened as an image "
        f"inside {image_folder}"
    )


def _class_sort_key(value: Hashable) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _categorical_colors(
    classes: list[Hashable],
    cmap_name: str = "tab20",
) -> dict[Hashable, tuple[int, int, int, int]]:
    """
    Assign deterministic RGBA colors from a Matplotlib categorical colormap.

    Colors repeat when there are more classes than discrete entries in the
    selected colormap.
    """
    cmap = colormaps[cmap_name]

    # Listed categorical colormaps such as tab10 and tab20 expose their
    # discrete palette size through cmap.N.
    colors: dict[Hashable, tuple[int, int, int, int]] = {}

    for index, cls in enumerate(classes):
        red, green, blue, _ = cmap(index % cmap.N)

        colors[cls] = (
            round(red * 255),
            round(green * 255),
            round(blue * 255),
            255,
        )

    return colors


def _composite_patch(
    base: Image.Image,
    patch_path: Path,
    position: tuple[int, int],
    expected_size: tuple[int, int],
) -> None:
    try:
        with Image.open(patch_path) as patch_image:
            patch_image.load()
            patch = patch_image.convert("RGBA")
    except OSError as error:
        raise OSError(f"Could not open patch image {patch_path}") from error

    if patch.size != expected_size:
        raise ValueError(
            f"Patch {patch_path} has size {patch.size}, "
            f"expected {expected_size}"
        )

    base.alpha_composite(patch, position)


def _resolve_output_path(dest: str | Path) -> Path:
    destination = Path(dest)

    # Existing directories and non-existing paths without a suffix are treated
    # as directories.
    if destination.is_dir() or not destination.suffix:
        destination.mkdir(parents=True, exist_ok=True)
        return destination / "combined.png"

    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def soft_label(
    df: pd.DataFrame,
    dest: str | Path,
    fpattern: str | re.Pattern[str],
    img_folder: str | Path | None,
    radius: int | float,
    gauss: int | float,
    patch_size: tuple[int, int] | None,
    patch_pos_by_idx: bool = True,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Reconstruct an image from patches and overlay blurred categorical labels.

    Expected dataframe columns
    --------------------------
    fpth:
        Patch filename or relative path.
    cls:
        Categorical class value.

    Filename pattern groups
    -----------------------
    Group 1:
        Patch x-pos.
    Group 2:
        Patch y-pos.
    Group 3, optional:
        Horizontal overlap in pixels.
    Group 4, optional:
        Vertical overlap in pixels.

    Parameters
    ----------
    df:
        Input dataframe. Has to have at least columns ["fpth","cls"]
    dest:
        Output file or directory. If it is a directory, the output is saved as
        ``combined.png``.
    fpattern:
        Regex used to extract patch coordinates and optional overlaps.
    img_folder:
        Directory containing the patch images. When provided, patch dimensions
        are inferred from the first existing image and the patches are composed
        into the base image.
    radius:
        Radius of each class-label circle, in pixels; or as a fraction 0<r<1, in %of patch-width.
    gauss:
        Gaussian blur radius, in pixels; or as a fraction 0<rg<1, in %of patch-width.
    patch_size:
        ``(width, height)`` used when ``img_folder`` is None. Ignored otherwise. 
        Make sure that when this is used that either patch_pos_by_idx is set or that the patch_size is set accordingly.
    patch_pos_by_idx:
        If the patch position is given by an index pair rather than a pixel position.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe containing the additional columns
        ``patch_x``, ``patch_y``, ``overlap_x``, and ``overlap_y``.
    """
    required_columns = {"fpth", "cls"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataframe is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    if df.empty:
        raise ValueError("df must contain at least one patch")

    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    if gauss < 0:
        raise ValueError(f"gauss must be non-negative, got {gauss}")
    
    df["cls"] = df["cls"].apply(str)

    pattern = (
        re.compile(fpattern)
        if isinstance(fpattern, str)
        else fpattern
    )

    result = df.copy()

    parsed = result["fpth"].map(
        lambda value: _parse_patch_info(value, pattern)
    )

    result[
        ["patch_x", "patch_y", "overlap_x", "overlap_y"]
    ] = pd.DataFrame(
        parsed.tolist(),
        index=result.index,
    )

    result = result.sort_values(
        ["patch_x", "patch_y"],
        kind="stable",
    ).reset_index(drop=True)

    image_folder = Path(img_folder) if img_folder is not None else None

    if image_folder is not None:
        if not image_folder.is_dir():
            raise NotADirectoryError(
                f"img_folder is not a directory: {image_folder}"
            )

        patch_width, patch_height = _find_first_image(
            result["fpth"],
            image_folder,
        )
    else:
        if patch_size is None:
            raise ValueError(
                "patch_size must be provided when img_folder is None"
            )

        patch_width, patch_height = map(int, patch_size)

    if patch_width <= 0 or patch_height <= 0:
        raise ValueError(
            "patch_size must contain positive dimensions, got "
            f"({patch_width}, {patch_height})"
        )

    if (result["overlap_x"] >= patch_width).any():
        raise ValueError(
            "Every horizontal overlap must be smaller than the patch width"
        )

    if (result["overlap_y"] >= patch_height).any():
        raise ValueError(
            "Every vertical overlap must be smaller than the patch height"
        )

    radius = (patch_width*radius) if radius<1 else radius
    gauss = (patch_width*gauss) if gauss<1 else gauss

    # Position of patch index n:
    #
    #     n * (patch dimension - overlap)
    #
    # Different rows may technically contain different overlap values, though
    # patches in the same column or row should normally agree.
    if patch_pos_by_idx:
        result["left"] = (
            result["patch_x"]
            * (patch_width - result["overlap_x"])
        ).astype(int)

        result["top"] = (
            result["patch_y"]
            * (patch_height - result["overlap_y"])
        ).astype(int)
    else:
        result["left"] = result["patch_x"].astype(int)
        result["top"] = result["patch_y"].astype(int)

    canvas_width = int((result["left"] + patch_width).max())
    canvas_height = int((result["top"] + patch_height).max())

    base = Image.new(
        "RGBA",
        (canvas_width, canvas_height),
        (0, 0, 0, 0),
    )

    classes = sorted(
        pd.unique(result["cls"]),
        key=_class_sort_key,
    )

    class_colors = _categorical_colors(
        classes,
        cmap_name="tab20",
    )
    class_colors[str(float("NaN"))] = (0,0,0,255)

    # Each class is blurred independently so categorical colors do not get
    # blended into arbitrary intermediate RGB values.
    class_masks = {
        cls: Image.new(
            "L",
            (canvas_width, canvas_height),
            0,
        )
        for cls in classes
    }

    class_draws = {
        cls: ImageDraw.Draw(mask)
        for cls, mask in class_masks.items()
    }

    expected_patch_size = (patch_width, patch_height)

    for row in result.itertuples(index=False):
        left = int(row.left)
        top = int(row.top)
        ovx = int(row.overlap_x)
        ovy = int(row.overlap_y)

        if image_folder is not None:
            patch_path = _resolve_patch_path(
                row.fpth,
                image_folder,
            )

            if patch_path.is_file():
                _composite_patch(
                    base=base,
                    patch_path=patch_path,
                    position=(left, top),
                    expected_size=expected_patch_size,
                )

        # Use the effective, non-overlapping patch area to determine the
        # position of the label marker.
        center_x = left + (patch_width - ovx) / 2
        center_y = top + (patch_height - ovy) / 2

        bounds = (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )

        # Draw only into this class's density mask.
        class_draws[row.cls].ellipse(
            bounds,
            fill=255,
        )

    density_maps = []

    for cls in classes:
        mask = class_masks[cls]

        if gauss > 0:
            mask = mask.filter(
                ImageFilter.GaussianBlur(radius=gauss)
            )

        density_maps.append(
            np.asarray(mask, dtype=np.float32) / 255.0
        )

    # Shape: (height, width, number_of_classes)
    density_stack = np.stack(
        density_maps,
        axis=-1,
    )

    # The strongest class determines the categorical color at each pixel.
    dominant_class = np.argmax(
        density_stack,
        axis=-1,
    )

    color_table = np.asarray(
        [class_colors[cls][:3] for cls in classes],
        dtype=np.uint8,
    )

    label_rgb = color_table[dominant_class]

    # Summing the independently blurred masks gives the overall local label
    # density. Normalization here is only for mapping density to display alpha;
    # density_stack itself remains the non-normalized density representation.
    total_density = density_stack.sum(axis=-1)
    max_density = float(total_density.max())

    if max_density > 0:
        label_alpha = np.clip(
            total_density / max_density,
            0.0,
            1.0,
        )
    else:
        label_alpha = np.zeros_like(
            total_density,
            dtype=np.float32,
        )

    # Ensure completely empty pixels are transparent. This also avoids assigning
    # the first class's color any visible alpha where no density exists.
    label_alpha[total_density <= 0] = 0.0

    label_array = np.empty(
        (canvas_height, canvas_width, 4),
        dtype=np.uint8,
    )
    label_array[..., :3] = label_rgb
    label_array[..., 3] = np.round(
        label_alpha * np.ceil((255*alpha))
    ).astype(np.uint8)

    label_layer = Image.fromarray(
        label_array,
        mode="RGBA",
    )

    if img_folder is None:
        combined = label_layer
    else:
        combined = Image.alpha_composite(
            base,
            label_layer,
        )


    combined.save(
        _resolve_output_path(dest)
    )

    return result.drop(columns=["left", "top"])