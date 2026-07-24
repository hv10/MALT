import hashlib
import io
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootpath
from matplotlib.axes import Axes
from nicegui import ui

rootpath.append()

from components.utils import load_table, has_header


def _stable_category_color(value, *, cmap_name: str = "tab20"):
    """Map a category deterministically to a color from a Matplotlib colormap."""
    digest = hashlib.blake2b(
        str(value).encode("utf-8"),
        digest_size=8,
    ).digest()

    hashed = int.from_bytes(digest, byteorder="big")

    cmap = plt.get_cmap(cmap_name)

    # Prefer discrete entries when the colormap defines them.
    n_colors = getattr(cmap, "N", 256)
    color_index = hashed % n_colors

    return np.asarray(cmap(color_index), dtype=np.float32)[:3]


def _normalize_numeric(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)

    normalized = np.full(len(series), np.nan, dtype=float)

    if not finite.any():
        return normalized

    low = values[finite].min()
    high = values[finite].max()

    if high > low:
        normalized[finite] = (values[finite] - low) / (high - low)
    else:
        normalized[finite] = 0.5

    return normalized


def _normalize_datetime(series: pd.Series) -> np.ndarray:
    values = pd.to_datetime(series, errors="coerce")
    missing = values.isna().to_numpy()

    numeric = values.astype("int64").to_numpy(dtype=float)
    numeric[missing] = np.nan

    return _normalize_numeric(pd.Series(numeric, index=series.index))


def representative_row_indices(
    length: int,
    count: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    if length <= count:
        return np.arange(length)

    edge_count = min(3, count // 4)
    random_count = count - 2 * edge_count

    rng = np.random.default_rng(seed)

    middle = np.arange(edge_count, length - edge_count)
    random_rows = rng.choice(
        middle,
        size=random_count,
        replace=False,
    )

    return np.sort(
        np.concatenate(
            [
                np.arange(edge_count),
                random_rows,
                np.arange(length - edge_count, length),
            ]
        )
    )


def dataframe_preview_rgb(
    df: pd.DataFrame,
    *,
    max_rows: int = 30,
    max_cols: int = 30,
    missing_color=(0.2, 0.2, 0.2),
    numeric_cmap: str = "viridis",
    categorical_cmap: str = "tab20",
):
    """
    Convert a DataFrame into an RGB image suitable for a mini heatmap.

    Numeric and datetime columns:
        Continuous colormap, normalized independently per column.

    Categorical columns:
        Stable hashed color per category.

    Boolean columns:
        Two distinct brightness levels.

    Missing values:
        Dedicated missing_color.
    """
    if df.empty or df.shape[1] == 0:
        return np.empty((0, 0, 3), dtype=np.float32)

    n_rows = min(len(df), max_rows)
    n_cols = min(df.shape[1], max_cols)

    row_idx = representative_row_indices(
        len(df),
        min(len(df), max_rows),
    )
    col_idx = np.linspace(0, df.shape[1] - 1, n_cols, dtype=int)

    sampled = df.iloc[row_idx, col_idx]
    image = np.empty((n_rows, n_cols, 3), dtype=np.float32)

    cmap = plt.get_cmap(numeric_cmap)
    missing_rgb = np.asarray(missing_color, dtype=np.float32)

    for column_index, column_name in enumerate(sampled.columns):
        series = sampled[column_name]
        missing = series.isna().to_numpy()

        if pd.api.types.is_bool_dtype(series.dtype):
            rgb = np.empty((n_rows, 3), dtype=np.float32)
            rgb[:] = missing_rgb

            valid = ~missing
            boolean_values = series[valid].astype(bool).to_numpy()

            rgb[valid] = np.where(
                boolean_values[:, None],
                np.array([0.85, 0.85, 0.85]),
                np.array([0.25, 0.25, 0.25]),
            )

        elif pd.api.types.is_numeric_dtype(series.dtype):
            normalized = _normalize_numeric(series)
            rgb = cmap(np.nan_to_num(normalized, nan=0.0))[:, :3]
            rgb[missing] = missing_rgb

        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            normalized = _normalize_datetime(series)
            rgb = cmap(np.nan_to_num(normalized, nan=0.0))[:, :3]
            rgb[missing] = missing_rgb

        else:
            rgb = np.empty((n_rows, 3), dtype=np.float32)

            for row_index, value in enumerate(series):
                if pd.isna(value):
                    rgb[row_index] = missing_rgb
                else:
                    rgb[row_index] = _stable_category_color(
                        value, cmap_name=categorical_cmap
                    )

        image[:, column_index, :] = rgb

    return image


def plot_dataframe_preview(
    df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    max_rows: int = 30,
    max_cols: int = 30,
    missing_color: tuple[float, float, float] = (0.2, 0.2, 0.2),
    numeric_cmap: str = "viridis",
    categorical_cmap: str = "tab20",
) -> Axes:
    image = dataframe_preview_rgb(
        df,
        max_rows=max_rows,
        max_cols=max_cols,
        missing_color=missing_color,
        numeric_cmap=numeric_cmap,
        categorical_cmap=categorical_cmap,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(1, 1), dpi=100)

    if image.size == 0:
        ax.text(
            0.5,
            0.5,
            "Empty",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.imshow(
            image,
            aspect="auto",
            interpolation="nearest",
        )

    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

    return ax


def series_plot_svg(series: pd.Series) -> str:
    """Create a small histogram/bar plot and return it as inline SVG."""

    fig, ax = plt.subplots(figsize=(1.6, 0.65))

    clean = series.dropna()

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        ax.hist(clean, bins=min(24, max(3, clean.nunique())))
    else:
        counts = clean.astype(str).value_counts().head(8)
        ax.bar(range(len(counts)), counts.values, width=1)
        ax.set_xticks([])

    ax.set_axis_off()
    ax.margins(x=0.0, y=0.1)

    buffer = io.StringIO()
    fig.savefig(
        buffer,
        format='svg',
        bbox_inches='tight',
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)

    svg = buffer.getvalue()

    # Remove the XML declaration because the SVG is embedded into HTML.
    return svg[svg.find('<svg'):]

def make_table_sample_preview(obj, source):
    prv = ui.matplotlib(figsize=(1, 1), dpi=100)
    with prv:
        ax = prv.figure.gca()
        plot_dataframe_preview(load_table(source), ax=ax)
    return prv.classes("cursor-pointer hover:opacity-90 transition-opacity max-w-full")


def make_table_detail_preview(obj, source):
    df = load_table(source, header=has_header(source))
    prv = ui.table.from_pandas(df).props("dense").classes("sticky")
    with prv.add_slot("header"):
        with prv.row():
            for column in prv.columns:
                column_name = column["name"]
                with prv.header(str(column_name)):
                    with ui.column().classes("items-stretch gap-1 min-w-[80px] py-1"):
                        ui.label(escape(str(column.get("label", column_name)))).classes(
                            "text-sm font-bold"
                        )
                        ui.html(
                            series_plot_svg(df[column_name]),
                            sanitize=False,
                        ).classes("w-full h-12 overflow-hidden")
    return prv.classes("h-full w-full object-contain")


if __name__ in {"__main__", "__mp_main__"}:
    make_table_sample_preview(None, "/Users/noeldanz/Software_Projects/MALT/data/phoneme/Phoneme_0_1.csv")
    make_table_detail_preview(None, "/Users/noeldanz/Software_Projects/MALT/data/phoneme/Phoneme_0_1.csv")
    ui.run(reload=True)
