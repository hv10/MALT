import pandas as pd
from nicegui import ui
import rootpath
rootpath.append()

from components.utils import has_header


def make_sample_plot(source):
    if source.endswith(".csv"):
        df = pd.read_csv(source, header=has_header(source))
    elif source.endswith(".tsv"):
        df = pd.read_csv(source, sep="\t", header=has_header(source, "\t"))
    else:
        df = pd.DataFrame()  # Empty DataFrame for unsupported formats.
        raise ValueError(
            "Unsupported file format for time-series. Please provide a CSV or TSV file."
        )
    plt = ui.matplotlib(figsize=(6, 2 * len(df.columns)))
    with plt:
        ax = plt.figure.gca()
        if len(df.columns) > 1:
            df.plot(  # type: ignore
                x=df.columns[0],  # type:ignore
                y=list(df.columns[1:]),
                sharex=True,
                subplots=True,
                layout=(len(df.columns) - 1, 1),
                ax=ax,
            )
        else:
            df.plot(subplots=True, ax=ax)

    return plt


def make_ts_sample_preview(obj, source):
    plt = make_sample_plot(source)
    return plt.classes("cursor-pointer hover:opacity-90 transition-opacity max-w-full")


def make_ts_detail_preview(obj, source):
    plt = make_sample_plot(source)
    return plt.classes("h-full w-full object-contain")
