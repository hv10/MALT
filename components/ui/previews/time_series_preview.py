from nicegui import ui
import pandas as pd

def make_sample_plot(source):
    if source.endswith('.csv'):
        df = pd.read_csv(source)
    elif source.endswith('.tsv'):
        df = pd.read_csv(source, sep='\t')
    else:
        df = pd.DataFrame() # Empty DataFrame for unsupported formats.
        raise ValueError("Unsupported file format for time-series. Please provide a CSV or TSV file.")
    plt = ui.matplotlib()
    with plt:
        df.plot(subplots=True, layout=(len(df.columns), 1), figsize=(6, 2 * len(df.columns)))
    return plt

def make_ts_sample_preview(obj, source):
    plt = make_sample_plot(source)
    return plt.classes(
        "cursor-pointer hover:opacity-90 transition-opacity max-w-full"
    )

def make_ts_detail_preview(obj, source):
    plt = make_sample_plot(source)
    return plt.classes("h-full w-full object-contain")
