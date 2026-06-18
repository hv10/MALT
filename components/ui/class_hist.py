import plotly.express as px
from nicegui import ui


@ui.refreshable
def class_hist(state):
    # Explode 'cls' so each label becomes a separate row
    cls_exploded = state.DATA["cls"].explode().astype(str)

    # Count occurrences of each label
    label_counts = (
        cls_exploded.value_counts()
        .reindex(state.META["classes"], fill_value=0)
        .reset_index()
    )
    label_counts.columns = ["label", "count"]

    # Plot using Plotly Express
    fig = px.bar(label_counts, x="count", y="label", orientation="h")
    ui.label("Class Frequency")
    ui.plotly(fig).classes("w-full")
