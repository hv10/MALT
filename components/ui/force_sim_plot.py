import numpy as np
import plotly.graph_objects as go
from nicegui import ui


@ui.refreshable
def force_similarity_plot(
    state,
    sign_invariant=False,
    n_iter=500,
    repulsion_strength=1.0,
    attraction_strength=1.0,
    seed=42,
):
    if state.PAC_MODEL is None:
        return go.Figure()  # no model, no plot
    W = np.stack(
        [np.squeeze(est.coef_) for est in state.PAC_MODEL["model"].estimators_]
    )
    labels = state.PAC_MODEL["labels"]

    np.random.seed(seed)
    n = W.shape[0]
    # ---- Normalize vectors ----
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    # ---- Cosine similarity ----
    S = W @ W.T
    if sign_invariant:
        S = S**2
    np.fill_diagonal(S, 0.0)
    # ---- Initialize random 2D positions ----
    pos = np.random.randn(n, 2)
    # ---- Force-directed layout ----
    for _ in range(n_iter):
        delta = pos[:, None, :] - pos[None, :, :]  # pairwise differences
        distance = np.linalg.norm(delta, axis=2) + 1e-6
        # Repulsion (Coulomb-like)
        repulsion = repulsion_strength / distance**2
        rep_force = (delta / distance[:, :, None]) * repulsion[:, :, None]
        # Attraction weighted by similarity
        attraction = attraction_strength * S
        attr_force = -(delta) * attraction[:, :, None]
        total_force = rep_force.sum(axis=1) + attr_force.sum(axis=1)
        pos += 0.01 * total_force  # small step size
    # ---- Build Plotly figure ----
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers+text",
            marker=dict(size=12),
            text=labels,
            textposition="top center",
        )
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=5, r=5, t=15, b=5),
    )
    ui.label(
        "Classifier Similarity (Sign-Invariant)"
        if sign_invariant
        else "Classifier Similarity"
    )
    ui.plotly(fig).classes("w-full")
