import numpy as np
from sklearn.utils.extmath import randomized_svd


def project_and_remove_direction(X, v):
    """
    Project each vector in X onto vector v and then remove this component from X.
    """
    v = v / np.linalg.norm(v)  # ensure |v| = 1
    projection_onto_v = np.dot(X, v)[:, np.newaxis] * v
    print(projection_onto_v.shape)
    X_residual = X - np.squeeze(projection_onto_v)  # remove projection
    return X_residual


def dir_of_max_variance(X, count=1):
    """
    Perform PCA to find the `count` first principal component (nD PCA).
    Optimized using Singular Value Decomposition (SVD).
    """
    n = len(X)
    fit_samples = min(n, max(2000, int(0.05 * n)))

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=fit_samples, replace=False)

    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    _, _, w = randomized_svd(
        X_centered[idx], n_components=count, n_iter=2, random_state=42
    )
    Vt = w.T
    X_projected = X_centered @ Vt
    return Vt, X_projected


def orthonormalize(vectors, return_indices=False):
    """
    Orthonormalize a set of vectors (columns of a matrix) using Gram-Schmidt.
    """
    Q = []
    keep = []
    for i, v in enumerate(vectors):
        v = v.copy()
        for q in Q:
            v -= np.dot(q, v) * q
        if np.linalg.norm(v) > 1e-10:
            keep.append(i)
            Q.append(v / np.linalg.norm(v))
    if return_indices:
        return np.array(Q), np.array(keep)
    return np.array(Q)


def project_onto_orthogonal_complement(X, coeffs):
    """
    Removes the components of X along the directions in coeffs simultaneously.
    """
    # Orthonormalize first
    C = orthonormalize(coeffs)  # shape (num_coeffs, n_features)
    C = C.T  # now shape (n_features, num_coeffs)

    # Projection matrix onto subspace spanned by C: P = C @ C.T
    # Projection onto orthogonal complement: I - P
    X_proj = (X @ C) @ C.T
    X_residual = X - X_proj
    return X_residual


def update_projection(state, coeffs, intercepts, project_X=True, run_pca=True):
    # project our data onto the (d-1)-simplex defined by coeffs[]
    # either:
    ## data_x will be the closest value to zero of X@coeff (minimal magnitude)
    ## or data_x will be the magnitude of the projection of X onto
    ## the span of coeffs negated (so that higher is more similar)
    # then remove its influence
    if project_X:
        X = np.stack(state.DATA["emb"])
        if state.CLS_TYPE == "MIN_MARGIN":
            coeffs_mat = np.array(coeffs)  # (k, d)
            norms = np.linalg.norm(coeffs_mat, axis=1, keepdims=True)  # (k, 1)
            px = np.transpose(
                (X @ coeffs_mat.T + np.array(intercepts)) / norms.T
            )  # (k, n) in one BLAS call
            min_idx = np.argmin(np.abs(px), axis=0)
            pos_x = px[min_idx, np.arange(px.shape[1])]
        else:  # PROJECTION
            Q, b_keep = orthonormalize(coeffs, return_indices=True)  # orthonormal span
            X_d = X @ Q.T + np.array(intercepts)[b_keep]  # project onto span
            pos_x = 1 / (1 + np.exp(-np.sum(X_d, axis=1)))
            pos_x = 2 * (pos_x - 0.5)
        projX = project_onto_orthogonal_complement(X, coeffs)
        state.PROJ_X = projX
    else:
        pos_x = state.DATA["pos_x"].to_numpy()

    # now we will reproject onto a 3D PCA Projection
    # which we will then reweigh according to our ternary.
    if run_pca:
        w, _ = dir_of_max_variance(state.PROJ_X, count=3)
        state.PCA_MODEL = w
    else:
        w = state.PCA_MODEL
    # w are our principal components
    # we now construct the axis by doing a linear combination of the columns of w
    P_ax = (
        state.META["axis"]["c"] * w[:, 0]
        + state.META["axis"]["a"] * w[:, 1]
        + state.META["axis"]["b"] * w[:, 2]
    )
    data_y = np.dot(state.PROJ_X, P_ax)
    return pos_x, data_y
