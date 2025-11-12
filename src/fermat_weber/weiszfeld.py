from __future__ import annotations
import numpy as np
from .core import Array, f, init_from_nodes, node_optimal_condition, f_at_nodes

def weiszfeld(
    A: Array,
    w: Array,
    *,
    x0: Array | None = None,
    eps: float = 1e-9,
    max_iter: int = 500,
    tol: float = 1e-8,
    denom_eps: float = 1e-12,
) -> dict:
    """
    Metodo de Weiszfeld clásico (globalmente lineal). No usa derivadas.
    Devuelve {"x","f","k","hist"} con hist: [{"k","f"}].
    - Si no se pasa x0, usa init_from_nodes(A, w) del paper para arrancar mejor.
    - Si x0 coincide con un nodo óptimo (según condición de subgradiente), devuelve ese nodo.
    """

    if x0 is None:
         # es mucho mejor esto que un x0 cualquiera
        x = init_from_nodes(A, w).astype(float)
    else:
        x = np.asarray(x0, float).copy()

    # si x0 ya es un nodo óptimo
    for i in range(A.shape[0]):
        if np.allclose(x, A[i], atol=eps) and node_optimal_condition(i, A, w):
            return {"x": A[i], "f": f(A[i], A, w), "k": 0, "hist": []}

    hist: list[dict] = []
    for k in range(max_iter):
        fx = f(x, A, w)
        hist.append({"k": k, "f": float(fx)})

        diffs = x[None, :] - A           # (m,n)
        dists = np.linalg.norm(diffs, axis=1)

        # si x cae en un nodo paramos
        close_idx = np.where(dists < denom_eps)[0]
        if close_idx.size > 0:
            x = A[int(close_idx[0])].copy()
            break

        weights_over_dist = w / np.maximum(dists, denom_eps)
        x_new = (weights_over_dist[:, None] * A).sum(axis=0) / weights_over_dist.sum()

        if np.linalg.norm(x_new - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break

        x = x_new

    return {"x": x, "f": f(x, A, w), "k": k, "hist": hist}
