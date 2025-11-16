from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hist_to_arrays(hist: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    ks = np.array([h["k"] for h in hist], dtype=float)
    fs = np.array([h["f"] for h in hist], dtype=float)
    return ks, fs


# ---------- plots atómicos ----------
def plot_f_history(
    hist: Sequence[dict],
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "f vs iteraciones",
    annotate_last: bool = True,
) -> plt.Axes:
    """Línea de f(x_k) vs k. Devuelve el Axes usado."""
    ks, fs = _hist_to_arrays(hist)
    ax = ax or plt.gca()
    ax.plot(ks, fs, marker="o",c="mediumseagreen")
    if annotate_last and fs.size:
        ax.annotate(f"{fs[-1]:.6g}", (ks[-1], fs[-1]), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Iteración k")
    ax.set_ylabel("$f(x_k)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.figure.tight_layout()
    return ax


def plot_points_and_solution_2d(
    A: Array,
    w: Array,
    x: Array,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Configuración en R²",
    path: Optional[Array] = None,
    path_markers: bool = True,
    size_scale: float = 80.0,
    equal_aspect: bool = True,
) -> plt.Axes:
    """
    Nodos (tamaño ~ w) y solución x*. Opcionalmente dibuja la trayectoria 'path' si la pasás.
    """
    A = np.asarray(A, float)
    w = np.asarray(w, float).ravel()
    x = np.asarray(x, float).ravel()
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("Este plot es solo para R² (A debe ser (m,2)).")

    ax = ax or plt.gca()
    ax.scatter(A[:, 0], A[:, 1], s=w * size_scale, label="Nodos $a_i$",c="teal")
    if path is not None:
        P = np.asarray(path, float)
        if P.ndim == 2 and P.shape[1] == 2 and P.shape[0] >= 2:
            ax.plot(P[:, 0], P[:, 1], marker="o" if path_markers else None,
                    label="Trayectoria $x_k$")
    ax.scatter(x[0], x[1], marker="*", s=150, label="$x^*$",c="crimson")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc=1)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    return ax


def plot_f_and_points(
    hist: Sequence[dict],
    A: Array,
    w: Array,
    x: Array,
    *,
    path: Optional[Array] = None,
    layout: str = "1x2",          # "1x2" o "2x1"
    method_title: str = "Método", # <--- nuevo: título genérico del método
    title_f: str = "f vs iteraciones",
    title_points: str = "Configuración en R²",
    figsize: Tuple[float, float] = (10, 4.5),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Figura con dos subplots: f(k) y nodos,x* si A es R².
    """
    if layout not in {"1x2", "2x1"}:
        raise ValueError("layout debe ser '1x2' o '2x1'.")

    if layout == "1x2":
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize)

    ax_f, ax_pts = axes[0], axes[1]

    plot_f_history(hist, ax=ax_f, title=title_f, annotate_last=True)


    # Si no es R², muestro un mensaje de control
    if np.asarray(A).ndim == 2 and np.asarray(A).shape[1] == 2:
        plot_points_and_solution_2d(A, w, x, ax=ax_pts, title=title_points,
                                    path=path, path_markers=True)
    else:
        ax_pts.axis("off")
        ax_pts.text(0.5, 0.5, "Visualización solo para R²",
                    ha="center", va="center", fontsize=11)

    fig.suptitle(f"Evolución — {method_title}", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180)

    return fig