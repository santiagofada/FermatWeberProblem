from __future__ import annotations
import numpy as np

Array = np.ndarray


def f(x: Array, A: Array, w: Array, norm: int | float = 2) -> float:
    """
    f(x) = Σ w_i ||x - a_i||_p
    (Por defecto usa la norma Euclídea p=2)
    """
    total = 0.0
    for i in range(A.shape[0]):
        total += w[i] * np.linalg.norm(x - A[i], ord=norm)
    return float(total)



def grad(x: Array, A: Array, w: Array, norm: int | float = 2, eps: float = 1e-10) -> Array:
    """
    Gradiente aproximado: ∇f(x) = Σ w_i (x - a_i)/||x - a_i||_p
    (fórmula exacta solo válida para p=2, pero el código generaliza a otras normas si se ajusta el denominador)
    """
    n = x.size
    grad_vector = np.zeros(n)
    for i in range(A.shape[0]):
        diff = x - A[i]
        r = np.linalg.norm(diff, ord=norm)
        if r < eps:
            raise ValueError(f"La función no es diferenciable en x = a_{i}, x conincide con un nodo a")
        grad_vector += (w[i] * diff) / r
    return grad_vector


def hess(x: Array, A: Array, w: Array, norm: int | float = 2, eps: float = 1e-10) -> Array:
    """
    ∇²f(x) = Σ w_i [I/||x - a_i|| - (x - a_i)(x - a_i)^T / ||x - a_i||³]
    (válida solo para la norma Euclídea, pero mantiene compatibilidad de interfaz con otras normas)
    """
    n = x.size
    H = np.zeros((n, n))

    for i in range(A.shape[0]):
        diff = x - A[i]
        r = np.linalg.norm(diff, ord=norm)
        if r < eps:
            raise ValueError(f"La función no es diferenciable en x = a_{i}, x conincide con un nodo a")
        u = diff / r
        H += w[i] * (np.eye(n) / r - np.outer(u, u) / r)

    return H


def f_at_nodes(A: Array, w: Array, norm: int | float = 2) -> Array:
    """
    Calcula el vector [f(a_1), ..., f(a_m)], donde
        f(a_i) = Σ_j w_j ||a_i - a_j||_p
    utilizando la función f() definida anteriormente.
    """
    m = A.shape[0]
    f_values = np.zeros(m, dtype=float)
    for i in range(m):
        f_values[i] = f(A[i], A, w, norm=norm)

    return f_values


def node_optimal_condition(p: int, A: Array, w: Array, eps: float = 1e-12) -> bool:
    """
    Verifica la condición suficiente de optimalidad en el nodo a_p:
        || Σ_{i≠p} w_i (a_i - a_p)/||a_i - a_p|| || <= w_p
    Retorna True si se cumple la condición (a_p es óptimo), False en caso contrario.
    """
    ap = A[p]
    n = A.shape[1]
    R_sum = np.zeros(n, dtype=float)

    # peso efectivo en a_p (acumula pesos de nodos coincidentes)
    w_p_eff = float(w[p])

    for i in range(A.shape[0]):
        if i == p:
            continue
        diff = A[i] - ap
        dist = float(np.linalg.norm(diff))
        if dist < eps:
            # nodo coincidente con a_p: acumular su peso y no sumar dirección
            w_p_eff += float(w[i])
            continue
        R_sum += float(w[i]) * (diff / dist)

    return float(np.linalg.norm(R_sum)) <= w_p_eff

def init_from_nodes(A: Array, w: Array, eps: float = 1e-12) -> Array:
    """
    1) Elegir p con f(a_p) mínimo.
    2) Si a_p cumple condición de subgradiente, devolver a_p.
    3) Si no, construir x^(0) = a_p + t_p d_p con:
       d_p = -R_p/||R_p||,  R_p = sum_{i≠p} w_i (a_p - a_i)/||a_p-a_i||
       t_p = (||R_p|| - w_p) / sum_{i≠p} w_i/||a_p-a_i||
    """
    # 1) p tal que f(a_p) es mínimo
    fa = f_at_nodes(A, w, norm=2)
    p = int(np.argmin(fa))
    ap = A[p].copy()

    # 2) si a_p ya es óptimo, terminar
    if node_optimal_condition(p, A, w, eps=eps):
        return ap

    # 3) construir x^(0) con las expresiones del paper
    m, n = A.shape
    Rp = np.zeros(n, dtype=float)
    L  = 0.0

    for i in range(m):
        if i == p:
            continue
        diff = ap - A[i]
        dist = float(np.linalg.norm(diff))

        Rp += float(w[i]) * (diff / dist)
        L  += float(w[i]) / dist

    Rp_norm = float(np.linalg.norm(Rp))

    dp = -Rp / Rp_norm

    tp = (Rp_norm - float(w[p])) / L

    return ap + tp * dp
