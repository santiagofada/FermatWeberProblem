from __future__ import annotations
import numpy as np
from .core import Array, f, grad, hess, init_from_nodes, node_optimal_condition


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

def newton_armijo(
    A: Array,
    w: Array,
    *,
    rho: float = 0.5,
    sigma: float = 1e-4,
    eps_grad: float = 1e-9,
    max_iter: int = 200,
    denom_eps: float = 0.0,
    hess_reg: float = 0.0,
) -> dict:
    """
    Newton globalizado con búsqueda de línea de Armijo para el problema de Fermat–Weber.

    Devuelve un diccionario con:
      - x: punto final
      - f: valor f(x)
      - k: número de iteraciones realizadas
      - hist: lista de dicts con {"k", "f", "norm_grad"}

    Parámetros clave
    ----------------
    rho       : factor de retroceso para Armijo (0<rho<1).
    sigma     : parámetro de Armijo (pequeño; p.ej. 1e-4).
    eps_grad  : tolerancia para ||∇f||.
    max_iter  : máximo de iteraciones de Newton.
    denom_eps : si >0, se usa como salvaguarda en grad/hess cuando ||x-a_i|| es muy chico.
    hess_reg  : si >0, agrega λI a la Hessiana antes de resolver (estabilización).
    """
    x0 = init_from_nodes(A, w)

    # Si x0 es un nodo que ya cumple óptimo, terminar
    for i in range(A.shape[0]):
        if np.allclose(x0, A[i]) and node_optimal_condition(i, A, w):
            return {"x": x0, "f": f(x0, A, w), "k": 0, "hist": []}

    x = x0.astype(float)
    hist: list[dict] = []

    # Parámetros internos de robustez para Armijo
    max_backtracks = 50
    t_min = 1e-12

    for k in range(max_iter):
        g = grad(x, A, w, eps=denom_eps)
        norm_grad = float(np.linalg.norm(g))
        fx = f(x, A, w)
        hist.append({"k": k, "f": fx, "norm_grad": norm_grad})

        if norm_grad <= eps_grad:
            break

        # Dirección de Newton (con regularización opcional)
        H = hess(x, A, w, eps=denom_eps)
        if hess_reg > 0.0:
            H = H + hess_reg * np.eye(H.shape[0])

        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # fallback con regularización adaptativa simple
            lam = 1e-8
            solved = False
            for _ in range(6):
                try:
                    d = np.linalg.solve(H + lam * np.eye(H.shape[0]), -g)
                    solved = True
                    break
                except np.linalg.LinAlgError:
                    lam *= 10
            if not solved:
                d = -g  # último recurso: descenso por gradiente

        # Asegurar dirección de descenso
        slope = float(np.dot(g, d))
        if slope >= 0.0:
            d = -g
            slope = -float(np.dot(g, g))

        # Armijo (backtracking)
        t = 1.0
        bt = 0
        while f(x + t * d, A, w) > fx + sigma * t * slope:
            t *= rho
            bt += 1
            if bt >= max_backtracks or t < t_min:
                break

        # Si el paso es demasiado pequeño, detener (evita bucles infinitos)
        if t < t_min:
            break

        x = x + t * d

    return {"x": x, "f": f(x, A, w), "k": k, "hist": hist}
