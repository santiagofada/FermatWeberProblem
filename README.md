# Fermat–Weber Problem

Project for an Optimization course focused on solving the Fermat–Weber problem (weighted median point) using numerical methods, with visualization utilities and analysis notebooks.

## Description
Given a set of points (nodes) `A = {a_i}` with weights `w_i > 0`, the problem is to find `x*` that minimizes:

$$
\min_{\,x\in\mathbb{R}^n} \; f(x) \;=\; \sum_{i=1}^m w_i \;\lVert x - a_i \rVert_2
$$

This repository includes:
- Implementation of `f`, its gradient and Hessian (when applicable), and core utilities.
- Weiszfeld method (derivative-free, linear convergence).
- Globalized Newton method with Armijo backtracking line search.
- Utilities to plot convergence history and 2D configuration.
- Analysis and test notebooks.

## Mathematical formulation
- Objective (Euclidean norm):

  $$
  f(x) = \sum_{i=1}^m w_i \, \lVert x - a_i \rVert_2
  $$

- Gradient for $x \neq a_i$ (Euclidean case $\lVert\cdot\rVert_2$):

  $$
  \nabla f(x) = \sum_{i=1}^m w_i \, \frac{x - a_i}{\lVert x - a_i \rVert_2}
  $$

- Hessian (Euclidean norm):
 
  $$
  \nabla^2 f(x) 
  = \sum_{i=1}^m w_i \left( \frac{I}{\lVert x - a_i \rVert_2} -
  \frac{(x - a_i)(x - a_i)^\top}{\lVert x - a_i \rVert_2^{\,3}} \right)
  $$

- Sufficient optimality condition at a node $a_p$:

  $$
  \Big\lVert \sum_{\substack{i=1 \\ i\ne p}}^m w_i \, \frac{a_i - a_p}{\lVert a_i - a_p \rVert_2} \Big\rVert_2 \;\le\; w_p
  $$

- Weiszfeld iteration (when $x_k$ does not coincide with a node):

  $$
  x_{k+1} 
  = \frac{\sum_{i=1}^m \dfrac{w_i}{\lVert x_k - a_i \rVert_2} \, a_i}
         {\sum_{i=1}^m \dfrac{w_i}{\lVert x_k - a_i \rVert_2}}
  $$

- Newton step with line search (Armijo):

  $$
  d_k = -\,\big(\nabla^2 f(x_k)\big)^{-1} \, \nabla f(x_k),
  \quad x_{k+1} = x_k + t_k d_k, \; t_k > 0
  $$

## Repository structure
- `src/fermat_weber/core.py`: Core functions (`f`, `grad`, `hess`), `init_from_nodes`, and node optimality check.
- `src/fermat_weber/weiszfeld.py`: Weiszfeld implementation (`weiszfeld`).
- `src/fermat_weber/newton.py`: Newton with Armijo implementation (`newton_armijo`).
- `scripts/plot_utils.py`: Plotting utilities for `f` history and 2D configuration.
- `notebooks/`: Analysis notebooks (`cualitative_analysis.ipynb`, `tests.ipynb`).
- `documments/`: Documents fot the course (`2025_OPTI_report.pdf` for the proyect investigation, `2025_OPTI_Presentation.pdf` with the slides).

## Requirements
- Python 3.9+
- numpy
- matplotlib (for plots)

## Quick usage

```python
import numpy as np
from fermat_weber.weiszfeld import weiszfeld
from fermat_weber.newton import newton_armijo

# Nodes in R² and weights
A = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
w = np.array([1.0, 2.0, 1.0])

# Weiszfeld
res_w = weiszfeld(A, w)
print("Weiszfeld:", res_w["x"], res_w["f"], res_w["k"]) 

# Newton
res_n = newton_armijo(A, w)
print("Newton:", res_n["x"], res_n["f"], res_n["k"]) 
```

## Run from file (CLI)

Use the helper script to load an instance from a file and run a method:

```bash
python scripts/run_from_file.py --input data/example.json --method weiszfeld
python scripts/run_from_file.py --input data/example.csv  --method newton
```

Optional plotting:

```bash
python scripts/run_from_file.py --input data/example.json --method weiszfeld --save-plot out/plot.png
python scripts/run_from_file.py --input data/example.json --method weiszfeld --show
```

Notes:
- Run commands from the project root. The script auto-adds `./src` to `sys.path`.
- Supported input formats: `.json` and `.csv` (see I/O utilities below).

## Visualization (optional)
You can plot the history of `f` and the 2D configuration using `scripts/plot_utils.py`.

```python
import numpy as np
from fermat_weber.weiszfeld import weiszfeld
from utils.plot_utils import plot_f_and_points
import matplotlib.pyplot as plt

A = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
w = np.array([1.0, 2.0, 1.0])
res = weiszfeld(A, w)

fig = plot_f_and_points(
    hist=res["hist"], A=A, w=w, x=res["x"],
    method_title="Weiszfeld", layout="1x2"
)
plt.show()
```

## I/O utilities

Read/write instances programmatically (`.json` or `.csv`).

```python
from fermat_weber.io import read_instance, write_instance
import numpy as np

A, w, info = read_instance("data/example.json")

write_instance("out/my_instance.json", A, w, info={"meta": {"note": "demo"}})

write_instance("out/my_instance.csv", A, w, info={"id": ["A","B","C"]})
```

Notes:
- The methods return dictionaries with at least `x` (solution), `f` (objective value), `k` (iterations) and `hist` (history).
- For R^2, `plot_points_and_solution_2d` draws nodes (size proportional to `w`) and the solution `x*`.
