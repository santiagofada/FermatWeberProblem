import argparse
from pathlib import Path
from typing import Optional
import sys

import numpy as np

# Allow running from project root without setting PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from fermat_weber.io import read_instance
from fermat_weber.weiszfeld import weiszfeld
from fermat_weber.newton import newton_armijo

# Plotting is optional; only import when needed

def _run_method(method: str, A: np.ndarray, w: np.ndarray):
    method = method.lower()
    if method in {"weiszfeld", "w"}:
        res = weiszfeld(A, w)
        title = "Weiszfeld"
    elif method in {"newton", "n"}:
        res = newton_armijo(A, w)
        title = "Newton"
    else:
        raise ValueError("Unknown method. Use 'weiszfeld' or 'newton'.")
    return title, res


def _maybe_plot(A: np.ndarray, w: np.ndarray, res: dict, *, title: str,
                save_path: Optional[Path] = None, show: bool = False) -> None:
    try:
        from utils.plot_utils import plot_f_and_points
    except Exception:
        utils_dir = _PROJECT_ROOT / "utils"
        if str(utils_dir) not in sys.path:
            sys.path.insert(0, str(utils_dir))
        try:
            from plot_utils import plot_f_and_points
        except Exception as e:
            raise RuntimeError(
                "Plotting requires matplotlib and plot_utils. "
                "Ensure matplotlib is installed and utils/plot_utils.py exists."
            ) from e

    import matplotlib.pyplot as plt

    fig = plot_f_and_points(
        hist=res.get("hist", []), A=A, w=w, x=res.get("x"),
        method_title=title, layout="1x2"
    )
    if save_path is not None:
        # If a directory path was provided, save a default filename inside it
        if save_path.exists() and save_path.is_dir():
            save_path = save_path / f"{title.lower()}_plot.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fermat–Weber from input file (CSV/JSON)")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input file (.csv or .json)")
    parser.add_argument("--method", "-m", default="weiszfeld", choices=["weiszfeld", "newton"],
                        help="Method to run")
    parser.add_argument("--save-plot", default=None, help="Path to save plot (optional)")
    parser.add_argument("--show", action="store_true", help="Show plot window (optional)")
    args = parser.parse_args()

    input_path = Path(args.input)
    A, w, info = read_instance(str(input_path))

    title, res = _run_method(args.method, A, w)

    x = res.get("x")
    fx = res.get("f")
    k = res.get("k")
    print(f"Method: {title}")
    print(f"x*: {x}")
    print(f"f(x*): {fx}")
    print(f"iterations: {k}")

    save_path = Path(args.save_plot) if args.save_plot else None
    if save_path is not None or args.show:
        _maybe_plot(A, w, res, title=title, save_path=save_path, show=args.show)


if __name__ == "__main__":
    main()
