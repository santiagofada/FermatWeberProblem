from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _is_number(x: Any) -> bool:
    try:
        return not isinstance(x, bool) and math.isfinite(float(x))
    except Exception:
        return False


def _parse_csv(path: Path) -> Tuple[Array, Array, Dict[str, Sequence[Any]]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV requires a header row with column names")
        headers = [h.strip() for h in reader.fieldnames]

        x_cols: list[Tuple[int, str]] = []
        for h in headers:
            if h.lower().startswith("x"):
                idx_part = h[1:]
                if idx_part.isdigit():
                    x_cols.append((int(idx_part), h))
        if not x_cols:
            raise ValueError("CSV must contain at least one coordinate column named x1,x2,...,xN")
        x_cols.sort(key=lambda t: t[0])
        x_headers = [h for _, h in x_cols]

        has_w = any(h.lower() == "w" for h in headers)
        has_id = any(h.lower() == "id" for h in headers)
        has_label = any(h.lower() == "label" for h in headers)

        rows = list(reader)
        if not rows:
            raise ValueError("CSV has no data rows")

        m = len(rows)
        n = len(x_headers)
        A = np.zeros((m, n), dtype=float)
        w = np.ones(m, dtype=float)
        ids: list[Any] = []
        labels: list[Any] = []

        for i, row in enumerate(rows):
            for j, h in enumerate(x_headers):
                val = row.get(h, None)
                if val is None or not _is_number(val):
                    raise ValueError(f"Invalid or missing value for column '{h}' at row {i+2}")
                A[i, j] = float(val)
            if has_w:
                valw = row.get("w", row.get("W"))
                if valw is None or not _is_number(valw):
                    raise ValueError(f"Invalid or missing weight 'w' at row {i+2}")
                w[i] = float(valw)
            if has_id:
                ids.append(row.get("id"))
            if has_label:
                labels.append(row.get("label"))

        if np.any(~np.isfinite(A)):
            raise ValueError("Non-finite values found in coordinates")
        if np.any(~np.isfinite(w)):
            raise ValueError("Non-finite values found in weights")
        if np.any(w <= 0):
            raise ValueError("Weights must be strictly positive")

        info: Dict[str, Sequence[Any]] = {}
        if has_id:
            info["id"] = ids
        if has_label:
            info["label"] = labels
        return A, w, info


def _parse_json(path: Path) -> Tuple[Array, Array, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object with keys like 'A' and 'w'")

    if "A" not in data or "w" not in data:
        raise ValueError("JSON must contain keys 'A' and 'w'")

    A = np.asarray(data["A"], dtype=float)
    w = np.asarray(data["w"], dtype=float).ravel()

    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError("'A' must be a non-empty 2D array")
    if w.ndim != 1 or w.shape[0] != A.shape[0]:
        raise ValueError("'w' must be a 1D array with the same length as rows of 'A'")
    if not np.all(np.isfinite(A)):
        raise ValueError("Non-finite values found in 'A'")
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite values found in 'w'")
    if np.any(w <= 0):
        raise ValueError("Weights must be strictly positive")

    info: Dict[str, Any] = {}
    for k in ("id", "label", "meta"):
        if k in data:
            info[k] = data[k]
    return A, w, info


def read_instance(path: str | Path) -> Tuple[Array, Array, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if ext == ".csv":
        return _parse_csv(p)
    if ext == ".json":
        return _parse_json(p)
    raise ValueError("Unsupported file extension. Use .csv or .json")


def write_instance(
    path: str | Path,
    A: Array,
    w: Array,
    info: Optional[Dict[str, Any]] = None,
) -> None:
    p = Path(path)
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError("'A' must be a non-empty 2D array")
    if w.ndim != 1 or w.shape[0] != A.shape[0]:
        raise ValueError("'w' must be a 1D array with the same length as rows of 'A'")
    if np.any(w <= 0):
        raise ValueError("Weights must be strictly positive")

    ext = p.suffix.lower()
    info = info or {}

    if ext == ".json":
        out: Dict[str, Any] = {"A": A.tolist(), "w": w.tolist()}
        for k in ("id", "label", "meta"):
            if k in info:
                out[k] = info[k]
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return

    if ext == ".csv":
        import csv

        n = A.shape[1]
        headers = [f"x{i+1}" for i in range(n)] + ["w"]
        has_id = "id" in info and isinstance(info["id"], Sequence)
        has_label = "label" in info and isinstance(info["label"], Sequence)
        if has_id:
            headers = ["id"] + headers
        if has_label:
            headers = headers + ["label"]

        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            m = A.shape[0]
            for i in range(m):
                row = []
                if has_id:
                    row.append(info["id"][i])
                row.extend([*map(lambda v: float(v), A[i].tolist()), float(w[i])])
                if has_label:
                    row.append(info["label"][i])
                writer.writerow(row)
        return

    raise ValueError("Unsupported file extension. Use .csv or .json")
