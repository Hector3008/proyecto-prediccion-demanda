"""Métricas de error para demanda. La métrica principal es WAPE.

WAPE = sum(|pred - real|) / sum(real). Se prefiere sobre MAPE porque la demanda tiene muchos
ceros (MAPE no está definido) y porque pondera por volumen: un error en un código que vende
mucho cuesta más que en uno que casi no vende, que es lo que importa para compras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


def _arr(y_true, y_pred):
    y = np.asarray(y_true, float)
    p = np.clip(np.asarray(y_pred, float), 0, None)   # la demanda no puede ser negativa
    return y, p


def wape(y_true, y_pred) -> float:
    y, p = _arr(y_true, y_pred)
    d = y.sum()
    return float("nan") if d <= 0 else float(np.abs(p - y).sum() / d)


def sesgo(y_true, y_pred) -> float:
    """(sum pred - sum real) / sum real. Negativo = se predice de menos (riesgo de quiebre)."""
    y, p = _arr(y_true, y_pred)
    d = y.sum()
    return float("nan") if d <= 0 else float((p - y).sum() / d)


def faltante(y_true, y_pred) -> float:
    """Parte de la demanda real que la predicción no cubre (costo de quiebre)."""
    y, p = _arr(y_true, y_pred)
    d = y.sum()
    return float("nan") if d <= 0 else float(np.clip(y - p, 0, None).sum() / d)


def exceso(y_true, y_pred) -> float:
    """Predicción por encima de lo real, sobre la demanda real (costo de sobrestock)."""
    y, p = _arr(y_true, y_pred)
    d = y.sum()
    return float("nan") if d <= 0 else float(np.clip(p - y, 0, None).sum() / d)


def cobertura(y_true, y_pred) -> float:
    """% de casos con venta real en que la predicción alcanzó a cubrirla."""
    y, p = _arr(y_true, y_pred)
    pos = y > 0
    return float("nan") if not pos.any() else float((p[pos] >= y[pos]).mean())


# scorer para sklearn: GridSearchCV/RandomizedSearchCV maximizan, por eso es -WAPE
wape_scorer = make_scorer(wape, greater_is_better=False)


def resumen(y_true, y_pred) -> dict:
    return {"WAPE": wape(y_true, y_pred), "sesgo": sesgo(y_true, y_pred),
            "faltante": faltante(y_true, y_pred), "exceso": exceso(y_true, y_pred),
            "cobertura": cobertura(y_true, y_pred)}


def tabla_metricas(y_true, preds: dict) -> pd.DataFrame:
    """Una fila por modelo: WAPE, sesgo, faltante, exceso y cobertura."""
    return pd.DataFrame({k: resumen(y_true, v) for k, v in preds.items()}).T
