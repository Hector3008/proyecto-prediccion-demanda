"""Construcción del dataset supervisado (código × origen) sin fuga de datos.

Para cada código y cada "origen" T (primera semana a pronosticar):
  - las variables usan SOLO las semanas anteriores a T;
  - el objetivo `y` es la demanda de las semanas T ... T+H-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .demanda import MODELOS, clasificar_serie, series_validas

H = 2            # horizonte: 2 semanas
MIN_SEM = 3      # semanas mínimas de historia para pronosticar (hay ~10 en total)
ROT_ETQ = ["baja", "media-baja", "media-alta", "alta"]

NUM_COLS = ["n_sem", "lag1", "lag2", "lag3", "media", "media_ult3", "desv", "maximo",
            "frac_semanas_con_venta", "semanas_desde_ult_venta", "tendencia",
            "adi", "cv2", "stock", "cobertura_sem", "agotado"]
CAT_COLS = ["marca", "subfamilia", "segmento", "rotacion"]
FEATURES = NUM_COLS + CAT_COLS

# modelos clásicos que se guardan como columnas bl_* (predicción a H semanas)
BASELINES = [m for m in MODELOS]          # cero, prom_hist, ma4, ses, croston, sba, tsb
COLS_BL = [f"bl_{m}" for m in BASELINES]


def _rotacion_por_origen(d: pd.DataFrame) -> pd.Series:
    """Cuartiles de venta semanal media entre los códigos con ventas, calculados POR origen."""
    rot = pd.Series("sin_ventas", index=d.index, dtype=object)
    for _, idx in d.groupby("T").groups.items():
        g = d.loc[idx]
        pos = g["media"] > 0
        if pos.sum() >= 4:
            rot.loc[g.index[pos]] = pd.qcut(g.loc[pos, "media"].rank(method="first"), 4,
                                            labels=ROT_ETQ).astype(str)
        else:
            rot.loc[g.index[pos]] = "media-alta"
    return rot


def construir_dataset(mat_dem: pd.DataFrame, cum_stock: pd.DataFrame, meta: pd.DataFrame,
                      h: int = H, min_sem: int = MIN_SEM) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por (código, origen T).

    mat_dem  : demanda semanal (código × semana).
    cum_stock: stock en los puntos semanales (n+1 columnas; la columna k = inicio de la semana k).
    meta     : marca y subfamilia por código.
    """
    filas = []
    stock_all = cum_stock.reindex(mat_dem.index).ffill(axis=1).to_numpy(float)
    pos_cod = {c: i for i, c in enumerate(mat_dem.index)}
    for cod, serie, ini, fin in series_validas(mat_dem):
        s = np.clip(serie, 0, None)
        for t in range(min_sem, len(s) - h + 1):
            hist = s[:t]
            adi, cv2, seg = clasificar_serie(hist, min_sem)
            nz = np.flatnonzero(hist > 0)
            stock = stock_all[pos_cod[cod], ini + t]
            media = float(hist.mean())
            fila = {
                "codigo": cod, "T": ini + t, "y": float(s[t:t + h].sum()),
                "n_sem": t,
                "lag1": hist[-1], "lag2": hist[-2], "lag3": hist[-3],
                "media": media, "media_ult3": float(hist[-3:].mean()),
                "desv": float(hist.std()), "maximo": float(hist.max()),
                "frac_semanas_con_venta": len(nz) / t,
                "semanas_desde_ult_venta": (t - 1 - nz[-1]) if len(nz) else t,
                "tendencia": float(hist[-2:].mean() - hist[:-2].mean()),
                "adi": adi, "cv2": cv2,
                "stock": max(stock, 0.0) if not np.isnan(stock) else np.nan,
                "agotado": float(stock <= 0) if not np.isnan(stock) else np.nan,
                "segmento": seg,
            }
            fila["cobertura_sem"] = (min(fila["stock"] / media, 52.0) if media > 0 and not np.isnan(stock)
                                     else (52.0 if not np.isnan(stock) else np.nan))
            for m in BASELINES:        # predicciones de los modelos clásicos a h semanas
                fila[f"bl_{m}"] = MODELOS[m](hist) * h
            filas.append(fila)
    d = pd.DataFrame(filas)
    d = d.join(meta[["marca", "subfamilia"]], on="codigo")
    d["marca"] = d["marca"].fillna("desconocida")
    d["subfamilia"] = d["subfamilia"].fillna("desconocida")
    d["rotacion"] = _rotacion_por_origen(d)
    return d.reset_index(drop=True)


def dividir_por_origen(d: pd.DataFrame, h: int = H) -> dict:
    """Particiones temporales (sin mezclar el futuro con el pasado).

    test      : último origen (T máximo), intocable hasta el final.
    purga     : origen T_test-1; su objetivo comparte la semana T_test con el de test.
    pool      : orígenes T <= T_test-h (entrenamiento + validación con CV temporal).
    validacion: último origen del pool; entrenamiento base = orígenes con T <= T_val-h.
    """
    t_test = int(d["T"].max())
    pool = d[d["T"] <= t_test - h]
    t_val = int(pool["T"].max())
    return {
        "t_test": t_test, "t_val": t_val,
        "test": d[d["T"] == t_test],
        "pool": pool,
        "train_base": pool[pool["T"] <= t_val - h],
        "validacion": pool[pool["T"] == t_val],
    }
