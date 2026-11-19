"""Series semanales, segmentación ADI/CV² y modelos clásicos de demanda intermitente."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Umbrales de Syntetos-Boylan para clasificar la demanda
ADI_CORTE, CV2_CORTE = 1.32, 0.49


# --------------------------------------------------------------------------- matrices
def matriz_acumulada_semanal(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Código × punto semanal del valor de `col`.

    Los cortes no son diarios (al inicio están espaciados), así que se interpola
    día a día y se toman puntos cada 7 días terminando en la última fecha.
    Devuelve n+1 columnas (puntos); n semanas completas.
    """
    piv = df.pivot(index="codigo", columns="fecha", values=col)
    dias = pd.date_range(piv.columns.min(), piv.columns.max(), freq="D")
    piv = piv.reindex(columns=dias).interpolate(axis=1, limit_area="inside")
    n = (dias[-1] - dias[0]).days // 7
    puntos = pd.date_range(end=dias[-1], periods=n + 1, freq="7D")
    return piv[puntos]


def matriz_semanal(df: pd.DataFrame, col: str = "consumo_demanda") -> pd.DataFrame:
    """Demanda semanal por código (diferencia del acumulado). NaN = aún sin datos."""
    return matriz_acumulada_semanal(df, col).diff(axis=1).iloc[:, 1:]


def series_validas(mat: pd.DataFrame):
    """Itera (código, serie contigua, índice inicial, índice final).

    Descarta series con huecos internos (códigos que desaparecen y reaparecen).
    """
    A = mat.to_numpy(float)
    for i, cod in enumerate(mat.index):
        fila = A[i]
        ok = ~np.isnan(fila)
        if not ok.any():
            continue
        ini = int(np.argmax(ok))
        fin = len(fila) - int(np.argmax(ok[::-1]))
        serie = fila[ini:fin]
        if np.isnan(serie).any():
            continue
        yield cod, serie, ini, fin


# --------------------------------------------------------------------------- segmentación
def clasificar_serie(x, min_sem: int = 3):
    """Devuelve (ADI, CV², segmento) usando solo la serie recibida (sin fuga de datos)."""
    x = np.clip(np.asarray(x, float), 0, None)
    n = len(x)
    if n < min_sem:
        return np.nan, np.nan, "historia_corta"
    nz = x[x > 0]
    if len(nz) == 0:
        return float(n), 0.0, "sin_ventas"
    adi = n / len(nz)
    cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
    if adi < ADI_CORTE:
        forma = "smooth" if cv2 < CV2_CORTE else "erratic"
    else:
        forma = "intermittent" if cv2 < CV2_CORTE else "lumpy"
    return adi, cv2, forma


# --------------------------------------------------------------------------- modelos clásicos
def _ses(x, alpha=0.3):
    nivel = x[0]
    for v in x[1:]:
        nivel = alpha * v + (1 - alpha) * nivel
    return float(nivel)


def _croston(x, alpha=0.2):
    nz = np.flatnonzero(x > 0)
    if len(nz) == 0:
        return 0.0
    z = float(x[nz[0]])      # tamaño medio de la demanda
    p = float(nz[0] + 1)     # intervalo medio entre demandas
    q = 1
    for v in x[nz[0] + 1:]:
        if v > 0:
            z += alpha * (v - z)
            p += alpha * (q - p)
            q = 1
        else:
            q += 1
    return z / p


def _sba(x, alpha=0.2):
    return _croston(x, alpha) * (1 - alpha / 2)   # corrección de sesgo de Syntetos-Boylan


def _tsb(x, alpha=0.2, beta=0.2):
    if not np.any(x > 0):
        return 0.0
    k = min(3, len(x))
    p = float(np.mean(x[:k] > 0))
    nz0 = x[:k][x[:k] > 0]
    z = float(nz0.mean()) if len(nz0) else float(x[x > 0][0])
    for v in x[k:]:
        if v > 0:
            p += beta * (1 - p)
            z += alpha * (v - z)
        else:
            p *= (1 - beta)
    return p * z


# Nivel semanal esperado a partir de la serie histórica (unidades por semana).
# alpha/beta altos a propósito: con ~10 semanas de historia, 0.1 casi no se mueve del valor inicial.
MODELOS = {
    "cero": lambda x: 0.0,
    "prom_hist": lambda x: float(np.mean(x)),
    "ma4": lambda x: float(np.mean(x[-4:])),
    "ses": _ses,
    "croston": _croston,
    "sba": _sba,
    "tsb": _tsb,
}
