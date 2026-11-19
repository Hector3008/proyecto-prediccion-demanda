"""Carga y preparación de los snapshots de inventario y ventas.

El CSV de entrada tiene una fila por (código, fecha de corte). Las columnas de ventas,
devoluciones, importaciones y ajustes son contadores acumulados: lo que interesa de ellas
es su variación entre cortes, no su valor absoluto.
"""
from __future__ import annotations

import pandas as pd

COLS_NUM = ["stock_lv", "stock_sjl", "total", "ventas", "devolucion",
            "importacion", "ajuste_lv", "ajuste_sjl"]
# contadores acumulados: se les resta su valor en el primer registro de cada código
COLS_ACUMULADAS = ["ventas", "devolucion", "importacion", "ajuste_lv", "ajuste_sjl"]
COLS_REQUERIDAS = ["codigo", "fecha", "marca", "subfamilia"] + COLS_NUM


def cargar_snapshots(ruta: str) -> pd.DataFrame:
    """Lee el CSV, valida columnas, convierte tipos y deja una fila por (código, fecha)."""
    df = pd.read_csv(ruta, dtype={"codigo": str, "marca": str, "subfamilia": str})
    faltan = [c for c in COLS_REQUERIDAS if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en {ruta}: {faltan}")
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    for c in COLS_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.drop_duplicates(["codigo", "fecha"])
            .sort_values(["codigo", "fecha"])
            .reset_index(drop=True))
    return df


def agregar_columnas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas derivadas del pipeline original (wrangling).

    - primera_fecha_con_data: primera fecha en que aparece el código.
    - <col>_desde_primer_registro: variación acumulada desde esa primera fecha.
    - consumo_demanda: ventas netas de devoluciones (demanda acumulada).
    """
    d = df.sort_values(["codigo", "fecha"]).copy()
    g = d.groupby("codigo")
    d["primera_fecha_con_data"] = g["fecha"].transform("min")
    d["dias_historial"] = (d["fecha"] - d["primera_fecha_con_data"]).dt.days
    for c in COLS_ACUMULADAS:
        d[f"{c}_desde_primer_registro"] = d[c] - g[c].transform("first")
    d["consumo_demanda"] = d["ventas"] - d["devolucion"]
    d["consumo_demanda_desde_inicio"] = (d["ventas_desde_primer_registro"]
                                         - d["devolucion_desde_primer_registro"])
    return d


def ultimo_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por código en la última fecha disponible."""
    return df[df["fecha"] == df["fecha"].max()].drop_duplicates("codigo")


def metadatos_codigo(df: pd.DataFrame) -> pd.DataFrame:
    """Marca y subfamilia de cada código (último valor conocido)."""
    return (df.sort_values("fecha").groupby("codigo")[["marca", "subfamilia"]].last())
