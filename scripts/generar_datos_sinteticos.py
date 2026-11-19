"""Genera datos SINTÉTICOS con la misma estructura que la muestra anonimizada.

Sirve para que el repositorio se pueda ejecutar sin datos privados. No son datos reales:
la demanda se simula con ventas intermitentes (probabilidad diaria de venta + tamaño aleatorio),
con la misma mezcla de rotaciones y el mismo calendario de cortes (espaciados al inicio y
diarios al final) que el dataset original.

Uso:  python scripts/generar_datos_sinteticos.py [--salida data/sample_snapshots.csv] [--n-codigos 450]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MARCAS = ["MARCA_A", "MARCA_B", "MARCA_C", "MARCA_D"]
PESOS_MARCA = [0.45, 0.25, 0.2, 0.1]
N_SUBFAM = 8


def calendario_cortes():
    """Cortes cada 3 días hasta fines de agosto y luego uno por día (como el dataset real)."""
    espaciados = pd.date_range("2025-07-14", "2025-08-31", freq="3D")
    diarios = pd.date_range("2025-09-01", "2025-09-25", freq="D")
    return espaciados.append(diarios)


def generar(n_codigos: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    dias = pd.date_range("2025-07-14", "2025-09-25", freq="D")
    cortes = set(calendario_cortes())
    filas = []
    for i in range(n_codigos):
        cod = f"COD-{i + 1:04d}"
        marca = rng.choice(MARCAS, p=PESOS_MARCA)
        subf = f"SUBF_{rng.integers(1, N_SUBFAM + 1):02d}"

        # perfil de demanda: rara / intermitente / regular
        u = rng.random()
        if u < 0.08:
            p_dia, lam = 0.0, 0.0                                    # sin ventas
        elif u < 0.45:
            p_dia, lam = rng.uniform(0.01, 0.07), rng.uniform(0.3, 1.5)    # venta espaciada
        elif u < 0.80:
            p_dia, lam = rng.uniform(0.08, 0.3), rng.uniform(0.5, 3.0)     # intermitente media
        else:
            p_dia, lam = rng.uniform(0.4, 0.9), rng.uniform(1.0, 8.0)      # rotación alta
        tendencia = rng.normal(0, 0.004)                              # cambio lento del nivel
        inicio = 0 if rng.random() > 0.08 else int(rng.integers(5, 35))   # algunos códigos entran después

        stock = int(rng.integers(0, 60)) + int(lam * p_dia * rng.integers(20, 120))
        base = {c: int(rng.integers(0, 400)) for c in ("ventas", "devolucion", "importacion", "ajuste_lv", "ajuste_sjl")}
        for k, dia in enumerate(dias):
            es_finde = dia.weekday() >= 5
            p = np.clip(p_dia * (1 + tendencia * k) * (0.15 if es_finde else 1.0), 0, 0.98)
            venta = int(1 + rng.poisson(lam)) if rng.random() < p else 0
            venta = min(venta, max(stock, 0) + 3)
            stock -= venta
            base["ventas"] += venta
            if venta and rng.random() < 0.03:
                dev = 1
                base["devolucion"] += dev
                stock += dev
            if stock <= max(2, 3 * lam * p_dia) and rng.random() < 0.04:       # reposición
                imp = int(rng.integers(20, 150))
                base["importacion"] += imp
                stock += imp
            if rng.random() < 0.006:                                            # ajuste por auditoría
                aj = int(rng.integers(-3, 6))
                base["ajuste_lv"] += aj
                stock += aj
            if dia in cortes and k >= inicio:
                lv = int(max(stock, 0) * 0.65) if stock > 0 else stock
                filas.append({"codigo": cod, "fecha": dia.date().isoformat(), "marca": marca, "subfamilia": subf,
                              "stock_lv": lv, "stock_sjl": stock - lv, "total": stock, **base})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salida", default="data/sample_snapshots.csv")
    ap.add_argument("--n-codigos", type=int, default=450)
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    df = generar(a.n_codigos, a.semilla)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.salida, index=False)
    print(f"{len(df):,} filas | {df['codigo'].nunique()} códigos | {df['fecha'].nunique()} cortes -> {a.salida}")
