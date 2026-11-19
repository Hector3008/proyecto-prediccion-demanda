"""Convierte el df.csv real (salida del pipeline `setup`) en la muestra anonimizada del repositorio.

EJECUTAR SOLO EN TU ENTORNO (Colab/PC), NUNCA subir el df.csv original. Qué hace:
  1. Consolida los VARIADO con la suma de sus VARIABLE (mismo criterio del wrangling original).
  2. Se queda con los registros VARIADO / sin carácter (los VARIABLE ya están sumados) y excluye PLAZA.
  3. Enmascara códigos (COD-0001...), marcas (MARCA_A...) y subfamilias (SUBF_01...) con una
     permutación aleatoria; elimina textos libres (detalle, nombre comercial, equivalentes, precio,
     proveedor, ubicación).
  4. Opcionalmente toma una muestra aleatoria de códigos.
  5. Verifica que ningún código original quede en el archivo de salida.

Uso (Colab):
    !python scripts/anonimizar_df.py --entrada /content/drive/MyDrive/HH/ARCHIVE/colab/csv/df.csv \
        --salida data/sample_snapshots.csv --n-codigos 600
El mapa original -> anónimo NO se guarda salvo que pidas --guardar-mapa (y entonces fuera del repo).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS_CONSOLIDAR = ["ventas", "movimientos", "devolucion", "importacion", "ajuste_lv", "ajuste_sjl"]
COLS_SALIDA = ["codigo", "fecha", "marca", "subfamilia", "stock_lv", "stock_sjl", "total",
               "ventas", "devolucion", "importacion", "ajuste_lv", "ajuste_sjl"]


def consolidar_variados(df: pd.DataFrame) -> pd.DataFrame:
    """VARIADO = suma de sus VARIABLE por (fecha, código base). Versión vectorizada."""
    df = df.copy()
    df["codigo_base"] = df["codigo"].astype(str).str.split("-").str[-1]
    var = df[df["CARACTER"] == "VARIABLE"]
    suma = var.groupby(["fecha", "codigo_base"])[COLS_CONSOLIDAR].sum().add_suffix("__suma").reset_index()
    df = df.merge(suma, on=["fecha", "codigo_base"], how="left")
    es_variado = df["CARACTER"] == "VARIADO"
    for c in COLS_CONSOLIDAR:
        s = df[f"{c}__suma"]
        df.loc[es_variado & s.notna(), c] = s[es_variado & s.notna()]
    return df.drop(columns=[f"{c}__suma" for c in COLS_CONSOLIDAR])


def anonimizar(df: pd.DataFrame, n_codigos: int | None, semilla: int):
    rng = np.random.default_rng(semilla)
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce"))
    df = consolidar_variados(df)
    df = df[df["CARACTER"] != "VARIABLE"]
    df = df[df["MARCA"].notna() & (df["MARCA"].astype(str).str.upper() != "PLAZA")]
    df = df.rename(columns={"MARCA": "marca", "SUBFAMILIA": "subfamilia"})
    df["subfamilia"] = df["subfamilia"].astype(str).str.strip().str.upper()

    originales = df["codigo"].astype(str).unique()
    if n_codigos and n_codigos < len(originales):
        elegidos = set(rng.choice(originales, n_codigos, replace=False))
        df = df[df["codigo"].astype(str).isin(elegidos)]
        originales = df["codigo"].astype(str).unique()

    def mapa(valores, prefijo, ancho=0, letras=False):
        v = np.array(sorted(valores))
        perm = rng.permutation(len(v))
        if letras:
            return {o: f"{prefijo}{chr(65 + int(i))}" for o, i in zip(v, perm)}
        return {o: f"{prefijo}{int(i) + 1:0{ancho}d}" for o, i in zip(v, perm)}

    m_cod = mapa(originales, "COD-", 4)
    m_mar = mapa(df["marca"].astype(str).unique(), "MARCA_", letras=True)
    m_sub = mapa(df["subfamilia"].unique(), "SUBF_", 2)
    out = df.assign(codigo=df["codigo"].astype(str).map(m_cod),
                    marca=df["marca"].astype(str).map(m_mar),
                    subfamilia=df["subfamilia"].map(m_sub))[COLS_SALIDA]
    out = out.dropna(subset=["fecha"]).sort_values(["codigo", "fecha"])
    out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")

    # verificación de privacidad: ningún valor original debe sobrevivir en el texto de salida
    texto = set(out["codigo"]) | set(out["marca"]) | set(out["subfamilia"])
    fuga = (set(originales) | set(df["marca"].astype(str)) | set(df["subfamilia"])) & texto
    assert not fuga, f"Quedaron valores originales en la salida: {sorted(fuga)[:5]}"
    return out, {"codigo": m_cod, "marca": m_mar, "subfamilia": m_sub}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="ruta del df.csv real")
    ap.add_argument("--salida", default="data/sample_snapshots.csv")
    ap.add_argument("--n-codigos", type=int, default=None, help="muestra aleatoria de códigos (opcional)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--guardar-mapa", default=None, help="ruta (FUERA del repo) para guardar el mapa original->anónimo")
    a = ap.parse_args()

    crudo = pd.read_csv(a.entrada, dtype={"codigo": str}, low_memory=False)
    salida, mapas = anonimizar(crudo, a.n_codigos, a.semilla)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(a.salida, index=False)
    if a.guardar_mapa:
        pd.concat({k: pd.Series(v) for k, v in mapas.items()}).to_csv(a.guardar_mapa)
    print(f"{len(salida):,} filas | {salida['codigo'].nunique()} códigos | {salida['fecha'].nunique()} cortes -> {a.salida}")
