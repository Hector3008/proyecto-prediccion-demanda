"""Pruebas rápidas: métricas, ausencia de fuga temporal y ejecución del flujo de datos."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import agregar_columnas_derivadas, cargar_snapshots, metadatos_codigo
from src.demanda import clasificar_serie, matriz_acumulada_semanal, matriz_semanal
from src.features import H, construir_dataset, dividir_por_origen
from src.metricas import faltante, exceso, sesgo, wape
from src.modelos import SplitPorOrigen

DATOS = Path(__file__).resolve().parents[1] / "data" / "sample_snapshots.csv"


def test_wape_y_sesgo():
    y, p = [10, 0, 5], [8, 2, 5]
    assert wape(y, p) == pytest.approx(4 / 15)
    assert sesgo(y, p) == pytest.approx(0.0)
    assert faltante(y, p) == pytest.approx(2 / 15) and exceso(y, p) == pytest.approx(2 / 15)
    assert np.isnan(wape([0, 0], [1, 1]))          # sin demanda real no está definido


def test_clasificacion():
    assert clasificar_serie([0, 0, 0, 0])[2] == "sin_ventas"
    assert clasificar_serie([5, 5, 6, 5, 4])[2] == "smooth"
    assert clasificar_serie([0, 0, 9, 0, 0, 1, 0, 0])[2] == "lumpy"


@pytest.fixture(scope="module")
def dataset():
    df = agregar_columnas_derivadas(cargar_snapshots(str(DATOS)))
    return construir_dataset(matriz_semanal(df), matriz_acumulada_semanal(df, "total"), metadatos_codigo(df))


def test_particiones_sin_solape(dataset):
    p = dividir_por_origen(dataset, H)
    assert p["train_base"]["T"].max() + H <= p["validacion"]["T"].min()
    assert p["pool"]["T"].max() + H <= p["test"]["T"].min()


def test_cv_temporal_sin_fuga(dataset):
    t = dividir_por_origen(dataset, H)["pool"]["T"].to_numpy()
    cv = SplitPorOrigen(t, H)
    assert cv.get_n_splits() >= 1
    for tr, va in cv.split():
        assert t[tr].max() + H <= t[va].min()


def test_no_hay_nulos_en_objetivo(dataset):
    assert dataset["y"].notna().all() and (dataset["y"] >= 0).all()
