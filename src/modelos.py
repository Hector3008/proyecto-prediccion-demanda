"""Modelos, validación temporal y búsqueda de hiperparámetros."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .features import CAT_COLS, COLS_BL, FEATURES, H, NUM_COLS
from .metricas import wape, wape_scorer

SEED = 42


# --------------------------------------------------------------------------- validación temporal
class SplitPorOrigen:
    """Validación cruzada temporal con purga.

    Cada fold valida en un origen v y entrena con orígenes T <= v - h. Así el objetivo de
    ninguna fila de entrenamiento se solapa con las semanas que se predicen en validación.
    Compatible con GridSearchCV / RandomizedSearchCV (cv=...).
    """

    def __init__(self, origenes, h: int = H):
        self.origenes = np.asarray(origenes)
        self.h = h

    def _folds(self):
        ts = sorted(set(self.origenes))
        for v in ts:
            tr = np.flatnonzero(self.origenes <= v - self.h)
            if len(tr):
                yield tr, np.flatnonzero(self.origenes == v)

    def split(self, X=None, y=None, groups=None):
        yield from self._folds()

    def get_n_splits(self, X=None, y=None, groups=None):
        return sum(1 for _ in self._folds())


# --------------------------------------------------------------------------- baselines (estimadores)
class BaselinePorSegmento(BaseEstimator, RegressorMixin):
    """Elige, por segmento ADI/CV², el modelo clásico (columnas bl_*) con menor WAPE en entrenamiento.

    'cero' solo puede ganar en 'sin_ventas': no se acepta "no pedir" donde sí hay demanda.
    """

    def fit(self, X, y):
        y = np.asarray(y, float)
        cand = [c for c in COLS_BL if c != "bl_cero"]
        glob = min(cand, key=lambda c: wape(y, X[c].to_numpy()))
        self.global_ = glob
        self.mapa_ = {}
        for seg, idx in X.groupby("segmento").indices.items():
            if seg == "sin_ventas":
                self.mapa_[seg] = "bl_cero"
                continue
            ys = y[idx]
            if ys.sum() <= 0:
                self.mapa_[seg] = glob
                continue
            self.mapa_[seg] = min(cand, key=lambda c: wape(ys, X[c].to_numpy()[idx]))
        return self

    def predict(self, X):
        out = np.zeros(len(X))
        for seg, idx in X.reset_index(drop=True).groupby("segmento").indices.items():
            col = self.mapa_.get(seg, self.global_)
            out[idx] = X[col].to_numpy()[idx]
        return out


class BaselineColumna(BaseEstimator, RegressorMixin):
    """Usa directamente una columna bl_* como predicción (sin entrenar)."""

    def __init__(self, columna="bl_croston"):
        self.columna = columna

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.columna].to_numpy(float)


class HibridoRotacion(BaseEstimator, RegressorMixin):
    """Modelo de ML solo donde hay rotación suficiente; baseline por segmento en el resto.

    Hipótesis de negocio: en los códigos de venta espaciada no hay señal que un modelo de ML
    pueda aprender con ~10 semanas; en los de alta rotación sí.
    """

    def __init__(self, ml=None, baseline=None, rotaciones_ml=("alta", "media-alta")):
        self.ml = ml
        self.baseline = baseline
        self.rotaciones_ml = rotaciones_ml

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.baseline_ = clone(self.baseline).fit(X, y)
        m = X["rotacion"].isin(self.rotaciones_ml).to_numpy()
        self.ml_ = clone(self.ml).fit(X[m], y[m]) if m.sum() > 10 else None
        return self

    def predict(self, X):
        p = np.asarray(self.baseline_.predict(X), float)
        if self.ml_ is not None:
            m = X["rotacion"].isin(self.rotaciones_ml).to_numpy()
            if m.any():
                p[m] = self.ml_.predict(X[m])
        return p


# --------------------------------------------------------------------------- modelos de ML
def preprocesador() -> ColumnTransformer:
    """Imputa y escala lo numérico; one-hot para lo categórico.

    - Imputación (mediana): stock puede faltar en códigos recién creados.
    - Escalado: necesario para Ridge (penalización L2); inocuo para árboles.
    - One-hot: marca, subfamilia, segmento y rotación no tienen orden natural.
      handle_unknown='ignore' evita errores si aparece una categoría nueva en test.
    """
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)],
                             verbose_feature_names_out=False)


def pipe(modelo) -> Pipeline:
    return Pipeline([("prep", preprocesador()), ("modelo", modelo)])


def catalogo_ml(seed: int = SEED) -> dict:
    """Modelos de ML con su espacio de búsqueda: nombre -> (pipeline, params, tipo de búsqueda)."""
    return {
        "Ridge": (pipe(Ridge()),
                  {"modelo__alpha": [0.01, 0.1, 1, 10, 100, 1000]}, "grid"),
        "RandomForest": (pipe(RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
                         {"modelo__max_depth": [3, 6, None],
                          "modelo__min_samples_leaf": [1, 5, 20],
                          "modelo__max_features": [0.5, 1.0]}, "grid"),
        "XGBoost": (pipe(XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist")),
                    {"modelo__n_estimators": randint(100, 500),
                     "modelo__max_depth": randint(2, 7),
                     "modelo__learning_rate": loguniform(0.01, 0.2),
                     "modelo__subsample": uniform(0.6, 0.4),
                     "modelo__colsample_bytree": uniform(0.5, 0.5),
                     "modelo__min_child_weight": [1, 3, 5, 10],
                     "modelo__reg_lambda": loguniform(0.1, 20),
                     # WAPE es un error absoluto: probar también el objetivo MAE
                     "modelo__objective": ["reg:squarederror", "reg:absoluteerror"]}, "random"),
    }


def buscar_hiperparametros(X, y, cv, seed: int = SEED, n_iter: int = 40, verbose: bool = True) -> dict:
    """Ejecuta GridSearchCV / RandomizedSearchCV por modelo con el scorer -WAPE."""
    resultados = {}
    for nombre, (estimador, params, tipo) in catalogo_ml(seed).items():
        if tipo == "grid":
            bus = GridSearchCV(estimador, params, scoring=wape_scorer, cv=cv, n_jobs=1, refit=True)
        else:
            bus = RandomizedSearchCV(estimador, params, n_iter=n_iter, scoring=wape_scorer, cv=cv,
                                     random_state=seed, n_jobs=1, refit=True)
        bus.fit(X, y)
        resultados[nombre] = bus
        if verbose:
            print(f"{nombre:13s} [{tipo:6s}] mejor WAPE CV = {-bus.best_score_:.3f} | {bus.best_params_}")
    return resultados


def predicciones_fuera_de_muestra(estimador, X, y, cv) -> pd.DataFrame:
    """Predicción de cada fold de validación (modelo reentrenado con su train del fold)."""
    piezas = []
    for tr, va in cv.split(X):
        m = clone(estimador).fit(X.iloc[tr], y.iloc[tr])
        piezas.append(pd.DataFrame({"idx": X.index[va], "real": y.iloc[va].to_numpy(),
                                    "pred": np.clip(m.predict(X.iloc[va]), 0, None)}))
    return pd.concat(piezas, ignore_index=True)


__all__ = ["SEED", "SplitPorOrigen", "BaselinePorSegmento", "BaselineColumna", "HibridoRotacion",
           "preprocesador", "pipe", "catalogo_ml", "buscar_hiperparametros",
           "predicciones_fuera_de_muestra", "LinearRegression", "FEATURES"]
