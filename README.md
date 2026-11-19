# Predicción de demanda intermitente a 2 semanas por código

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hector3008/proyecto-prediccion-demanda/blob/main/notebooks/proyecto_final.ipynb)


Proyecto final de Machine Learning (Coderhouse). Predice cuántas unidades se venderán de cada código de repuesto en las
**próximas 2 semanas**, a partir de cortes (snapshots) diarios de inventario y ventas de una importadora con dos almacenes (LV y SJL).
Compara modelos clásicos de demanda intermitente (Croston, SBA, TSB) contra Ridge, Random Forest y XGBoost, con validación
temporal, optimización de hiperparámetros y explicabilidad con SHAP.

> **Sobre los datos incluidos.** `data/sample_snapshots.csv` es una **muestra anonimizada de datos reales**: 600 códigos, 17 semanas
> (14-jul-2026 a 16-nov-2026). Códigos, marcas y subfamilias están enmascarados y se eliminaron todos los textos libres; el archivo
> original es privado y no se publica. Los números de este README y del notebook corresponden a esa muestra. También se incluye un
> generador de datos sintéticos con la misma estructura (ver [Datos](#5-datos)).

## 1. El problema

| | |
|---|---|
| Tipo | **Regresión** |
| Objetivo `y` | unidades netas vendidas (`ventas − devolución`) en las 2 semanas siguientes al origen T, por código |
| Métrica principal | **WAPE** = Σ\|pred − real\| / Σ real (sirve con ceros y pondera por volumen) |
| Métricas de apoyo | sesgo, *faltante* (demanda no cubierta → quiebres), *exceso* (sobrestock), cobertura |
| Criterio de éxito | un modelo de ML solo es útil si supera al mejor modelo clásico en validación temporal |
| Relevancia de negocio | decidir cuánto comprar con un lead time de ~45 días; subestimar genera ventas perdidas, sobrestimar inmoviliza capital |

La demanda es **muy intermitente** (en la muestra, 85 % de las semanas código-semana no tiene ventas y 79 % de los objetivos a 2 semanas es 0) y el
historial es corto (17 semanas), por lo que no se asume que el ML gane: se verifica.

## 2. Qué contiene el notebook

`notebooks/proyecto_final.ipynb` (ya ejecutado, con salidas visibles) sigue la rúbrica en orden:

1. **Formulación del problema**: tipo, objetivo, métrica y criterio de éxito.
2. **EDA** con pandas y matplotlib: estadísticos, nulos, retrocesos en contadores, calendario de cortes, distribución de la demanda,
   segmentación ADI/CV², demanda por marca, curva de Pareto y outliers (regla de Tukey). Cada gráfico lleva un *insight* calculado.
3. **Ingeniería de atributos** (justificada): rezagos, medias móviles, intermitencia (ADI, CV², % semanas con venta, semanas desde la
   última venta), tendencia, stock y cobertura, categorías; imputación, escalado y one-hot dentro de un `Pipeline`. Incluye una
   **prueba anti-fuga** automática.
4. **Entrenamiento** con split temporal train / validación / test y purga (sin mezclar futuro y pasado).
5. **Optimización de hiperparámetros**: `GridSearchCV` (Ridge, Random Forest) y `RandomizedSearchCV` (XGBoost, 40 combinaciones) con
   validación cruzada temporal propia (`SplitPorOrigen`) y scorer −WAPE.
6. **Selección del modelo**: tabla de validación cruzada con WAPE, sesgo, faltante, exceso y cobertura; regla de parsimonia
   (entre modelos empatados dentro de 0,02 de WAPE se elige el más simple); modelo híbrido (ML en alta rotación + baseline);
   confirmación en test con intervalo de confianza bootstrap; evaluación walk-forward.
7. **SHAP** sobre el mejor modelo de árboles: importancia global, efecto por variable, dependencia y caso individual (waterfall).
8. **Conclusiones y limitaciones**, calculadas con los resultados de la ejecución.

### Resultados con la muestra anonimizada (semilla 42)

| Modelo | WAPE CV temporal | WAPE test |
|---|---|---|
| **XGBoost optimizado (elegido)** | **0,748** | **0,702** |
| Híbrido (XGBoost en alta rotación + baseline) | 0,781 | 0,728 |
| Baseline por segmento (Croston/SBA/TSB) | 0,870 | 0,730 |
| TSB | 0,851 | 0,745 |
| Ridge optimizado | 0,871 | 0,795 |
| Random Forest optimizado | 0,874 | 0,810 |
| Croston / SBA | 1,033 / 0,975 | 0,921 / 0,899 |

* **XGBoost (con `reg:absoluteerror`) es el mejor en validación cruzada** (9 folds temporales) y supera al baseline por segmento en
  los 11 orígenes del walk-forward (WAPE medio 0,745 vs 0,866). Mejora sobre todo en rotación media-alta y alta.
* **En test (un solo origen) la ventaja es pequeña**: 0,702 vs 0,730, y el intervalo bootstrap de la diferencia (−0,060 a +0,159)
  incluye el cero. No es concluyente por sí sola; la evidencia fuerte es la de la validación temporal repetida.
* **Advertencia de negocio: el modelo subestima.** Sesgo en CV de −39 % (faltante 57 %, exceso 18 %) frente a +3 % del baseline
  por segmento. Con 79 % de objetivos en 0, una pérdida de error absoluto tiende a predecir valores cercanos a la mediana: baja el
  WAPE a costa de más quiebres. Para decidir compras habría que calibrarlo o usar un escenario prudente; el WAPE solo no basta.
* **SHAP**: el modelo se apoya sobre todo en el nivel de ventas pasado (`media`, `media_ult3`, `frac_semanas_con_venta`); las 5
  primeras variables concentran 84 % del efecto.

## 3. Estructura del repositorio

```
proyecto-prediccion-demanda/
├── README.md
├── requirements.txt
├── data/
│   └── sample_snapshots.csv        # muestra anonimizada de datos reales (600 códigos)
├── notebooks/
│   └── proyecto_final.ipynb        # entregable principal, ejecutado
├── src/
│   ├── data.py                     # carga, tipos y columnas derivadas (primer registro, consumo)
│   ├── demanda.py                  # matriz semanal, segmentación ADI/CV², Croston/SBA/TSB/SES/MA
│   ├── features.py                 # dataset supervisado sin fuga + particiones temporales
│   ├── metricas.py                 # WAPE, sesgo, faltante, exceso, cobertura
│   └── modelos.py                  # CV temporal, baselines, híbrido, búsqueda de hiperparámetros
├── scripts/
│   ├── generar_datos_sinteticos.py # genera data/sample_snapshots.csv sintético
│   └── anonimizar_df.py            # convierte el df.csv real en una muestra anonimizada
└── tests/test_basico.py            # pruebas de métricas, particiones y CV sin fuga
```

## 4. Cómo ejecutarlo

**Opción A: Google Colab (sin instalar nada).** Abrir el notebook con el botón de arriba y *Entorno de ejecución → Ejecutar todo*.
La primera celda clona el repositorio si hace falta e instala lo que falte. *Colab requiere una cuenta de Google para ejecutar; sin cuenta, usar la opción B (local).*

**Opción B: local.**

```bash
git clone https://github.com/Hector3008/proyecto-prediccion-demanda.git
cd proyecto-prediccion-demanda
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/proyecto_final.ipynb --output proyecto_final.ipynb   # o abrirlo con Jupyter
pytest -q                                                                                                # pruebas
```

Tarda unos minutos. Es determinista (semilla 42). Para usar otro archivo de datos: `DATA_PATH=ruta/al.csv` antes de ejecutar.

## 5. Datos

`data/sample_snapshots.csv` tiene una fila por código y fecha de corte:

| Columna | Descripción |
|---|---|
| `codigo`, `marca`, `subfamilia` | identificadores enmascarados (`COD-0001`, `MARCA_A`, `SUBF_01`) |
| `fecha` | fecha del corte (AAAA-MM-DD). Cortes espaciados al inicio y diarios al final |
| `stock_lv`, `stock_sjl`, `total` | stock por almacén y total |
| `ventas`, `devolucion`, `importacion`, `ajuste_lv`, `ajuste_sjl` | **contadores acumulados**; importa su variación entre cortes |

**Datos sintéticos (alternativa):** `python scripts/generar_datos_sinteticos.py` genera 450 códigos con la misma estructura
(mezcla de rotación alta, intermitente y casi sin ventas). Sobrescribe `data/sample_snapshots.csv`.

**Muestra anonimizada de los datos reales** (la que trae el repositorio; se generó una vez, en el entorno donde está el `df.csv` privado, y nunca se sube el original):

```bash
python scripts/anonimizar_df.py --entrada /ruta/privada/df.csv --salida data/sample_snapshots.csv --n-codigos 600
```

El script consolida los VARIADO con la suma de sus VARIABLE, excluye PLAZA, enmascara códigos, marcas y subfamilias con una
permutación aleatoria, elimina todos los textos libres (detalle, nombre comercial, equivalentes, precios, proveedor, ubicación) y
verifica que ningún valor original quede en la salida. El mapa original → anónimo no se guarda salvo que se pida con
`--guardar-mapa` (y entonces fuera del repositorio). Si se reemplaza el archivo, volver a ejecutar el notebook: los
textos de conclusiones se recalculan.

## 6. Decisiones de diseño

* **Split temporal con purga.** Un split aleatorio filtraría el futuro. El objetivo de un origen usa las semanas T y T+1; por eso se
  descarta el origen vecino y se entrena solo con orígenes cuyo objetivo termina antes de lo que se valida.
* **CV temporal propia** (`SplitPorOrigen`) compatible con `GridSearchCV` / `RandomizedSearchCV`. Con 17 semanas hay 9 folds; con menos historia serían pocos. Es una
  limitación declarada, no un detalle oculto.
* **WAPE** como métrica y **objetivo `reg:absoluteerror`** como una de las opciones de XGBoost, porque la métrica es un error absoluto.
* **Sin transformar `y`**: un `log1p` sesgaría las predicciones a la baja y aumentaría los quiebres.
* **Parsimonia**: con diferencias de WAPE menores a la incertidumbre del experimento, se prefiere el modelo más simple y auditable.

## 7. Limitaciones

1. Historial corto (17 semanas): pocos orígenes de evaluación (un único origen de test); las diferencias pequeñas de WAPE no son concluyentes.
2. El error se mide a 2 semanas: no garantiza el error a 45 días (lead time) ni a horizontes largos.
3. No se descuentan pedidos en tránsito; sin promociones, estacionalidad anual ni quiebres de proveedor.
4. La demanda está censurada por quiebres (un código agotado vende 0 aunque haya demanda).
5. Lead time fijo de 45 días en el análisis de negocio; no se optimizó.
6. La muestra son 600 códigos elegidos al azar, sin VARIABLE ni PLAZA: las cifras describen esa muestra, no todo el inventario.
7. El modelo elegido subestima la demanda (sesgo −39 % en CV): requiere calibración antes de usarse para pedir.

## 8. Origen del proyecto

Nace de un sistema de siete notebooks en Google Colab (carga desde Google Sheets, wrangling, EDA con reportes PDF/PPTX, predicción de
consumo con modelos propios y unión de reportes). Este repositorio condensa y reproduce lo esencial para su evaluación: sin secretos,
sin rutas de Drive y con scikit-learn, XGBoost y SHAP. Las funciones del pipeline original (matriz semanal, ADI/CV², Croston/SBA/TSB,
backtest walk-forward) se conservan en `src/`.