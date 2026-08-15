"""Pruebas mínimas del paquete.

Ejecutar con::

    pip install .[dev]
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from act3_pipeline import (
    FEATURES,
    REJILLA_RAPIDA,
    TARGET,
    baseline,
    busqueda_grid,
    cargar_datos,
    construir_pipeline,
    evaluar_en_prueba,
    extraer_datos,
    filtrar_datos,
    nombres_de_hiperparametros,
)


def test_dataset_empaquetado():
    """El CSV viaja dentro del paquete y se limpia a 144 partidos."""
    crudo = extraer_datos()
    assert crudo.shape == (151, 18)
    limpio = filtrar_datos(crudo)
    assert len(limpio) == 144
    assert limpio[TARGET].isna().sum() == 0
    # Las columnas con fuga de informacion ya no estan.
    assert "score" not in limpio.columns
    assert "winner" not in limpio.columns


def test_separacion_estratificada():
    X_train, X_test, y_train, y_test = cargar_datos()
    assert len(X_train) == 115 and len(X_test) == 29
    assert list(X_train.columns) == FEATURES
    # Las tres clases aparecen en ambos conjuntos.
    assert set(y_train.unique()) == set(y_test.unique())


def test_pipeline_tiene_los_pasos_esperados():
    pipe = construir_pipeline()
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["preprocesamiento", "seleccion", "modelo"]


def test_hiperparametros_calibrables_existen():
    """Cada llave de la rejilla debe existir en el pipeline."""
    llaves = set(nombres_de_hiperparametros())
    for clave in REJILLA_RAPIDA:
        assert clave in llaves


def test_preprocesamiento_convierte_texto_a_numero():
    X_train, _, y_train, _ = cargar_datos()
    pipe = construir_pipeline()
    Xt = pipe.named_steps["preprocesamiento"].fit_transform(X_train)
    assert Xt.dtype.kind == "f"          # todo salio numerico
    assert not np.isnan(Xt).any()        # la imputacion no dejo nulos
    assert Xt.shape[0] == len(X_train)


@pytest.mark.parametrize("rejilla", [REJILLA_RAPIDA])
def test_busqueda_supera_al_baseline(rejilla):
    """La calibracion corre de punta a punta y le gana al baseline."""
    X_train, X_test, y_train, y_test = cargar_datos()
    search = busqueda_grid(rejilla, n_jobs=1)
    assert isinstance(search, GridSearchCV)
    search.fit(X_train, y_train)

    metricas = evaluar_en_prueba(search.best_estimator_, X_test, y_test)
    dummy = baseline(X_train, y_train, X_test, y_test)
    assert metricas["f1_macro"] > dummy["f1_macro"]
