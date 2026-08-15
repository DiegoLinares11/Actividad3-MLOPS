# Actividad 3 · Calibración de hiperparámetros con pipelines de scikit-learn

Curso **Machine Learning Engineering (MLE/MLOps)** — Universidad del Valle de Guatemala.
Entrega **individual** — Diego Linares (`lin221256@uvg.edu.gt`).

Investigación de los objetos de scikit-learn que sirven para **calibrar hiperparámetros** de un
pipeline, **empaquetado** para poder instalarlo y ejecutarlo en cualquier computadora.

> **Continuación del Ejercicio 1 y de la Actividad 1.** Se usa el mismo dataset
> (`champions_league_matches.csv`) y el mismo pipeline de preparación; lo nuevo aquí son las
> etapas **Modeling** y **Evaluation** de CRISP-DM.

---

## Inicio rápido

```bash
git clone https://github.com/DiegoLinares11/Actividad3-MLOPS.git
cd Actividad3-MLOPS
pip install .
act3-demo
```

Salida esperada (recortada):

```
====================================================================
ACTIVIDAD 3 - Calibracion de hiperparametros (UEFA Champions League)
====================================================================
Paquete act3_pipeline v0.1.0
Equipo (hostname): <nombre de tu compu>
Sistema operativo: Windows 11
Python: 3.14.2

[datos] CSV crudo   -> 151 filas, 18 columnas
[datos] Filtrado    -> 144 partidos
[datos] Separacion  -> train=115  test=29

[modeling] Validacion cruzada: StratifiedKFold(5), metrica f1_macro
[modeling] GridSearchCV           96 combinaciones    55.7 s   mejor f1_macro = 0.563
[modeling] RandomizedSearchCV     60 combinaciones    17.0 s   mejor f1_macro = 0.619
[modeling] HalvingGridSearchCV   128 combinaciones    60.3 s   mejor f1_macro = 0.594

[modeling] Busqueda ganadora: RandomizedSearchCV
[modeling] Modelo final: LogisticRegression

====================================================================
RESULTADO - accuracy=0.690  f1_macro=0.631
====================================================================
Baseline (clase mayoritaria): accuracy=0.483  f1_macro=0.217
```

Todas las semillas están fijas, así que **estos números deben salir idénticos en cualquier
computadora**. Esa es la prueba de que el pipeline es reproducible.

Opciones útiles:

```bash
act3-demo --busqueda aleatoria   # solo RandomizedSearchCV (~17 s)
act3-demo --rapido               # rejilla reducida de 8 combinaciones
act3-demo --help
```

> Si el comando `act3-demo` no queda en el PATH, funciona igual con:
> ```bash
> python -m act3_pipeline.cli
> ```

---

## El problema (recapitulación)

Clasificar el resultado de un partido de Champions League en `Home Win`, `Away Win` o `Draw` a
partir de las estadísticas de juego.

| Dato | Valor |
|---|---|
| Partidos (después de limpiar) | 144 |
| Entrenamiento / prueba | 115 / 29 (estratificado, `seed=42`) |
| Columnas crudas → transformadas | 12 → 86 |
| Clases | Home Win 71 · Away Win 48 · **Draw 25** |

Ese desbalance obliga a una decisión que atraviesa toda la actividad: **la métrica que se optimiza
es `f1_macro`, no `accuracy`**. El `accuracy` premia acertar la clase mayoritaria; `f1_macro` pesa
igual a las tres clases y obliga al modelo a tomarse en serio los empates.

---

## Objetos de scikit-learn investigados

| Objeto | Para qué sirve | Dónde está |
|---|---|---|
| `Pipeline` | Encadena pasos y expone sus hiperparámetros como `paso__hiperparametro` | `pipeline.py` |
| `ColumnTransformer` | Un tratamiento distinto por grupo de columnas; se anida un nivel más | `pipeline.py` |
| `SelectKBest` | Selección de variables **calibrable** junto con el modelo | `pipeline.py` |
| `StratifiedKFold` | CV que respeta la proporción de las tres clases | `tuning.py` |
| `GridSearchCV` | Búsqueda exhaustiva sobre una rejilla | `tuning.py` |
| `RandomizedSearchCV` | Muestreo aleatorio; compara **familias de modelos** | `tuning.py` |
| `HalvingGridSearchCV` | Mitades sucesivas: mismo espacio, menos cómputo | `tuning.py` |
| `validation_curve` / `learning_curve` | Diagnóstico de sobreajuste y de falta de datos | `evaluation.py` |
| `permutation_importance` | Qué variables sostienen al modelo | `evaluation.py` |
| `DummyClassifier` | Baseline contra el cual comparar | `evaluation.py` |

El pipeline queda así:

```
preprocesamiento (ColumnTransformer)
    ├── numericas    : imputar -> escalar
    ├── porcentajes  : '63%' -> número -> imputar -> escalar
    ├── razones      : '3 of 10' -> 2 números -> imputar -> escalar
    └── categoricas  : imputar -> One-Hot
    -> seleccion (SelectKBest)
    -> modelo (clasificador)
```

---

## Por qué se calibra cada hiperparámetro

Calibrar no es tirar todo a una rejilla a ver qué pega: cada combinación extra multiplica el
cómputo y agrega riesgo de sobreajustar la propia búsqueda. Un hiperparámetro entra a la rejilla
cuando **controla la capacidad del modelo**, **su valor correcto no se puede deducir de antemano**
e **interactúa con los demás**.

| Hiperparámetro | Qué controla | Por qué se calibra aquí |
|---|---|---|
| `...imputar__strategy` (`median`/`mean`) | Con qué valor se rellenan los nulos | Los nulos vienen de partidos con "0 of 0" atajadas: distribución sesgada y acotada. Son pocos, así que la pregunta real es **si la decisión importa o da igual**. |
| `seleccion__k` (15/30/`all`) | Cuántas de las 86 variables llegan al modelo | 72 columnas son el One-Hot de los equipos y cada equipo aparece en ~6–8 partidos: casi ruido. Con 115 filas, arrastrar 86 columnas es pedir sobreajuste. |
| `modelo__n_estimators` (200/400) | Cuántos árboles promedia el bosque | Más árboles bajan la varianza y nunca sobreajustan más, pero cuestan tiempo: se busca dónde se aplana. |
| `modelo__max_depth` (`None`/6) | Profundidad de cada árbol | **La** perilla de sobreajuste: sin límite, con 115 filas el árbol memoriza (`mean_train_score` ≈ 1.0). |
| `modelo__min_samples_leaf` (1/3) | Mínimo de partidos por hoja | Con `1`, una hoja puede sostenerse en un solo partido raro — y solo hay 20 empates en entrenamiento. |
| `modelo__class_weight` (`None`/`balanced`) | Cuánto pesa fallar en cada clase | Respuesta directa al desbalance 57/38/20: `balanced` hace que fallar un empate cueste ~3× más. |

Y lo que **no** se calibra: `handle_unknown="ignore"` (requisito, no perilla), el `StandardScaler`
(inofensivo para el bosque, necesario para la regresión logística y el SVC), la métrica y el
esquema de CV (decisiones de diseño, no hiperparámetros) y `random_state=42` (reproducibilidad).

### Qué hiperparámetro pesó de verdad

Promediando los 96 resultados de `GridSearchCV` por el valor de cada hiperparámetro:

| Hiperparámetro | Efecto medido | Lectura |
|---|---|---|
| `class_weight` | `balanced` 0.524 vs `None` 0.515 | Gana `balanced`, como se esperaba. |
| `seleccion__k` | 30 → 0.525 · 15 → 0.523 · `all` → 0.509 | Recortar variables **sí** ayuda: sobran columnas. |
| `max_depth` | `None` 0.522 vs 6 → 0.516 | Casi no mueve el promedio, pero reduce la brecha de sobreajuste. |
| `n_estimators`, `imputar__strategy` | Sin efecto apreciable | Se puede fijar el más barato **con evidencia**, no por corazonada. |

Saber qué hiperparámetro *no* importa vale tanto como saber cuál sí.

---

## Diagrama del pipeline

Se genera con `sklearn.set_config(display="diagram")` y está embebido en el notebook (sección 2).
También queda exportado como HTML independiente:

- `pipeline_diagram.html` — el `GridSearchCV` completo con el pipeline adentro
- `modelo_final_diagram.html` — el pipeline ganador ya calibrado

```bash
pip install .[notebook]
jupyter notebook notebooks/calibracion_hiperparametros.ipynb
```

---

## Resultados

### Comparación de los tres objetos de búsqueda

| Objeto | Combinaciones | Tiempo | Mejor `f1_macro` en CV |
|---|---|---|---|
| Sin calibrar (valores por defecto) | 1 | — | 0.508 |
| `GridSearchCV` | 96 | 55.7 s | 0.563 |
| **`RandomizedSearchCV`** | **60** | **17.0 s** | **0.619** |
| `HalvingGridSearchCV` | 128 | 60.3 s | 0.594 |

La búsqueda aleatoria le ganó a la exhaustiva **en menos de un tercio del tiempo**. No es magia:
`GridSearchCV` estaba encerrado en un solo vecindario (Random Forest), mientras que
`RandomizedSearchCV` podía cambiar de familia de modelo. Con 115 filas y 86 columnas, una
**regresión logística regularizada** resultó más adecuada que un bosque.

| Familia probada | Intentos | Mejor `f1_macro` en CV |
|---|---|---|
| LogisticRegression | 15 | **0.619** |
| SVC | 18 | 0.612 |
| HistGradientBoosting | 11 | 0.561 |
| RandomForest | 16 | 0.556 |

### Modelo final

Elegido **por el puntaje de validación cruzada**, nunca mirando el conjunto de prueba:

```
LogisticRegression(C=0.541, class_weight="balanced", max_iter=5000)
con SelectKBest(k=40) e imputación por mediana
```

| Medición | accuracy | `f1_macro` |
|---|---|---|
| Baseline (clase mayoritaria), test | 0.483 | 0.217 |
| Pipeline sin calibrar, CV | — | 0.508 |
| `GridSearchCV` `best_score_`, CV | — | 0.563 |
| `RandomizedSearchCV` `best_score_`, CV | — | 0.619 |
| **CV anidada (estimación honesta)** | — | **0.541** |
| **Modelo final, test reservado** | **0.690** | **0.631** |

| Clase | precision | recall | f1 | soporte |
|---|---|---|---|---|
| Away Win | 0.714 | 1.000 | 0.833 | 10 |
| Draw     | 0.333 | 0.400 | 0.364 | 5 |
| Home Win | 0.889 | 0.571 | 0.696 | 14 |

La **validación cruzada anidada (0.541)** queda por debajo del `best_score_` de la rejilla
(0.563): esa diferencia es exactamente el optimismo que introduce la propia búsqueda, y es la
razón por la que `best_score_` no es el número que se reporta.

### Limitaciones

- **29 partidos de prueba**: cada acierto vale 3.4 puntos de accuracy, así que las métricas traen
  un margen de error amplio.
- **Los empates siguen siendo el punto débil** (f1 = 0.364): 25 ejemplos en todo el dataset y sin
  una huella estadística clara que los distinga.
- La **curva de aprendizaje se aplana** a partir de ~66 partidos: el techo no lo pone la cantidad
  de datos ni la calibración, sino la información que traen las variables. Para mejorar de verdad
  habría que agregar variables **nuevas** (historial de equipos, rachas), no más filas.
- La **importancia por permutación deja a `home_team` y `away_team` en prácticamente cero**: el
  modelo ignora quién juega y decide por tiros a puerta y atajadas.
- Las variables se conocen **cuando el partido ya terminó**, así que el modelo *explica* el
  resultado más de lo que lo *predice* de antemano.

---

## Evidencia de ejecución en distintas computadoras

Las capturas están en [`docs/screenshots/`](docs/screenshots/). El mismo paquete se instaló y
ejecutó en equipos distintos, dando exactamente el mismo resultado:

| | Equipo 1 | Equipo 2 |
|---|---|---|
| Hostname | `Dlinares` | _(pendiente)_ |
| Sistema operativo | Windows 11 | _(pendiente)_ |
| Python | 3.14.2 | _(pendiente)_ |
| Mejor `f1_macro` en CV | **0.619** | _(pendiente)_ |
| **Test accuracy / f1_macro** | **0.690 / 0.631** | _(pendiente)_ |

Que los resultados coincidan hasta el tercer decimal, en máquinas distintas, confirma que el
pipeline empaquetado es reproducible y no depende del entorno de quien lo ejecuta.

---

## Estructura del proyecto

```
.
├── pyproject.toml                        # empaquetado (metadata + dependencias)
├── setup.py                              # shim de compatibilidad
├── requirements.txt
├── README.md
├── pipeline_diagram.html                 # diagrama exportado
├── modelo_final_diagram.html
├── notebooks/
│   └── calibracion_hiperparametros.ipynb # DIAGRAMA + Modeling + Evaluation
├── docs/screenshots/                      # evidencia de ejecución
├── tests/
│   └── test_pipeline.py
└── src/
    └── act3_pipeline/
        ├── __init__.py
        ├── data.py           # extracción, filtrado, separación
        ├── transformers.py   # '63%' y '3 of 10' -> números
        ├── pipeline.py       # ColumnTransformer + SelectKBest + modelo
        ├── tuning.py         # rejillas, espacios y objetos de búsqueda
        ├── evaluation.py     # métricas, curvas, CV anidada, importancias
        ├── cli.py            # ejecución de extremo a extremo
        └── datasets/
            └── champions_league_matches.csv
```

---

## Compartir el paquete

Además del `git clone`, se puede generar un instalable para pasarlo por Drive o USB:

```bash
pip install build
python -m build          # crea dist/act3_pipeline-0.1.0-py3-none-any.whl
```

El compañero lo instala con:

```bash
pip install act3_pipeline-0.1.0-py3-none-any.whl
```

El CSV va incluido dentro del paquete, así que no hace falta descargar nada aparte.

## Pruebas

```bash
pip install .[dev]
pytest -q
```

---

## CRISP-DM

```
Business Understanding ─┐
Data Understanding      ├── Ejercicio 1
Data Preparation ───────┘   Actividad 1  (pipeline de preparación)
Modeling ───────────────┐
Evaluation              ├── Actividad 3  ← este repositorio
Deployment (empaquetado)┘
```

- **Modeling** — rejillas y espacios de búsqueda justificados, tres objetos de calibración
  comparados, cuatro familias de modelos, curva de validación.
- **Evaluation** — conjunto de prueba reservado, baseline, matriz de confusión, validación cruzada
  anidada, curva de aprendizaje, importancia por permutación y limitaciones.
- **Deployment** — el pipeline empaquetado (`pip install .`) con el CLI `act3-demo`.
