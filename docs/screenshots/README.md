# Evidencia de ejecución en distintas computadoras

Esta carpeta guarda las capturas que demuestran que el pipeline empaquetado corre en más de una
computadora dando el **mismo resultado**.

## Cómo generar una captura

En cada computadora:

```bash
git clone https://github.com/DiegoLinares11/Actividad3-MLOPS.git
cd Actividad3-MLOPS
pip install .
act3-demo
```

Tomar la captura de la terminal asegurándose de que se vea:

- el **hostname**, el **sistema operativo** y la **versión de Python** (los imprime el propio
  comando en el encabezado),
- la tabla de las tres búsquedas con el `f1_macro` de cada una,
- la línea `RESULTADO - accuracy=... f1_macro=...`.

Guardar el archivo aquí como `equipo1.png`, `equipo2.png`, etc., y actualizar la tabla de
evidencia del `README.md` de la raíz.

> La celda final del notebook (`5.2 Evidencia de ejecución en distintas computadoras`) imprime la
> misma información, por si se prefiere capturar desde Jupyter.

## Qué debe coincidir

Todas las semillas están fijas (`random_state=42` en la partición, en la validación cruzada y en
los modelos), así que estos valores tienen que salir **idénticos** en cualquier máquina:

| | Valor esperado |
|---|---|
| Búsqueda ganadora | `RandomizedSearchCV` |
| Modelo final | `LogisticRegression` |
| Mejor `f1_macro` en CV | 0.619 |
| Test accuracy | 0.690 |
| Test `f1_macro` | 0.631 |

Lo único que debe cambiar entre computadoras son el hostname, el sistema operativo, la versión de
Python y los **tiempos** de cada búsqueda (dependen del número de núcleos).

## Capturas

| Archivo | Equipo | Sistema operativo | Python |
|---|---|---|---|
| _(pendiente)_ | Dlinares | Windows 11 | 3.14.2 |
| _(pendiente)_ | | | |
