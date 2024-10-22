# Procesamiento MAGDAS

Este repositorio contiene las herramientas y scripts desarrollados para el procesamiento de datos del campo magnético utilizando el módulo `Procesamiento_MAGDAS`. El enfoque principal es la lectura, procesamiento y graficación de datos del campo magnético obtenidos de la enstación MAGDAS, ubicada en Jerusalem

## Descripción del Proyecto

El módulo **Procesamiento_MAGDAS** facilita la lectura de datos en formato .JRS, los convierte en archivos .CSV para luego realizar el procesamiento de los componentes magnéticos y el campo magnético total, y la visualización de la serie temporal resultante. El procesamiento incluye funcionalidades como:

- Suavizado de los datos con diferentes métodos.
- Eliminación de valores atípicos.
- Aplicación de transformaciones para obtener los componentes magnéticos (Hcomp, Dcomp, Zcomp) y el campo magnético total (B).
- Graficación de las series temporales tanto de los datos originales como procesados.

El propósito de este proyecto es proporcionar una herramienta flexible y eficiente para el análisis de datos magnéticos obtenidos de estaciones MAGDAS.
NOTA: Los archivos .JRS deben estar colocados en carpetas cuyos nombres correspondan al año en el que fueron recolectados dichos datos.
## Estructura del Proyecto

- **main.py**: Script principal que utiliza el módulo `Procesamiento_MAGDAS` para leer archivos de datos, procesarlos y generar gráficos de las series temporales.
- **Procesamiento_MAGDAS.py**: Módulo que contiene las funciones de procesamiento de datos, incluyendo lectura, suavizado, eliminación de outliers y graficación.
- **data_series_temporales**: Carpeta donde se almacenan los archivos CSV con los datos a procesar.
- **Graficos**: Carpeta donde se almacenan los gráficos generados.

## Funcionalidades

1. **Lectura de Datos**: Se leen archivos en formato `.JRS` con los datos crudos del campo magnético y se transforman a datos `.csv` para su posterior procesamiento.
2. **Procesamiento**:
   - Remuestreo de los datos.
   - Suavizado con diferentes métodos.
   - Eliminación de outliers basada en desviación estándar.
   - Cálculo de los componentes magnéticos (H, D, Z) y del campo magnético total (B).
3. **Graficación**:
   - Gráficos que comparan las series originales y las procesadas.
   - Visualización de la tendencia en los datos procesados.

