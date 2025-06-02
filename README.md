# Procesamiento y Análisis de Series Temporales del Sistema MAGDAS

## Introducción

El presente documento describe el uso y los fundamentos teóricos del programa desarrollado para el procesamiento y análisis de series temporales de datos del sistema MAGDAS (Magnetic Data Acquisition System). El objetivo es transformar los datos crudos obtenidos en archivos `.JRS` en series temporales limpias, procesadas y analizadas para su interpretación científica, enfocándose en el estudio del campo magnético terrestre.

## Descripción General del Programa

El programa está dividido en dos componentes principales:

- Un módulo con funciones específicas para la lectura, procesamiento, análisis y visualización de datos magnéticos.
- Un archivo principal (`main.py`) que orquesta el flujo completo de procesamiento para un año determinado.

Este esquema modular permite flexibilidad para usar funciones individuales o ejecutar un flujo completo automatizado.

## Proceso de Lectura de Datos Crudos

Los datos originales se almacenan en archivos binarios con extensión `.JRS`. Estos archivos contienen medidas horarias de las componentes magnéticas H (horizontal), D (declinación), y Z (vertical).

La función `leer_datosJRS` se encarga de:

- Leer estos archivos desde una carpeta específica del año.
- Decodificar los datos binarios.
- Aplicar conversiones para obtener valores físicos en nanoteslas (nT).
- Calcular el campo magnético total
- Concatenar los datos diarios en vectores completos.

Este paso es fundamental para obtener una base de datos accesible en formato CSV para análisis posteriores.

## Limpieza y Suavizado de la Serie Temporal

Para eliminar valores atípicos y ruidos, se aplica un filtrado basado en la eliminación de *outliers*, definidos como puntos que exceden un cierto número de desviaciones estándar respecto a la media.

Posteriormente, se realiza un suavizado móvil con ventana diaria (o configurable), usando media o mediana, para reducir fluctuaciones rápidas no representativas.

Este proceso se realiza con la función `procesar_csv`, que:

- Carga el archivo CSV original.
- Elimina outliers.
- Aplica suavizado móvil.
- Guarda un nuevo archivo con la serie procesada.

## Remuestreo y Separación de Tendencia

Los datos procesados se remuestrean a resoluciones temporales más bajas (por ejemplo, cada 8 horas) para facilitar análisis a largo plazo y reducir el volumen de datos.

Luego, con el filtro de Hodrick-Prescott, se descompone la serie en dos componentes:

- **Tendencia:** evolución lenta y global del campo magnético.
- **Ciclo:** variaciones rápidas y oscilatorias alrededor de la tendencia.

Este filtro es ampliamente usado en econometría y análisis de series temporales para separar señales lentas de fluctuaciones rápidas.

## Análisis Espectral y de Dependencias Temporales

Se realiza un análisis mediante Transformada Rápida de Fourier (FFT) para identificar y filtrar frecuencias altas, reconstruyendo la señal para resaltar componentes relevantes.

También se calcula la autocorrelación para evaluar la dependencia temporal y periodicidades, lo cual es útil para detectar patrones repetitivos o ciclos.

## Visualización de Resultados

El programa incluye funciones para graficar:

- Serie original.
- Serie suavizada.
- Tendencia extraída.
- Comparación entre series originales y procesadas.
- Gráficos espectrales y de autocorrelación.

La visualización es clave para interpretar correctamente la evolución del campo magnético y validar los procesos de limpieza y análisis.

## Flujo de Trabajo Recomendado

1. **Extracción:** Usar `leer_datosJRS` y `guardar_datos_csv` para convertir datos binarios en CSV.
2. **Procesamiento:** Limpiar y suavizar con `procesar_csv`.
3. **Remuestreo:** Aplicar `Remuestreo_FiltroHP` para obtener tendencia y ciclo.
4. **Visualización:** Graficar resultados con las funciones correspondientes.
5. **Análisis avanzado:** FFT con `transformada_fourier_filtrado` y autocorrelación con `graficar_autocorrelacion`.
6. **Comparación anual:** Visualizar años múltiples con `leer_y_graficar_serie_anual`.

