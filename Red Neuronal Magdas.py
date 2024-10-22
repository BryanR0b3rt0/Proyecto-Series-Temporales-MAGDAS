import os
import pandas as pd
import matplotlib.pyplot as plt

import dask.dataframe as dd

# Cargar los archivos CSV usando dask
ruta_archivos = r'C:\Users\Lenovo\Desktop\OAQ\Series Temporales Magdas\data_series_temporales'

# Leer los archivos con dask
df_dask = dd.read_csv(os.path.join(ruta_archivos, 'serie_temporal_*_processed.csv'))

# Aplicar la interpolación a cada partición
df_dask['B_total_smooth'] = df_dask.map_partitions(lambda df: df['B_total_smooth'].interpolate())

# Computar el resultado final en un DataFrame de pandas
df_total_interpolated = df_dask.compute()

# Mostrar una muestra de los datos interpolados
print(df_total_interpolated.head())

# --- Graficar los datos antes y después de la interpolación ---

# Cargar solo una pequeña muestra de los datos originales (con NaN) para el gráfico
df_dask_original = dd.read_csv(os.path.join(ruta_archivos, 'serie_temporal_2012_processed.csv')).compute()

# Visualizar antes y después de la interpolación
plt.figure(figsize=(12, 6))

# Gráfico con valores originales (con NaN)
plt.plot(df_dask_original['Time'][:100000], df_dask_original['B_total_smooth'][:100000], label='Original con NaN', alpha=0.5)

# Gráfico con valores interpolados
plt.plot(df_total_interpolated['Time'][:100000], df_total_interpolated['B_total_smooth'][:100000], label='Interpolado', alpha=0.8)

# Personalización del gráfico
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Campo Magnético B (suavizado)')
plt.title('Comparación antes y después de la interpolación')
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
