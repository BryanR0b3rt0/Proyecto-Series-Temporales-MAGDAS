import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

# Especifica la carpeta del año y la ruta de los archivos .JRS
year_folder = '2014'
save_path = './data_series_temporales'

csv_file = os.path.join(save_path, f'serie_temporal_{year_folder}.csv')
csv_depured = csv_file.replace('.csv', '_processed.csv')

# Cargar el archivo CSV depurado
df = pd.read_csv(csv_depured)

# Eliminar los valores NaN del DataFrame
df_cleaned = df.dropna()

# Asegurarse de que la columna 'Time' sea de tipo datetime
df_cleaned['Time'] = pd.to_datetime(df_cleaned['Time'])

# Configurar la columna 'Time' como índice del DataFrame
df_cleaned.set_index('Time', inplace=True)

# Remuestrear la serie a intervalos de 24 horas y aplicar la media
df_remuestradeo = df_cleaned.resample('86400S').mean()

# Rellenar los NaN en el DataFrame remuestreado (forward fill o interpolación)
df_remuestradeo.fillna(method='ffill', inplace=True)  # o df_remuestradeo.interpolate(inplace=True)

# Verificar el resultado
print(df_remuestradeo.head(10))

# Guardar el DataFrame remuestreado en un archivo CSV
csv_output = os.path.join(save_path, f'serie_temporal_{year_folder}_remuestreado.csv')
df_remuestradeo.to_csv(csv_output, index=True)

print(f'DataFrame remuestreado guardado en: {csv_output}')

# Graficar la serie temporal
plt.figure(figsize=(15, 6))
plt.plot(df_remuestradeo.index, df_remuestradeo['B_total_smooth'], label='Campo Magnético Suavizado')
plt.title('Serie Temporal del Campo Magnético Suavizado - 2014')
plt.xlabel('Tiempo')
plt.ylabel('B_total_smooth (nT)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mayor claridad
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# Seleccionar solo la columna 'B_total_smooth' para el análisis
b_total_smooth = df_remuestradeo['B_total_smooth']

# Aplicar el filtro de Hodrick-Prescott
cycle, trend = hpfilter(b_total_smooth, lamb=12000)

# Añadir las columnas 'cycle' y 'trend' al DataFrame
df_remuestradeo['cycle'] = cycle
df_remuestradeo['trend'] = trend

# Verificar el resultado
print(df_remuestradeo.head(20))

# Graficar la tendencia
plt.figure(figsize=(15, 6))
plt.plot(df_remuestradeo.index, df_remuestradeo['trend'])
plt.title(f'Tendencia de B_total_smooth - Año {year_folder}')
plt.xlabel('Tiempo')
plt.ylabel('B_total_smooth (Tendencia)')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.show()

