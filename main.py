from Procesamiento_MAGDAS import leer_datosJRS, guardar_datos_csv, graficar_campo_total, graficar_trend
from Procesamiento_MAGDAS import procesar_csv
from Procesamiento_MAGDAS import transformada_fourier_filtrado
from Procesamiento_MAGDAS import graficar_autocorrelacion
from Procesamiento_MAGDAS import Remuestreo_FiltroHP
from Procesamiento_MAGDAS import graficar_original_y_procesada
from Procesamiento_MAGDAS import leer_y_graficar_serie_anual

import pandas as pd 
import os

# Especifica la carpeta del año y la ruta de los archivos .JRS
year_folder = '2023'
save_path = './data_series_temporales'
    
# Asegurarse de que la carpeta de destino exista
os.makedirs(save_path, exist_ok=True)

#________________Guardar los datos en CSV____________________
#guardar_datos_csv(year_folder, save_path)
csv_file = os.path.join(save_path, f'serie_temporal_{year_folder}.csv')


#________________Realizamos el procesamiento de datos_____________
#procesar_csv(csv_file)
csv_depured=csv_file.replace('.csv', '_processed.csv')


#__________Remuestreamos la Serie y aplicamos el Filtro HP para la tendencia___
#Remuestreo_FiltroHP(csv_depured,window='8H')
csv_remuestreada=csv_depured.replace('.csv', '_remuestreado.csv')

#__________Graficos Generales______________
#graficar_campo_total(csv_file) #Serie original sola
#graficar_campo_total(csv_depured, columna='B_total_smooth') #Serie Procesada
#graficar_campo_total(csv_remuestreada, columna='B_total_smooth') #Serie remuestreada
#graficar_trend(csv_remuestreada) #Tendencia de la serie Remuestreada
# graficar_original_y_procesada(csv_file,csv_depured,csv_remuestreada) #Todos los graficos unidos
df_completo = leer_y_graficar_serie_anual(save_path, start_year=2016, end_year=2017)

#______________REALIZAMOS LAS TRANSFORMADAS______________________
# transformada_fourier_filtrado(csv_remuestreada,columna='B_total_smooth')
# graficar_autocorrelacion(csv_remuestreada,lags=45)