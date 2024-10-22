import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.fftpack import fft, ifft, fftfreq
import pywt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.filters.hp_filter import hpfilter

directorio_actual = os.getcwd()

def leer_datosJRS(year_folder, path='.'):
    """Reads all .JRS files from the specified year folder and extracts Hcomp, Dcomp, Zcomp, and B (total magnetic field)."""
    
    Hcomp_all = []
    Dcomp_all = []
    Zcomp_all = []
    B_all = []  #lista para almacenar el campo magnético total B

    folder_path = os.path.join(path, year_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.JRS') and filename.startswith('S'):
            # Intentamos extraer la fecha del nombre del archivo
            try:
                year = int(filename[1:3]) + 2000  # Extraemos el año
                month = int(filename[3:5])  # Extraemos el mes
                day = int(filename[5:7])    # Extraemos el día

                # Verificamos que el mes y el día estén en rangos válidos
                if month < 1 or month > 12 or day < 1 or day > 31:
                    raise ValueError(f'Invalid month/day in filename: {filename}')

            except Exception as e:
                print(f'Error parsing date from filename: {filename} - {str(e)}')
                continue

            # Inicializamos los arrays
            Hcomp = np.full(86400, np.nan)
            Dcomp = np.full(86400, np.nan)
            Zcomp = np.full(86400, np.nan)

            full_path = os.path.join(folder_path, filename)

            if os.path.exists(full_path):
                with open(full_path, 'rb') as fp:
                    buf = np.frombuffer(fp.read((30 + 17 * 600) * 144), dtype=np.uint8)
                    buf = buf.reshape(144, 10230)
                    buf = np.transpose(buf)
                    buf = buf[30:, :]
                    buf = np.transpose(buf)
                    buf = buf.reshape(1468800, 1)

                    Hcomp = (buf[2::17] * 2**16 + buf[1::17] * 2**8 + buf[0::17]).astype(float)
                    Dcomp = (buf[5::17] * 2**16 + buf[4::17] * 2**8 + buf[3::17]).astype(float)
                    Zcomp = (buf[8::17] * 2**16 + buf[7::17] * 2**8 + buf[6::17]).astype(float)

                print(f'{full_path} loaded successfully!')

                # Ajustamos los datos
                Hcomp[Hcomp >= 2**23] -= 2**24
                Hcomp *= 0.01
                Hcomp[Hcomp > 80000] = np.nan

                Dcomp[Dcomp >= 2**23] -= 2**24
                Dcomp *= 0.01
                Dcomp[Dcomp > 80000] = np.nan

                Zcomp[Zcomp >= 2**23] -= 2**24
                Zcomp *= 0.01
                Zcomp[Zcomp > 80000] = np.nan

                # Cálculo del campo magnético total B
                B = np.sqrt(Hcomp**2 + Dcomp**2 + Zcomp**2)
                
                # Almacenamos los datos
                Hcomp_all.append(Hcomp.flatten())
                Dcomp_all.append(Dcomp.flatten())
                Zcomp_all.append(Zcomp.flatten())
                B_all.append(B.flatten())  # Almacenamos el campo total B
            else:
                print(f'{full_path} not found!')

    # Concatenamos todos los resultados
    Hcomp_all = np.concatenate(Hcomp_all) if Hcomp_all else np.array([])
    Dcomp_all = np.concatenate(Dcomp_all) if Dcomp_all else np.array([])
    Zcomp_all = np.concatenate(Zcomp_all) if Zcomp_all else np.array([])
    B_all = np.concatenate(B_all) if B_all else np.array([])  # Concatenamos los campos magnéticos totales B
    
    return Hcomp_all, Dcomp_all, Zcomp_all, B_all  # Devolvemos también el campo magnético total B

def guardar_datos_csv(year_folder, save_path, path='.'):
    """Guarda los datos obtenidos de los archivos .JRS en un archivo CSV con formato de serie temporal."""
    
    # Leemos los datos
    Hcomp_all, Dcomp_all, Zcomp_all, B_all = leer_datosJRS(year_folder, path)
    
    # Creamos la serie de tiempo (basado en los segundos en un día)
    time_index = pd.date_range(start=f'{year_folder[:4]}-01-01 00:00:00', periods=len(Hcomp_all), freq='S')
    
    # Creamos el DataFrame con los datos
    df = pd.DataFrame({
        'Time': time_index,
        'Hcomp': Hcomp_all,
        'Dcomp': Dcomp_all,
        'Zcomp': Zcomp_all,
        'B_total': B_all
    })

    # Guardamos el DataFrame en un archivo CSV
    csv_filename = os.path.join(save_path, f'serie_temporal_{year_folder}.csv')
    df.to_csv(csv_filename, index=False)

    print(f'Datos guardados en {csv_filename}')

def graficar_campo_total(csv_file, columna='B_total'):
    """Grafica la serie temporal del campo magnético total B desde un archivo CSV."""
    
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(csv_file)
    
    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora
    df['Time'] = pd.to_datetime(df['Time'])

    # Extraer los datos de tiempo y el campo magnético total B
    tiempo = df['Time']
    B_total = df[columna]

    # Obtener el año del primer dato de la columna 'Time'
    year = tiempo.dt.year.iloc[0]

    # Crear la figura y el gráfico
    plt.figure(figsize=(15, 6))

    # Graficar la serie temporal de B_total
    plt.plot(tiempo, B_total, color='m', label='Campo Magnético Total B')

    # Formato de la fecha en el eje X (mostrar solo los meses)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotar las etiquetas del eje X para que sean legibles
    plt.gcf().autofmt_xdate()

    # Añadir títulos y etiquetas
    plt.title('Campo Magnético Total B')
    plt.xlabel(f'{year}')
    plt.ylabel('Campo Magnético Total B (nT)')

    # Añadir cuadrícula para mayor legibilidad
    plt.grid(True)

    # Mostrar la leyenda
    plt.legend()

    # Ajustar el layout para evitar el solapamiento de las etiquetas
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

def procesar_csv(csv_file, window=86400, smooth_method='mean', remove_outliers=True, std_factor=3):
    """Limpia y procesa los datos de la serie temporal en el archivo CSV, y guarda solo el tiempo y B_total_smooth.
    Args:
         csv_file: str. Ruta al archivo CSV.
         window: int. Numero de datos para promediar el suavizado en este caso 1 dia (86400 segundos)
         remove_outliers: bool. Eliminar o no los datos atipicos.
         std_factor: int. Numero de desviaciones estandar para considerar un dato atipico        
    """
    
    # Cargar los datos desde el archivo CSV original
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f'Error al cargar el archivo: {e}')
        return None

    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora
    if 'Time' not in df.columns:
        print('La columna "Time" no se encuentra en el archivo CSV.')
        return None

    df['Time'] = pd.to_datetime(df['Time'])

    # Eliminar outliers si se indica
    if remove_outliers:
        mean = df['B_total'].mean()  # Media de la columna B_total
        std_dev = df['B_total'].std()  # Desviación estándar de la columna B_total
        
        # Filtrar los valores que están dentro de las 3 desviaciones estándar
        outlier_condition = (df['B_total'] < (mean - std_factor * std_dev)) | (df['B_total'] > (mean + std_factor * std_dev))
        outlier_count = df[outlier_condition].shape[0]  # Contar los outliers
        df = df[~outlier_condition]  # Eliminar los outliers

        print(f"Outliers eliminados: {outlier_count}")
        print(f"Cantidad de datos restantes después de eliminar outliers: {len(df)}")

    # Suavizado de la columna 'B_total'
    if smooth_method == 'mean':
        df['B_total_smooth'] = df['B_total'].rolling(window=window, min_periods=1).mean()
    elif smooth_method == 'median':
        df['B_total_smooth'] = df['B_total'].rolling(window=window, min_periods=1).median()
    else:
        print("Método de suavizado no reconocido. Usando media como método por defecto.")
        df['B_total_smooth'] = df['B_total'].rolling(window=window, min_periods=1).mean()

    # Crear un nuevo DataFrame con solo las columnas 'Time' y 'B_total_smooth'
    df_processed = df[['Time', 'B_total_smooth']]

    # Guardar el archivo procesado
    processed_csv = csv_file.replace('.csv', '_processed.csv')
    df_processed.to_csv(processed_csv, index=False)

    # Registro de acciones realizadas
    print(f'Datos procesados guardados en {processed_csv}')
    print(f'Valores nulos después de la interpolación: {df["B_total_smooth"].isnull().sum()}')

    return df_processed

def Remuestreo_FiltroHP(csv_file, window='24H', lamb_value=129600):
    """
    Procesa los datos de una serie temporal desde un archivo CSV, aplicando el filtro de Hodrick-Prescott y guardando los resultados.
    
    Args:
        csv_file: str. Ruta al archivo CSV.
        window: str. Intervalo para remuestreo de los datos (por defecto 24 horas).
        lamb_value: int. Valor de lambda para el filtro de Hodrick-Prescott (por defecto 129600 para datos mensuales).
            
    Returns:
        df_remuestradeo: pd.DataFrame. DataFrame procesado con columnas 'cycle' y 'trend' añadidas.
    """
    # Cargar los datos desde el archivo CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f'Error al cargar el archivo: {e}')
        return None

    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora
    if 'Time' not in df.columns:
        print('La columna "Time" no se encuentra en el archivo CSV.')
        return None

    df['Time'] = pd.to_datetime(df['Time'])

    # Eliminar valores NaN
    df_cleaned = df.dropna()

    # Configurar la columna 'Time' como índice
    df_cleaned.set_index('Time', inplace=True)

    # Remuestrear la serie temporal a intervalos de 'window'
    df_remuestradeo = df_cleaned.resample(window).mean()

    # Rellenar los valores NaN tras el remuestreo (forward fill)
    df_remuestradeo.fillna(method='ffill', inplace=True)

    # Aplicar el filtro de Hodrick-Prescott a 'B_total_smooth'
    cycle, trend = hpfilter(df_remuestradeo['B_total_smooth'], lamb=lamb_value)

    # Añadir las columnas 'cycle' y 'trend' al DataFrame
    df_remuestradeo['cycle'] = cycle
    df_remuestradeo['trend'] = trend

    # Guardar el DataFrame procesado en un archivo CSV nuevo
    processed_csv = csv_file.replace('.csv', '_remuestreado.csv')
    df_remuestradeo.to_csv(processed_csv)

    print(f"DataFrame procesado y guardado en: {processed_csv}")
    
    return df_remuestradeo

def graficar_trend(csv_file, columna='trend'):
    """Grafica la tendencia serie temporal del campo magnético total B desde un archivo CSV."""
    
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(csv_file)
    
    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora
    df['Time'] = pd.to_datetime(df['Time'])

    # Extraer los datos de tiempo y el campo magnético total B
    tiempo = df['Time']
    trend = df[columna]

    # Obtener el año del primer dato de la columna 'Time'
    year = tiempo.dt.year.iloc[0]

    # Crear la figura y el gráfico
    plt.figure(figsize=(15, 6))

    # Graficar la tendencia de la serie temporal 
    plt.plot(tiempo, trend, color='m', label='Tendencia del Campo Magnético B')

    # Formato de la fecha en el eje X (mostrar solo los meses)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotar las etiquetas del eje X para que sean legibles
    plt.gcf().autofmt_xdate()

    # Añadir títulos y etiquetas
    plt.title('Tendencia del Campo Magnético B')
    plt.xlabel(f'{year}')
    plt.ylabel('Campo Magnético Total B (nT)')

    # Añadir cuadrícula para mayor legibilidad
    plt.grid(True)

    # Mostrar la leyenda
    plt.legend()

    # Ajustar el layout para evitar el solapamiento de las etiquetas
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

def transformada_fourier_filtrado(csv_file, cutoff_freq=0.1, sampling_rate=10, plot_result=True, columna='B_total_smooth'):
    """
    Realiza la Transformada de Fourier Rápida (FFT) sobre los datos de B_total,
    filtra las frecuencias altas y reconstruye la señal con las frecuencias bajas (pasa-bajos).
    
    Args:
        csv_file: str. Ruta al archivo CSV que contiene los datos.
        cutoff_freq: float. Frecuencia de corte para el filtro pasa-bajos (en Hz, por defecto 0.1).
        sampling_rate: int. Frecuencia de muestreo de la señal en Hz (por defecto 1 Hz, 1 dato por segundo).
        plot_result: bool. Si se debe mostrar la señal original y filtrada en una gráfica (default=True).
        
    Returns:
        DataFrame: DataFrame con la señal original y la señal filtrada.
    """
    
    # Cargar los datos desde el archivo CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f'Error al cargar el archivo: {e}')
        return None

    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora
    df['Time'] = pd.to_datetime(df['Time'])

    # Obtener los valores de la columna B_total_smooth (señal original)
    y_signal = df[columna].values
    n = len(y_signal)  # Número de puntos en la señal

    # 1. Transformada de Fourier Rápida (FFT)
    fft_values = fft(y_signal)
    
    # Frecuencias asociadas a la FFT, ajustadas por la tasa de muestreo
    freqs = fftfreq(n, d=1/sampling_rate)  # sampling_rate es la frecuencia de muestreo (e.g., 1 Hz si es cada segundo)

    # 2. Filtrar las frecuencias altas (Filtro pasa-bajos)
    # Aplicar un filtro pasa-bajos eliminando las frecuencias por encima de cutoff_freq
    fft_values[np.abs(freqs) > cutoff_freq] = 0  # Filtro de frecuencias altas

    # 3. Reconstrucción de la señal a partir de las frecuencias filtradas
    y_filtered = ifft(fft_values).real  # Usamos ifft para reconstruir la señal en el dominio del tiempo

    # 4. Crear un nuevo DataFrame con los datos originales y la señal filtrada
    df['B_total_filtered'] = y_filtered

    # Guardar el archivo con la señal filtrada (opcional)
    # filtered_csv = csv_file.replace('.csv', '_fft_filtered.csv')
    # df.to_csv(filtered_csv, index=False)
    # print(f'Señal filtrada guardada en {filtered_csv}')

    # 5. Graficar el resultado (opcional)
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Time'], df[columna], label='Señal Original', alpha=0.6)
        plt.plot(df['Time'], df['B_total_filtered'], label='Señal Filtrada (Baja Frecuencia)', color='r')
        plt.title('Transformada de Fourier y Filtro Pasa-Bajos')
        plt.xlabel('Tiempo')
        plt.ylabel('Campo Magnético Total B (nT)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df

def graficar_autocorrelacion(csv_file, column='B_total_smooth', lags=3, plot_title='Gráfico de Autocorrelación con Lags'):
    """
    Genera el gráfico de autocorrelación de una serie temporal usando FFT, con la opción de especificar los lags.
    
    Args:
        csv_file: str. Ruta al archivo CSV que contiene los datos.
        column: str. Nombre de la columna a analizar (default='B_total_smooth').
        lags: int. Número de lags a considerar para el análisis de autocorrelación (default=3).
        plot_title: str. Título del gráfico (default='Gráfico de Autocorrelación con Lags').
        
    Returns:
        None. Solo genera y muestra el gráfico.
    """
    
    # Cargar los datos desde el archivo CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f'Error al cargar el archivo: {e}')
        return None

    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora
    if 'Time' not in df.columns:
        print('La columna "Time" no se encuentra en el archivo CSV.')
        return None

    df['Time'] = pd.to_datetime(df['Time'])

    # Verificar que la columna a analizar esté presente
    if column not in df.columns:
        print(f'La columna "{column}" no está presente en el archivo CSV.')
        return None
    
    # Obtener la señal de la columna seleccionada, eliminando valores nulos
    signal = df[column].dropna().values

    # Graficar la autocorrelación con el número de lags especificado
    plt.figure(figsize=(10, 6))
    plot_acf(signal, lags=lags)
    plt.title(plot_title)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelación')
    plt.grid(True)
    plt.show()

def graficar_original_y_procesada(csv_file, csv_depured, csv_remuestreada, columna_original='B_total', columna_procesada='B_total_smooth', columna_tendencia='trend'):
    """
    Grafica la serie temporal original en la parte superior y la serie procesada con la tendencia en la parte inferior.
    
    Args:
        csv_file: str. Ruta al archivo CSV que contiene la serie original.
        csv_depured: str. Ruta al archivo CSV que contiene la serie procesada.
        csv_remuestreada: str. Ruta al archivo CSV que contiene la tendencia.
        columna_original: str. Nombre de la columna con los datos originales (default='B_total').
        columna_procesada: str. Nombre de la columna con los datos procesados (default='B_total_smooth').
        columna_tendencia: str. Nombre de la columna con los datos de tendencia (default='trend').
        
    Returns:
        None. Solo genera y muestra el gráfico.
    """
    
    # Cargar los datos de la serie original desde el archivo CSV
    try:
        df_original = pd.read_csv(csv_file)
        df_depured = pd.read_csv(csv_depured)
        df_remuestreada = pd.read_csv(csv_remuestreada)
    except Exception as e:
        print(f'Error al cargar los archivos: {e}')
        return None

    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora en todos los archivos
    df_original['Time'] = pd.to_datetime(df_original['Time'])
    df_depured['Time'] = pd.to_datetime(df_depured['Time'])
    df_remuestreada['Time'] = pd.to_datetime(df_remuestreada['Time'])

    # Crear la figura y los subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Gráfico de la serie original
    ax1.plot(df_original['Time'], df_original[columna_original], label='Serie Original', color='blue')
    ax1.set_title('Serie Original')
    ax1.set_ylabel('Campo Magnético Total (nT)')
    ax1.grid(True)
    
    # Gráfico de la serie procesada con la tendencia
    ax2.plot(df_depured['Time'], df_depured[columna_procesada], label='Serie Procesada', color='green')
    ax2.plot(df_remuestreada['Time'], df_remuestreada[columna_tendencia], label='Tendencia', color='red', linestyle='--')
    ax2.set_title('Serie Procesada con Tendencia')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Campo Magnético Total (nT)')
    ax2.grid(True)

    # Ajustar el formato de las fechas en el eje X
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # Añadir las leyendas
    ax1.legend()
    ax2.legend()

    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

def leer_y_graficar_serie_anual(path, start_year, end_year):
    """
    Lee los archivos CSV en el formato 'serie_temporal_{año}_processed_remuestreado.csv' desde start_year hasta end_year,
    une todos los datos en un solo DataFrame y grafica la serie temporal B_total_smooth junto con la tendencia 'trend'.
    
    Args:
        path: str. Ruta donde se encuentran los archivos CSV.
        start_year: int. Año inicial de los archivos a leer.
        end_year: int. Año final de los archivos a leer.
        
    Returns:
        df_completo: pd.DataFrame. DataFrame con los datos unidos de todos los años.
    """
    
    # Inicializar una lista para almacenar los DataFrames de cada año
    dataframes = []
    
    # Recorrer cada año en el rango especificado
    for year in range(start_year, end_year + 1):
        file_name = f'serie_temporal_{year}_processed_remuestreado.csv'
        file_path = os.path.join(path, file_name)
        
        # Verificar si el archivo existe antes de intentar cargarlo
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['Time'] = pd.to_datetime(df['Time'])  # Asegurarse de que la columna 'Time' sea datetime
                dataframes.append(df)  # Añadir el DataFrame a la lista
                print(f"Archivo {file_name} cargado correctamente.")
            except Exception as e:
                print(f"Error al cargar {file_name}: {e}")
        else:
            print(f"Archivo {file_name} no encontrado.")
    
    # Concatenar todos los DataFrames en uno solo
    if dataframes:
        df_completo = pd.concat(dataframes, ignore_index=True)
    else:
        print("No se encontraron archivos para procesar.")
        return None
    
    # Graficar la serie temporal B_total_smooth y la tendencia 'trend'
    plt.figure(figsize=(14, 8))

    # Graficar B_total_smooth
    plt.plot(df_completo['Time'], df_completo['B_total_smooth'], label='B_total_smooth')

    # Graficar la tendencia 'trend'
    plt.plot(df_completo['Time'], df_completo['trend'], label='Trend', color='red', linestyle='--')

    # Formato de la fecha en el eje X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    # Rotar las etiquetas del eje X
    plt.gcf().autofmt_xdate()

    # Añadir títulos y etiquetas
    plt.title('Serie Temporal B_total_smooth y Tendencia')
    plt.xlabel('Tiempo')
    plt.ylabel('Campo Magnético Total (nT)')

    # Mostrar la leyenda
    plt.legend()

    # Mostrar la cuadrícula
    plt.grid(True)

    # Ajustar el layout
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

    return df_completo


    

