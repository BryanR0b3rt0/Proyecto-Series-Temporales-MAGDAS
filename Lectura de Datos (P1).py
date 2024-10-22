import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    
    # # Contar valores NaN
    # nan_hcomp = np.isnan(Hcomp_all).sum()
    # nan_dcomp = np.isnan(Dcomp_all).sum()
    # nan_zcomp = np.isnan(Zcomp_all).sum()
    # nan_bcomp = np.isnan(B_all).sum()  # Contamos los NaN en B

    # # Calcular porcentaje de NaN
    # total_hcomp = len(Hcomp_all)
    # total_dcomp = len(Dcomp_all)
    # total_zcomp = len(Zcomp_all)
    # total_bcomp = len(B_all)

    # print(f'Número de NaN en Hcomp: {nan_hcomp} ({(nan_hcomp / total_hcomp) * 100:.2f}%)')
    # print(f'Número de NaN en Dcomp: {nan_dcomp} ({(nan_dcomp / total_dcomp) * 100:.2f}%)')
    # print(f'Número de NaN en Zcomp: {nan_zcomp} ({(nan_zcomp / total_zcomp) * 100:.2f}%)')
    # print(f'Número de NaN en B (Campo Total): {nan_bcomp} ({(nan_bcomp / total_bcomp) * 100:.2f}%)')

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


def graficar_campo_total(csv_file):
    """Grafica la serie temporal del campo magnético total B desde un archivo CSV."""
    
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(csv_file)
    
    # Asegurarse de que la columna 'Time' se interprete como formato de fecha y hora
    df['Time'] = pd.to_datetime(df['Time'])

    # Extraer los datos de tiempo y el campo magnético total B
    tiempo = df['Time']
    B_total = df['B_total']

    # Crear la figura y el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar la serie temporal de B_total
    plt.plot(tiempo, B_total, color='m', label='Campo Magnético Total B')

    # Formato de la fecha en el eje X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    # Rotar las etiquetas del eje X para que sean legibles
    plt.gcf().autofmt_xdate()

    # Añadir títulos y etiquetas
    plt.title('Serie Temporal del Campo Magnético Total B')
    plt.xlabel('Tiempo')
    plt.ylabel('Campo Magnético Total B (nT)')

    # Añadir cuadrícula para mayor legibilidad
    plt.grid(True)

    # Mostrar la leyenda
    plt.legend()

    # Ajustar el layout para evitar el solapamiento de las etiquetas
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()


# Definimos el nombre de la carpeta que queremos leer
nombre_carpeta = '2020'
save_path = './data_series_temporales'  # Directorio donde se guardará el CSV

# Asegurarse de que la carpeta de destino exista
os.makedirs(save_path, exist_ok=True)

# Llamamos a la función para leer los datos de los archivos en la carpeta especificada
Hcomp, Dcomp, Zcomp, Btotal = leer_datosJRS(nombre_carpeta)

# Llamamos a la función para guardar los datos
guardar_datos_csv(nombre_carpeta, save_path)

csv_file = './data_series_temporales/serie_temporal_2020.csv'  # Ajusta el nombre del archivo según sea necesario

    # Llamar a la función para graficar el campo magnético total B
graficar_campo_total(csv_file)





