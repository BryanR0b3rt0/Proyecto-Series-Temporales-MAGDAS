import numpy as np
import pandas as pd
import os

def read_604rcsv(year, month, days):
    #MAC
    #directorio_actual = os.getcwd()
    #path = directorio_actual
    #directorio_carpeta=os.path.join(mag)
    path = '/Users/Lenovo/Downloads/mag'
    #PC
    #path = '/mnt/c/Users/CLIMA ESPACIAL/OneDrive - Escuela Politécnica Nacional/Escritorio/datos magdas_/DATOS20121017-20130114'
    Hcomp = np.full(86400, np.nan)
    Dcomp = np.full(86400, np.nan)
    Zcomp = np.full(86400, np.nan)
    IXcomp = np.full(86400, np.nan)
    IYcomp = np.full(86400, np.nan)
    TempS = np.full(86400, np.nan)
    TempP = np.full(86400, np.nan)

    if year > 2000:
        year -= 2000

    filename = f'{path}/S{year:02d}{month:02d}{days:02d}.JRS'

    if os.path.exists(filename):
        with open(filename, 'rb') as fp:
            buf = np.frombuffer(fp.read((30 + 17 * 600) * 144), dtype=np.uint8)
            buf = buf.reshape(144,10230)
            buf = np.transpose(buf)
            buf = buf[30:, :]
            buf = np.transpose(buf)
            buf = buf.reshape(1468800, 1)

            Hcomp = (buf[2::17] * 2**16 + buf[1::17] * 2**8 + buf[0::17]).astype(float)
            Dcomp = (buf[5::17] * 2**16 + buf[4::17] * 2**8 + buf[3::17]).astype(float)
            Zcomp = (buf[8::17] * 2**16 + buf[7::17] * 2**8 + buf[6::17]).astype(float)
            IXcomp = (buf[10::17] * 2**8 + buf[9::17]).astype(float)
            IYcomp = (buf[12::17] * 2**8 + buf[11::17]).astype(float)
            TempS = (buf[14::17] * 2**8 + buf[13::17]).astype(float)
            TempP = (buf[16::17] * 2**8 + buf[15::17]).astype(float)

        print(f'{filename} ok!')
    else:
        print(f'{filename} No existe!')

    Hcomp[Hcomp >= 2**23] -= 2**24
    Hcomp *= 0.01
    Hcomp[Hcomp > 80000] = np.nan

    Dcomp[Dcomp >= 2**23] -= 2**24
    Dcomp *= 0.01
    Dcomp[Dcomp > 80000] = np.nan

    Zcomp[Zcomp >= 2**23] -= 2**24
    Zcomp *= 0.01
    Zcomp[Zcomp > 80000] = np.nan

    IXcomp[IXcomp >= 2**15] -= 2**16
    IXcomp *= 0.1
    IXcomp[IXcomp > 3000] = np.nan

    IYcomp[IYcomp >= 2**15] -= 2**16
    IYcomp *= 0.1
    IYcomp[IYcomp > 3000] = np.nan

    if year < 100:
        year += 2000

    Time = pd.date_range(f'{year}-{month:02d}-{days:02d} 00:00:00', f'{year}-{month:02d}-{days:02d} 23:59:59', freq='S')

    return {'Hcomp': Hcomp, 'Dcomp': Dcomp, 'Zcomp': Zcomp,
            'IXcomp': IXcomp, 'IYcomp': IYcomp,
            'TempS': TempS, 'TempP': TempP, 'Time': Time}

# Example usage:
# data = read_604rcsv(22, 6, 29)
# print(data['Hcomp'])

