import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

directorio_actual = os.getcwd()

def read_604rcsv(year, month, day, path=directorio_actual):
    """Reads a .JRS file and extracts Hcomp, Dcomp, and Zcomp."""
    
    Hcomp = np.full(86400, np.nan)
    Dcomp = np.full(86400, np.nan)
    Zcomp = np.full(86400, np.nan)
    
    if year > 2000:
        year -= 2000

    filename = f'{path}/S{year:02d}{month:02d}{day:02d}.JRS'

    if os.path.exists(filename):
        with open(filename, 'rb') as fp:
            buf = np.frombuffer(fp.read((30 + 17 * 600) * 144), dtype=np.uint8)
            buf = buf.reshape(144, 10230)
            buf = np.transpose(buf)
            buf = buf[30:, :]
            buf = np.transpose(buf)
            buf = buf.reshape(1468800, 1)

            Hcomp = (buf[2::17] * 2**16 + buf[1::17] * 2**8 + buf[0::17]).astype(float)
            Dcomp = (buf[5::17] * 2**16 + buf[4::17] * 2**8 + buf[3::17]).astype(float)
            Zcomp = (buf[8::17] * 2**16 + buf[7::17] * 2**8 + buf[6::17]).astype(float)

        print(f'{filename} loaded successfully!')
    else:
        print(f'{filename} not found!')

    # Adjust the data
    Hcomp[Hcomp >= 2**23] -= 2**24
    Hcomp *= 0.01
    Hcomp[Hcomp > 80000] = np.nan

    Dcomp[Dcomp >= 2**23] -= 2**24
    Dcomp *= 0.01
    Dcomp[Dcomp > 80000] = np.nan

    Zcomp[Zcomp >= 2**23] -= 2**24
    Zcomp *= 0.01
    Zcomp[Zcomp > 80000] = np.nan

    # Ensure that the arrays are 1-dimensional
    Hcomp = Hcomp.flatten()
    Dcomp = Dcomp.flatten()
    Zcomp = Zcomp.flatten()

    return Hcomp, Dcomp, Zcomp

def read_multiple_jrs(start_day, end_day, month, year, path=directorio_actual):
    """Reads multiple .JRS files from start_day to end_day and combines the data into one DataFrame."""
    
    all_data = []
    
    for day in range(start_day, end_day + 1):
        Hcomp, Dcomp, Zcomp = read_604rcsv(year, month, day, path)
        
        # Create time series for each day
        time = pd.date_range(f'{year}-{month:02d}-{day:02d} 00:00:00', periods=86400, freq='s')
        
        # Calculate B = sqrt(Hcomp^2 + Dcomp^2 + Zcomp^2)
        B = np.sqrt(Hcomp**2 + Dcomp**2 + Zcomp**2)
        
        # Create DataFrame for this day's data
        daily_df = pd.DataFrame({
            'Time': time,
            'Hcomp': Hcomp,
            'Dcomp': Dcomp,
            'Zcomp': Zcomp,
            'B': B
        })
        
        # Append the daily data to the list
        all_data.append(daily_df)
    
    # Concatenate all the daily data into one DataFrame
    combined_df = pd.concat(all_data, axis=0).reset_index(drop=True)
    
    return combined_df

def plot_components(df):
    """Plots Hcomp, Dcomp, Zcomp, and B from a combined DataFrame with x-axis showing hours only."""
    
    plt.figure(figsize=(10, 10))  # Adjust the height to accommodate 4 plots

    # Formatter for x-axis to show only the hour
    hours_fmt = mdates.DateFormatter('%H:%M')

    # Plot Hcomp
    plt.subplot(4, 1, 1)  # 4 rows for the 4 plots
    plt.plot(df['Time'], df['Hcomp'], color='b')
    plt.title('H Component')
    plt.ylabel('Hcomp (nT)')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(hours_fmt)

    # Plot Dcomp
    plt.subplot(4, 1, 2)
    plt.plot(df['Time'], df['Dcomp'], color='r')
    plt.title('D Component')
    plt.ylabel('Dcomp (nT)')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(hours_fmt)

    # Plot Zcomp
    plt.subplot(4, 1, 3)
    plt.plot(df['Time'], df['Zcomp'], color='g')
    plt.title('Z Component')
    plt.ylabel('Zcomp (nT)')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(hours_fmt)

    # Plot B (Total magnetic field magnitude)
    plt.subplot(4, 1, 4)
    plt.plot(df['Time'], df['B'], color='m')
    plt.title('Total Magnetic Field Magnitude (B)')
    plt.ylabel('B (nT)')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(hours_fmt)

    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Read data from 17th to 23rd October 2012
    combined_df = read_multiple_jrs(start_day=17, end_day=23, month=10, year=2012, path='./data_jrs')
    
    # Plot the combined data
    plot_components(combined_df)


