import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def apply_fft():
    # Load the data
    file = input_file_entry.get()
    column = column_entry.get()
    
    try:
        df = pd.read_csv(file, parse_dates=True, index_col=0)  # assumes date is the first column
        y = df[column].interpolate().astype(float)  # Interpolate NaNs and ensure float data type
        
        N = len(y)
        total_days = 365
        T = total_days / N
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

        # Plot FFT
        plt.figure(figsize=(10,6))
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.title(f"Fourier Transform of the Time Series Data for {column}")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Amplitude")
        plt.grid()

        # Highlight the most significant frequencies
        max_amplitude_idx = np.argmax(2.0/N * np.abs(yf[0:N//2]))
        max_frequency = xf[max_amplitude_idx]
        max_amplitude = (2.0/N * np.abs(yf[0:N//2]))[max_amplitude_idx]
        plt.annotate(f'Max Amplitude at Frequency: {max_frequency:.4f}',
                     xy=(max_frequency, max_amplitude),
                     xytext=(max_frequency + 0.01, max_amplitude - 5),
                     arrowprops=dict(facecolor='red', arrowstyle='->'),
                     color='red')

        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Apply FFT on Time Series Data")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select CSV File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_input_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Choose column for FFT
column_label = tk.Label(frame, text="Column for FFT:")
column_label.grid(row=1, column=0, pady=10, sticky=tk.W)

column_entry = tk.Entry(frame, width=20)
column_entry.grid(row=1, column=1, pady=10, sticky=tk.W)

# Process button
process_btn = tk.Button(frame, text="Apply FFT", command=apply_fft)
process_btn.grid(row=2, column=0, columnspan=3, pady=20)

root.mainloop()
