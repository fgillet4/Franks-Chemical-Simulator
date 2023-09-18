import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def check_and_resample():
    # Load the data
    file = input_file_entry.get()
    freq = freq_entry.get()
    
    try:
        data = pd.read_excel(file, parse_dates=True, index_col=0)  # assumes date is the first column
        
        if not (data.index.is_monotonic and (data.index.to_series().diff().mode()[0] == pd.to_timedelta(freq))):
            resampled_data = data.resample(freq).mean()
            resampled_data = resampled_data.interpolate()
            output_file = file.replace(".xlsx", "_resampled.xlsx")
            resampled_data.to_excel(output_file)
            messagebox.showinfo("Resampled", f"Data was not uniformly spaced. Resampled and saved as {output_file}.")
        else:
            messagebox.showinfo("Uniformly Spaced", "Data is already uniformly spaced.")
            
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Ensure Uniform Spacing in Time Series Data")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select Excel File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_input_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Choose resampling frequency
freq_label = tk.Label(frame, text="Desired Frequency (e.g., 'D' for daily):")
freq_label.grid(row=1, column=0, pady=10, sticky=tk.W)

freq_entry = tk.Entry(frame, width=10)
freq_entry.grid(row=1, column=1, pady=10, sticky=tk.W)
freq_entry.insert(0, "D")

# Process button
process_btn = tk.Button(frame, text="Check and Resample", command=check_and_resample)
process_btn.grid(row=2, column=0, columnspan=3, pady=20)

root.mainloop()
