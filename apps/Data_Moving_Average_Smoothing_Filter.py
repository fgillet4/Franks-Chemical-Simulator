import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def process_file():
    file = file_entry.get()
    
    if not file:
        messagebox.showerror("Error", "Please select a valid file.")
        return

    # Extract the file extension
    file_extension = '.' + file.split('.')[-1]

    try:
        if file_extension in ['.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
            Data = pd.read_excel(file)
        elif file_extension == '.csv':
            Data = pd.read_csv(file)
        elif file_extension == '.pkl':
            Data = pd.read_pickle(file)
        elif file_extension == '.json':
            Data = pd.read_json(file)
        else:
            raise Exception("Not a valid file extension")

        # Extract column indices from user input
        columns_to_smooth = [int(x) for x in columns_entry.get().split(",")]

        # Window size for moving average
        window_size = int(window_entry.get())

        # Apply moving average to selected columns
        for i in columns_to_smooth:
            Data.iloc[:,i] = Data.iloc[:,i].rolling(window=window_size).mean()

        # Drop rows with NaN values resulting from the moving average
        Data.dropna(inplace=True)

        output_file = file.replace(file_extension, '_smoothed' + file_extension)
        Data.to_excel(output_file, index=False)

        messagebox.showinfo("Success", f"File processed and saved as {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Time Series Smoothing Filter")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

label = tk.Label(frame, text="Select Excel File:")
label.grid(row=0, column=0, pady=10, sticky=tk.W)

file_entry = tk.Entry(frame, width=50)
file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

label = tk.Label(frame, text="Columns to Smooth (comma separated):")
label.grid(row=1, column=0, pady=10, sticky=tk.W)

columns_entry = tk.Entry(frame, width=50)
columns_entry.grid(row=1, column=1, pady=10)

label = tk.Label(frame, text="Window Size:")
label.grid(row=2, column=0, pady=10, sticky=tk.W)

window_entry = tk.Entry(frame, width=50)
window_entry.grid(row=2, column=1, pady=10)

process_btn = tk.Button(frame, text="Process File", command=process_file)
process_btn.grid(row=3, column=0, columnspan=3, pady=20)

root.mainloop()
