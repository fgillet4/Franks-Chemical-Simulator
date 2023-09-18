import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import pysindy as ps

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    if not file_path:
        return
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def apply_sindy():
    file_path = input_file_entry.get()
    if not file_path:
        messagebox.showerror("Error", "Please select an Excel file.")
        return
    try:
        df = pd.read_excel(file_path)
        data = df.values
        
        # Check if data is in the right format (time, variables)
        if data.shape[1] < 2:
            raise ValueError("Data should have at least two columns: time and one observation variable.")
        
        # Fetch the custom function library
        custom_functions = simpledialog.askstring("Input", "Enter your custom functions separated by comma (e.g. 'sin,cos,tan'):")
        if custom_functions:
            custom_functions = [x.strip() for x in custom_functions.split(",")]

            # Define the library
            function_library = ps.CustomLibrary(library_functions=custom_functions)

            # Apply SINDy
            model = ps.SINDy(feature_library=function_library)
            model.fit(data[:, 1:], t=data[:, 0])
            model_str = model.print()

            # Save to a text file
            with open('sindy_output.txt', 'w') as file:
                file.write(model_str)

            # Inform the user that the file has been saved
            messagebox.showinfo("Success", "SINDy model saved to 'sindy_output.txt'!")
        
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("SINDy Analysis")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

input_file_label = tk.Label(frame, text="Select Excel File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=load_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

apply_btn = tk.Button(frame, text="Apply SINDy", command=apply_sindy)
apply_btn.grid(row=1, column=0, columnspan=3, pady=20)

root.mainloop()
