import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def check_consistency():
    # Load the data
    file = input_file_entry.get()
    
    messages = []

    try:
        data = pd.read_excel(file)

        # Checking for data type consistency
        for column in data.columns:
            dtype_set = set(data[column].map(type))
            if len(dtype_set) > 1:
                messages.append(f"Inconsistent data types in column '{column}'.")

        # Check date columns for chronological order (assumes any column with 'date' in its name is a date column)
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        for date_col in date_cols:
            if not all(data[date_col].sort_values() == data[date_col]):
                messages.append(f"Column '{date_col}' is not in chronological order.")
        
        # Display result
        if messages:
            messagebox.showerror("Consistency Issues Found", "\n".join(messages))
        else:
            messagebox.showinfo("Success", "No consistency issues found.")
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Check Data Consistency in Excel")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select Excel File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_input_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Process button
process_btn = tk.Button(frame, text="Check Consistency", command=check_consistency)
process_btn.grid(row=1, column=0, columnspan=3, pady=20)

root.mainloop()
