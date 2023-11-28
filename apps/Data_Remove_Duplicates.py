import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def remove_duplicates():
    # Load the data
    file = input_file_entry.get()

    try:
        data = pd.read_excel(file)
        
        # Removing duplicates
        initial_length = len(data)
        data = data.drop_duplicates()
        removed = initial_length - len(data)

        # Save the cleaned data to a new Excel file
        output_file = file.replace(".xlsx", "_no_duplicates.xlsx")
        data.to_excel(output_file, index=False)
        messagebox.showinfo("Success", f"Duplicates removed: {removed}. Cleaned data saved as {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Remove Duplicates from Excel Data")

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
process_btn = tk.Button(frame, text="Remove Duplicates", command=remove_duplicates)
process_btn.grid(row=1, column=0, columnspan=3, pady=20)

root.mainloop()
