import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    if not file_path:
        return

    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

    try:
        df = pd.read_excel(file_path)
        listbox.delete(0, tk.END)  # clear current items
        for column in df.columns:
            listbox.insert(tk.END, column)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def calculate_correlations():
    response_var = listbox.get(tk.ACTIVE)
    if not response_var:
        messagebox.showerror("Error", "Please select a response variable.")
        return

    file = input_file_entry.get()
    output_file = "correlations.txt"

    try:
        df = pd.read_excel(file)
        obs_columns = [col for col in df.columns if col != response_var]

        correlations = []
        for col in obs_columns:
            pearson_corr = df[col].corr(df[response_var], method='pearson')
            spearman_corr = df[col].corr(df[response_var], method='spearman')
            correlations.append((col, pearson_corr, spearman_corr))

        # Sort by Pearson R value
        correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

        with open(output_file, 'w') as f:
            f.write(f"Response Variable: {response_var}\n")
            f.write("Observation Column | Pearson Correlation | Spearman Correlation\n")
            f.write('-'*80 + '\n')
            for col, pearson, spearman in correlations:
                f.write(f"{col:20} | {pearson:20.3f} | {spearman:20.3f}\n")

        messagebox.showinfo("Success", f"Correlations saved to {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Correlation Analysis")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select Excel File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Load", command=load_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Listbox for column names
listbox = tk.Listbox(frame, width=50, height=10)
listbox.grid(row=1, column=0, columnspan=3, pady=10)

# Calculate correlations button
calculate_btn = tk.Button(frame, text="Calculate Correlations", command=calculate_correlations)
calculate_btn.grid(row=2, column=0, columnspan=3, pady=20)

root.mainloop()
