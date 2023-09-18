import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def perform_reduction():
    # Load the data
    file = input_file_entry.get()
    n_components = int(components_entry.get())

    try:
        data = pd.read_excel(file)

        # Apply PCA or SVD based on selection
        if reduction_var.get() == "PCA":
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(data)
        else:  # SVD
            U, s, VT = np.linalg.svd(data, full_matrices=False)
            reduced_data = U[:, :n_components] @ np.diag(s[:n_components])

        # Save the reduced data to a new Excel file
        output_file = file.replace(".xlsx", "_reduced.xlsx")
        pd.DataFrame(reduced_data).to_excel(output_file, index=False)
        messagebox.showinfo("Success", f"Data reduced and saved as {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Data Reduction with PCA/SVD")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select Excel File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_input_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Number of components
components_label = tk.Label(frame, text="Number of Components:")
components_label.grid(row=1, column=0, pady=10, sticky=tk.W)

components_entry = tk.Entry(frame, width=10)
components_entry.grid(row=1, column=1, pady=10, sticky=tk.W)
components_entry.insert(0, "2")

# Choose reduction method
reduction_var = tk.StringVar(value="PCA")
reduction_label = tk.Label(frame, text="Reduction Method:")
reduction_label.grid(row=2, column=0, pady=10, sticky=tk.W)

pca_radio = tk.Radiobutton(frame, text="PCA", variable=reduction_var, value="PCA")
pca_radio.grid(row=2, column=1, pady=10, sticky=tk.W)

svd_radio = tk.Radiobutton(frame, text="SVD", variable=reduction_var, value="SVD")
svd_radio.grid(row=2, column=1, pady=10, sticky=tk.E)

# Process button
process_btn = tk.Button(frame, text="Perform Reduction", command=perform_reduction)
process_btn.grid(row=3, column=0, columnspan=3, pady=20)

root.mainloop()
