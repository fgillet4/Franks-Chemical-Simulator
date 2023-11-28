import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def select_columns():
    global obs_columns
    global resp_columns

    # Load the data
    file = input_file_entry.get()
    try:
        df = pd.read_excel(file)
        all_columns = df.columns.tolist()

        # Ask user for observation and response columns
        obs_columns = simpledialog.askstring("Select columns", "Enter observation columns separated by commas:").split(',')
        resp_columns = simpledialog.askstring("Select columns", "Enter response columns separated by commas:").split(',')

    except Exception as e:
        messagebox.showerror("Error", str(e))

def plot_data():
    global obs_columns
    global resp_columns

    # Load the data
    file = input_file_entry.get()
    try:
        df = pd.read_excel(file)

        # Create subplots for each combination
        fig, axs = plt.subplots(len(obs_columns), len(resp_columns), figsize=(15,10))
        for i, obs_col in enumerate(obs_columns):
            for j, resp_col in enumerate(resp_columns):
                axs[i, j].scatter(df[obs_col], df[resp_col])
                axs[i, j].set_xlabel(obs_col)
                axs[i, j].set_ylabel(resp_col)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Subplot n observations vs m responses")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select Excel File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_input_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Select columns
select_btn = tk.Button(frame, text="Select Columns", command=select_columns)
select_btn.grid(row=1, column=0, columnspan=3, pady=10)

# Plot button
plot_btn = tk.Button(frame, text="Plot Data", command=plot_data)
plot_btn.grid(row=2, column=0, columnspan=3, pady=20)

root.mainloop()
