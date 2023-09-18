import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

def run_anova():
    factor = factor_var.get()
    response = response_var.get()
    output_file = "anova_results.txt"

    if not factor or not response:
        messagebox.showerror("Error", "Please select both factor and response.")
        return

    try:
        df = pd.read_excel(input_file_entry.get())

        # Formulate the model
        model = ols(f'{response} ~ C({factor})', data=df).fit()
        
        # Perform ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)

        p_value = anova_table["PR(>F)"][0]

        # Interpret the results
        if p_value < 0.05:
            interpretation = "There is a statistically significant difference between the means."
        else:
            interpretation = "There is no statistically significant difference between the means."

        # Write results to file
        with open(output_file, 'w') as f:
            f.write("ANOVA Results\n")
            f.write("-"*30 + "\n")
            f.write(f"Factor: {factor}\n")
            f.write(f"Response: {response}\n")
            f.write("-"*30 + "\n")
            f.write(str(anova_table) + "\n\n")
            f.write("Interpretation:\n")
            f.write(interpretation + "\n")

        messagebox.showinfo("Success", f"ANOVA results saved to {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("ANOVA Analysis")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select Excel File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Load", command=load_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Dropdowns for selecting factor and response
factor_var = tk.StringVar(frame)
factor_label = tk.Label(frame, text="Select Factor:")
factor_label.grid(row=1, column=0, pady=10)
factor_dropdown = tk.OptionMenu(frame, factor_var, "")
factor_dropdown.grid(row=1, column=1, pady=10, padx=5, sticky=tk.W)

response_var = tk.StringVar(frame)
response_label = tk.Label(frame, text="Select Response:")
response_label.grid(row=2, column=0, pady=10)
response_dropdown = tk.OptionMenu(frame, response_var, "")
response_dropdown.grid(row=2, column=1, pady=10, padx=5, sticky=tk.W)

# Update the dropdown options
listbox = tk.Listbox(frame, width=50, height=10, exportselection=0)
listbox.grid(row=3, column=0, columnspan=3, pady=10)
listbox.bind('<<ListboxSelect>>', lambda e: update_dropdowns())

def update_dropdowns():
    selected = [listbox.get(i) for i in listbox.curselection()]
    factor_dropdown['menu'].delete(0, 'end')
    response_dropdown['menu'].delete(0, 'end')
    
    for name in selected:
        factor_dropdown['menu'].add_command(label=name, command=tk._setit(factor_var, name))
        response_dropdown['menu'].add_command(label=name, command=tk._setit(response_var, name))

# Button for running ANOVA
run_btn = tk.Button(frame, text="Run ANOVA", command=run_anova)
run_btn.grid(row=4, column=0, columnspan=3, pady=20)

root.mainloop()
