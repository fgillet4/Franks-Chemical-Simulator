import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def run_symbolic_regression():
    try:
        # Load data
        data = pd.read_excel(file_name_entry.get(), engine='openpyxl')
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Get functions from checkboxes
        functions = []
        for func, var in function_vars.items():
            if var.get():
                functions.append(func)

        # Create symbolic regressor with user input
        sym_reg = SymbolicRegressor(
            population_size=int(pop_size_entry.get()),
            generations=int(generations_entry.get()),
            stopping_criteria=float(stopping_criteria_entry.get()),
            p_crossover=float(p_crossover_entry.get()),
            p_subtree_mutation=float(p_subtree_mutation_entry.get()),
            p_hoist_mutation=float(p_hoist_mutation_entry.get()),
            p_point_mutation=float(p_point_mutation_entry.get()),
            max_samples=float(max_samples_entry.get()),
            verbose=1,
            parsimony_coefficient=float(parsimony_coefficient_entry.get()),
            function_set=functions,
            random_state=0
        )

        # Fit
        sym_reg.fit(X_train, y_train)

        # Predict
        y_pred_train = sym_reg.predict(X_train)
        y_pred_test = sym_reg.predict(X_test)

        # Display results
        result_text.set(f"Empirical equation: {sym_reg._program}\nR^2 Train: {r2_score(y_train, y_pred_train)}\nR^2 Test: {r2_score(y_test, y_pred_test)}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
app = tk.Tk()
app.title("Symbolic Regression with Genetic Programming")

frame = ttk.Frame(app)
frame.grid(padx=20, pady=20, sticky=tk.W)

# Entry for filename
ttk.Label(frame, text="Data File:").grid(row=0, column=0)
file_name_entry = ttk.Entry(frame, width=40)
file_name_entry.grid(row=0, column=1, sticky=tk.W)
browse_button = ttk.Button(frame, text="Browse", command=lambda: file_name_entry.insert(0, filedialog.askopenfilename()))
browse_button.grid(row=0, column=2, sticky=tk.W)

# GP parameters entries
param_labels = ["Population Size", "Generations", "Stopping Criteria", "P Crossover", "P Subtree Mutation",
                "P Hoist Mutation", "P Point Mutation", "Max Samples", "Parsimony Coefficient"]
param_entries = [ttk.Entry(frame, width=10) for _ in param_labels]

for idx, (label, entry) in enumerate(zip(param_labels, param_entries)):
    ttk.Label(frame, text=label+":").grid(row=idx+1, column=0, sticky=tk.W)
    entry.grid(row=idx+1, column=1, sticky=tk.W)

pop_size_entry, generations_entry, stopping_criteria_entry, p_crossover_entry, p_subtree_mutation_entry, \
p_hoist_mutation_entry, p_point_mutation_entry, max_samples_entry, parsimony_coefficient_entry = param_entries

# Function checkboxes
ttk.Label(frame, text="Functions:").grid(row=len(param_labels)+1, column=0, sticky=tk.W)
function_vars = {"add": tk.BooleanVar(), "sub": tk.BooleanVar(), "mul": tk.BooleanVar(), "div": tk.BooleanVar()}
for idx, (func, var) in enumerate(function_vars.items()):
    ttk.Checkbutton(frame, text=func, variable=var).grid(row=len(param_labels)+2+idx, column=0, sticky=tk.W)

# Run button
run_button = ttk.Button(frame, text="Run", command=run_symbolic_regression)
run_button.grid(row=len(param_labels)+2+len(function_vars), column=0, columnspan=3)

# Result label
result_text = tk.StringVar()
result_label = ttk.Label(frame, textvariable=result_text)
result_label.grid(row=len(param_labels)+3+len(function_vars), column=0, columnspan=3)

app.mainloop()
