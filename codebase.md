# apps/Data_ANOVA.py

```py
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

```

# apps/Data_Check_Consistency.py

```py
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

```

# apps/Data_FFT_Technique.py

```py
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def apply_fft():
    # Load the data
    file = input_file_entry.get()
    column = column_entry.get()
    
    try:
        df = pd.read_csv(file, parse_dates=True, index_col=0)  # assumes date is the first column
        y = df[column].interpolate().astype(float)  # Interpolate NaNs and ensure float data type
        
        N = len(y)
        total_days = 365
        T = total_days / N
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

        # Plot FFT
        plt.figure(figsize=(10,6))
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.title(f"Fourier Transform of the Time Series Data for {column}")
        plt.xlabel("Frequency (cycles per day)")
        plt.ylabel("Amplitude")
        plt.grid()

        # Highlight the most significant frequencies
        max_amplitude_idx = np.argmax(2.0/N * np.abs(yf[0:N//2]))
        max_frequency = xf[max_amplitude_idx]
        max_amplitude = (2.0/N * np.abs(yf[0:N//2]))[max_amplitude_idx]
        plt.annotate(f'Max Amplitude at Frequency: {max_frequency:.4f}',
                     xy=(max_frequency, max_amplitude),
                     xytext=(max_frequency + 0.01, max_amplitude - 5),
                     arrowprops=dict(facecolor='red', arrowstyle='->'),
                     color='red')

        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Apply FFT on Time Series Data")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Input file selection
input_file_label = tk.Label(frame, text="Select CSV File:")
input_file_label.grid(row=0, column=0, pady=10, sticky=tk.W)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_input_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

# Choose column for FFT
column_label = tk.Label(frame, text="Column for FFT:")
column_label.grid(row=1, column=0, pady=10, sticky=tk.W)

column_entry = tk.Entry(frame, width=20)
column_entry.grid(row=1, column=1, pady=10, sticky=tk.W)

# Process button
process_btn = tk.Button(frame, text="Apply FFT", command=apply_fft)
process_btn.grid(row=2, column=0, columnspan=3, pady=20)

root.mainloop()

```

# apps/Data_Gaussian_Filter.py

```py
import pandas as pd
import numpy as np
import scipy.ndimage
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
        columns_to_filter = [int(x) for x in columns_entry.get().split(",")]

        # Sigma for Gaussian filter
        sigma = float(sigma_entry.get())

        # Apply Gaussian filter and identify outliers based on Z-scores
        for i in columns_to_filter:
            smoothed = scipy.ndimage.gaussian_filter(Data.iloc[:,i], sigma=sigma)
            z_scores = (Data.iloc[:,i] - smoothed) / np.std(smoothed)
            outliers = np.abs(z_scores) > 2.5
            Data.iloc[outliers, i] = np.nan  # replace outliers with NaN for now

        # Drop rows with NaN values (which are outliers)
        Data.dropna(inplace=True)

        output_file = file.replace(file_extension, '_filtered' + file_extension)
        Data.to_excel(output_file, index=False)

        messagebox.showinfo("Success", f"File processed and saved as {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Outliers Removal with Gaussian Filter")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

label = tk.Label(frame, text="Select Excel File:")
label.grid(row=0, column=0, pady=10, sticky=tk.W)

file_entry = tk.Entry(frame, width=50)
file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

label = tk.Label(frame, text="Columns to Filter (comma separated):")
label.grid(row=1, column=0, pady=10, sticky=tk.W)

columns_entry = tk.Entry(frame, width=50)
columns_entry.grid(row=1, column=1, pady=10)

label = tk.Label(frame, text="Sigma for Gaussian Filter:")
label.grid(row=2, column=0, pady=10, sticky=tk.W)

sigma_entry = tk.Entry(frame, width=50)
sigma_entry.grid(row=2, column=1, pady=10)

process_btn = tk.Button(frame, text="Process File", command=process_file)
process_btn.grid(row=3, column=0, columnspan=3, pady=20)

root.mainloop()

```

# apps/Data_Matrix_Filler_Interpolation.py

```py
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
        columns_to_interpolate = [int(x) for x in columns_entry.get().split(",")]

        # Interpolate selected columns
        for i in range(0, max(columns_to_interpolate)+1):
            if i in columns_to_interpolate:
                Data.iloc[:,i].interpolate(method='linear', axis=0, inplace=True)

        # Fill in any empty cells with 0 that we did not want to interpolate
        Data.fillna(0)

        # Only keep rows where all the values are greater than 0
        Data = Data[(Data > 0).all(axis=1)]

        output_file = file.replace(file_extension, '_filtered' + file_extension)
        Data.to_excel(output_file, index=False)

        messagebox.showinfo("Success", f"File processed and saved as {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Time Series Data Cleaner")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

label = tk.Label(frame, text="Select Excel File:")
label.grid(row=0, column=0, pady=10, sticky=tk.W)

file_entry = tk.Entry(frame, width=50)
file_entry.grid(row=0, column=1, pady=10)

browse_btn = tk.Button(frame, text="Browse", command=select_file)
browse_btn.grid(row=0, column=2, pady=10, padx=5)

label = tk.Label(frame, text="Columns to Interpolate (comma separated):")
label.grid(row=1, column=0, pady=10, sticky=tk.W)

columns_entry = tk.Entry(frame, width=50)
columns_entry.grid(row=1, column=1, pady=10)

process_btn = tk.Button(frame, text="Process File", command=process_file)
process_btn.grid(row=2, column=0, columnspan=3, pady=20)

root.mainloop()

```

# apps/Data_Moving_Average_Smoothing_Filter.py

```py
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

```

# apps/Data_Optimizer_RandomForest_ParticleSwarm.py

```py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pyswarms as ps
import pandas as pd

def on_run():
    # Get the filename from the entry field
    filename = file_name_entry.get()

    try:
        df = pd.read_excel(filename)
        df_filtered = df[(df > 0).all(axis=1)]

        data_array = np.array(df_filtered)
        values, params = data_array[:, 0], data_array[:, 1:]

        # Split the data
        params_train, params_test, values_train, values_test = train_test_split(params, values, test_size=0.2, random_state=42)

        # Initialize a random forest regressor
        rf = RandomForestRegressor(n_estimators=1500, random_state=42)

        # Train the model on your data
        rf.fit(params_train, values_train)

        # Predict function values for the test set
        values_pred = rf.predict(params_test)

        # Calculate the mean squared error of the predictions
        mse = r2_score(values_test, values_pred)

        def cost_func(position):
            return rf.predict(position)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        max_val, min_val = [], []
        dim = params.shape[1]
        for i in range(dim):
            max_val.append(np.amax(params[:, i]))
            min_val.append(np.amin(params[:, i]))
        bounds = (min_val, max_val)

        optimizer = ps.single.GlobalBestPSO(n_particles=1000, dimensions=dim, options=options, bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(cost_func, iters=1000)

        # Display results in GUI
        mse_label.config(text=f'MSE: {mse}')
        cost_label.config(text=f'Optimal Cost: {cost}')
        pos_label.config(text=f'Optimal Position: {pos}')

        save_to_file("results.txt", mse, cost, pos)
        messagebox.showinfo("Success", "Operation completed and results saved.")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def save_to_file(filename, mse, cost, pos):
    with open(filename, 'w') as f:
        f.write("Random Forest with Particle Swarm Optimization Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mean Squared Error of the predictions: {mse}\n\n")
        f.write("Optimization Results with PSO:\n")
        f.write(f"Optimal Cost: {cost}\n")
        f.write("Optimal Position: " + ', '.join(map(str, pos)) + "\n")

# GUI setup
app = tk.Tk()
app.title("Random Forest PSO App")

frame = ttk.Frame(app)
frame.grid(padx=20, pady=20)

# Create widgets
ttk.Label(frame, text="File Path:").grid(row=0, column=0, sticky=tk.W)
file_name_entry = ttk.Entry(frame, width=40)
file_name_entry.grid(row=0, column=1)

browse_button = ttk.Button(frame, text="Browse", command=lambda: file_name_entry.insert(0, filedialog.askopenfilename()))
browse_button.grid(row=0, column=2)

run_button = ttk.Button(frame, text="Run", command=on_run)
run_button.grid(row=1, columnspan=3)

mse_label = ttk.Label(frame, text="MSE: ")
mse_label.grid(row=2, columnspan=3)

cost_label = ttk.Label(frame, text="Optimal Cost: ")
cost_label.grid(row=3, columnspan=3)

pos_label = ttk.Label(frame, text="Optimal Position: ")
pos_label.grid(row=4, columnspan=3)

app.mainloop()

```

# apps/Data_PIML_Genetic_Programming.py

```py
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

```

# apps/Data_PIML_SINDy.py

```py
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

```

# apps/Data_Reduction_PCA_SVD.py

```py
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

```

# apps/Data_Remove_Duplicates.py

```py
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

```

# apps/Data_Spearman_Pearson_v2.py

```py
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

# Buttons for calculating correlations
select_all_btn = tk.Button(frame, text="Select All", command=lambda: listbox.select_set(0, tk.END))
select_all_btn.grid(row=2, column=0, columnspan=1, pady=10)

calculate_btn = tk.Button(frame, text="Calculate Correlations", command=calculate_correlations)
calculate_btn.grid(row=2, column=1, columnspan=2, pady=20)

root.mainloop()

```

# apps/Data_Spearman_Pearson.py

```py
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

```

# apps/Data_Subplotter.py

```py
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

```

# apps/Data_Temporal_Resampling.py

```py
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

```

# apps/Design_Electric_Motor_Design.py

```py

```

# apps/Design_Safety_Valve_Designer.py

```py
import tkinter as tk
from tkinter import ttk, StringVar
import fluids

def calculate_API520_A_g():
    try:
        m = float(entry_m_g.get())
        T = float(entry_T_g.get())
        Z = float(entry_Z_g.get())
        MW = float(entry_MW_g.get())
        k = float(entry_k_g.get())
        P1 = float(entry_P1_g.get())
        P2 = float(entry_P2_g.get())
        result = fluids.safety_valve.API520_A_g(m, T, Z, MW, k, P1, P2)
        result_var_g.set(f"Area: {result:.2f} [m^2]")
    except ValueError:
        result_var_g.set("Invalid input.")

def calculate_API520_A_steam():
    try:
        m = float(entry_m_steam.get())
        T = float(entry_T_steam.get())
        P1 = float(entry_P1_steam.get())
        result = fluids.safety_valve.API520_A_steam(m, T, P1)
        result_var_steam.set(f"Area: {result:.2f} [m^2]")
    except ValueError:
        result_var_steam.set("Invalid input.")

# ... Implement other functions similarly ...

root = tk.Tk()
root.title("Safety Valve Sizing")

# For API520_A_g
frame_g = ttk.LabelFrame(root, text="API520_A_g")
frame_g.pack(padx=10, pady=5, fill=tk.X)

labels_g = ["m (Mass Flow Rate) [kg/s]", "T (Temperature) [K]", "Z (Compressibility)", 
            "MW (Molecular Weight) g/mol", "k (Specific Heat Ratio)", 
            "P1 (Initial Pressure) [Pa]", "P2 (Final Pressure) [Pa]"]

entries_g = [entry_m_g, entry_T_g, entry_Z_g, entry_MW_g, entry_k_g, entry_P1_g, entry_P2_g] = [ttk.Entry(frame_g) for _ in labels_g]

for i, (label, entry) in enumerate(zip(labels_g, entries_g)):
    ttk.Label(frame_g, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry.grid(row=i, column=1, padx=10, pady=5)

ttk.Button(frame_g, text="Calculate", command=calculate_API520_A_g).grid(row=len(labels_g), column=0, columnspan=2, pady=10)
result_var_g = StringVar()
ttk.Label(frame_g, textvariable=result_var_g).grid(row=len(labels_g)+1, column=0, columnspan=2, pady=10)

# For API520_A_steam
frame_steam = ttk.LabelFrame(root, text="API520_A_steam")
frame_steam.pack(padx=10, pady=5, fill=tk.X)

labels_steam = ["Mass Flow Rate [kg/s]", "Temperature [K]", "P1 (Initial Pressure) [Pa]"]

entries_steam = [entry_m_steam, entry_T_steam, entry_P1_steam] = [ttk.Entry(frame_steam) for _ in labels_steam]

for i, (label, entry) in enumerate(zip(labels_steam, entries_steam)):
    ttk.Label(frame_steam, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry.grid(row=i, column=1, padx=10, pady=5)

ttk.Button(frame_steam, text="Calculate", command=calculate_API520_A_steam).grid(row=len(labels_steam), column=0, columnspan=2, pady=10)
result_var_steam = StringVar()
ttk.Label(frame_steam, textvariable=result_var_steam).grid(row=len(labels_steam)+1, column=0, columnspan=2, pady=10)

# ... You would add more frames for the other functions similarly ...

root.mainloop()

```

# apps/Design_ST_HEX_Orjan_Gui.py

```py
import os
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime
import numpy as np

# Assuming current_directory is defined as:
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'results' directory
results_dir = os.path.join(current_directory, 'results')

# Check if 'results' directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Generate the filename based on the current date including seconds
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"Shell_Tube_Heat_Exchanger_Results_{current_time}.xlsx"
excel_filename = f"Pipe_Cost_Results_{current_time}.xlsx"
csv_filename = f"Pipe_Cost_Results_{current_time}.csv"

excel_file_path = os.path.join(results_dir, excel_filename)
csv_file_path = os.path.join(results_dir, csv_filename)


def ST_HEX_solver_Orjan(tubeside_flow_l_s, tubeside_temp_in_C, tubeside_temp_out,
                        shellside_flow_l_s, shellside_temp_in_C, shellside_temp_out,
                        tubeside_pressure_in_Pa, shellside_pressure_in_Pa,
                        shell_length_mm, shell_diameter_mm, tube_length_mm, tube_diameter_mm,
                        tube_thickness_mm, tube_fouling_thickness_inner_mm, tube_fouling_thickness_outer_mm,
                        k_fouling_inner, k_fouling_outer, k_tube_material, nr_baffles, nr_tubes,
                        distance_between_tubes, angle_between_tubes, tube_row_length, tubeside_fluid_CAS,
                        shellside_fluid_CAS, tubeside_fluid_common_name, shellside_fluid_common_name,
                        parts, tolerance, cocurrent_or_countercurrent_flow, nr_HEX_passes, nr_heat_exchangers,
                        extra_area_procent,zig_zag_straight):
    
    def Cp_Orjan_etylenglykol(tempC, glykolhalt_procent):
        Cp = Cp_Orjan_vatten(tempC) - (Cp_Orjan_vatten(tempC) - 0.41) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100
        return Cp

    def Cp_Orjan_propylenglykol(tempC, glykolhalt_procent):
        Cp = (Cp_Orjan_vatten(tempC) * 1000 - (Cp_Orjan_vatten(tempC) * 1000 - 3590 + 3.875 * (tempC - 20)) * glykolhalt_procent / 50) / 1000
        return Cp

    def Cp_Orjan_vatten(tempC):
        if tempC < 40:
            Cp = 4.218 - 0.0018 * (tempC - 0) + 0.00004 * (tempC - 0) * (tempC - 20)
        else:
            Cp = 4.178 + 0.00045 * (tempC - 40) + 9.6875 * 10 ** -6 * (tempC - 40) * (tempC - 80) + 4.42708 * 10 ** -8 * (tempC - 40) * (tempC - 80) * (tempC - 120)
        return Cp

    def densitet_Orjan_etylenglykol(tempC, glykol_halt_procent):
        rho = densitet_Orjan_vatten(tempC) + (1080 - 0.75 * (tempC - 20) - densitet_Orjan_vatten(tempC)) * (0.8 * (glykol_halt_procent - 0) + 0.03 * (glykol_halt_procent - 0) * (glykol_halt_procent - 10)) / 100
        return rho

    def densitet_Orjan_propylenglykol(tempC):
        rho = densitet_Orjan_vatten(tempC) + 30
        return rho

    def densitet_Orjan_vatten(tempC):
        if tempC < 100:
            rho = 999.84 - 0.13966667 * (tempC - 0) - 0.004583333 * (tempC - 0) * (tempC - 30) + 1.73457 * 10 ** -5 * (tempC - 0) * (tempC - 30) * (tempC - 60)
        else:
            rho = 958.35 - 0.785 * (tempC - 100) - 0.002138889 * (tempC - 100) * (tempC - 130) - 9.25926 * 10 ** -7 * (tempC - 100) * (tempC - 130) * (tempC - 160)
        return rho

    def dynamisk_visk_Orjan_etylenglykol(tempC):
        mu = densitet_Orjan_etylenglykol(tempC) * kinematisk_visk_Orjan_etylenglykol(tempC)
        return mu

    def dynamisk_visk_Orjan_propylenglykol(tempC, glykolhalt_procent):
        mu = densitet_Orjan_propylenglykol(tempC) * kinematisk_visk_Orjan_propylenglykol(tempC, glykolhalt_procent)
        return mu

    def dynamisk_visk_Orjan_vatten(tempC):
        mu = (1792 - 33.16666667 * (tempC - 0) + 0.36944444 * (tempC - 0) * (tempC - 30) - 0.003006173 * (tempC - 0) * (tempC - 30) * (tempC - 60)) * 10 ** -6
        return mu

    def kinematisk_visk_Orjan_etylenglykol(tempC, glykolhalt_procent):
        if tempC < 80:
            mu = (3.8 - 0.075 * (tempC - 20) + 0.001 * (tempC - 20) * (tempC - 40) - 0.0000104167 * (tempC - 20) * (tempC - 40) * (tempC - 60) - kinematisk_visk_Orjan_vatten(tempC) * 10 ** 6) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        else:
            mu = (1.2 - (0.4 / 20) * (tempC - 80) - kinematisk_visk_Orjan_vatten(tempC) * 10 ** 6) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        return mu

    def kinematisk_visk_Orjan_propylenglykol(tempC, glykolhalt_procent):
        if tempC < 80:
            mu = 5.9 - 0.15 * (tempC - 20) + 0.002375 * (tempC - 20) * (tempC - 40) - 0.0000291667 * (tempC - 20) * (tempC - 40) * (tempC - 60) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        else:
            mu = 1.5 - (0.7 / 20) * (tempC - 80) - kinematisk_visk_Orjan_vatten(tempC) * 10 ** 6 * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        return mu

    def kinematisk_visk_Orjan_vatten(tempC):
        mu = dynamisk_visk_Orjan_vatten(tempC) / densitet_Orjan_vatten(tempC)
        return mu

    def D_eq_triangular_pitch(Pt, Do):
        D_eq = 4 * ((math.sqrt(3) / 4 * Pt ** 2 - math.pi * Do ** 2 / 8)) / (math.pi * Do * 0.5)
        return D_eq

    def D_eq_square_pitch(Pt, Do):
        D_eq = 4 * (Pt ** 2 - math.pi * Do ** 2 / 4) / (math.pi * Do)
        return D_eq

    def tube_orientation(S_T, S_L, diameter, zig_zag_or_straight):
        x = S_T / diameter
        y = S_L / diameter

        if zig_zag_or_straight == "straight":
            p00 = -0.5943
            p10 = 0.166
            p01 = 1.62
            p20 = 0.2007
            p11 = -0.3674
            p02 = -0.6684
            p30 = -0.05379
            p21 = 0.05296
            p12 = 0.02012
            p03 = 0.1051
            n = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3

            p00 = 3.447
            p10 = -1.878
            p01 = -2.832
            p20 = 0.3357
            p11 = 0.7697
            p02 = 1.117
            p30 = 0.0145
            p21 = -0.1731
            p12 = 0.01476
            p03 = -0.1903
            C = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3
        else:
            p00 = 0.8551
            p10 = -0.03898
            p01 = -0.7839
            p20 = 0.09826
            p11 = -0.1391
            p02 = 0.8411
            p30 = -0.02162
            p21 = -0.0209
            p12 = 0.06605
            p03 = -0.3466
            p31 = 0.008917
            p22 = -0.00415
            p13 = -0.007945
            p04 = 0.04896
            n = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3 + p31 * x ** 3 * y + p22 * x ** 2 * y ** 2 + p13 * x * y ** 3 + p04 * y ** 4

            p00 = 0.9872
            p10 = 2.321
            p01 = -6.566
            p20 = -0.628
            p11 = -5.315
            p02 = 15.27
            p21 = 1.335
            p12 = 3.608
            p03 = -13.23
            p22 = -0.8294
            p13 = -0.7861
            p04 = 4.799
            p23 = 0.1474
            p14 = 0.03581
            p05 = -0.6194
            C = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3 + p22 * x ** 2 * y ** 2 + p13 * x * y ** 3 + p04 * y ** 4 + p23 * x ** 2 * y ** 3 + p14 * x * y ** 4 + p05 * y ** 5

        return C, n

    def dittus_forced_convection(Re, Pr, n):
        Nu = 0.023 * Re ** 0.8 * Pr ** n
        return Nu

    def Gr_D(T, Ts, Tinf, D, nu):
        g = 9.81
        beta = 1 / T
        Gr_D = g * beta * (Ts - Tinf) * D ** 3 / nu ** 2
        return Gr_D

    def Ra_L(beta, Ts, Tinf, L, nu):
        Ra_L = 9.81 * beta * (Ts - Tinf) * L ** 3 / nu ** 2
        return Ra_L

    def k_Orjan_etylenglykol(tempC, glykolhalt_procent):
        k = k_Orjan_vatten(tempC) - (k_Orjan_vatten(tempC) - 0.41) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100
        return k

    def k_Orjan_propylenglykol(tempC, glykolhalt_procent):
        k = k_Orjan_vatten(tempC) - (k_Orjan_vatten(tempC) - 0.38) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100
        return k

    def k_Orjan_vatten(tempC):
        k = 0.553 + 0.002033 * (tempC - 0) - 1.277778 * 10 ** -5 * (tempC - 0) * (tempC - 30) + 5.55556 * 10 ** -8 * (tempC - 0) * (tempC - 30) * (tempC - 60)
        return k

    def Nu_D(h, D, k):
        Nu_D = h * D / k
        return Nu_D

    def Pe_D(velocity, D, L):
        Pe_D = velocity / (D / L)
        return Pe_D

    def Pr(mu, Cp, k):
        Pr = mu * Cp / k
        return Pr

    def Ra_L(Pr, Gr_L):
        Ra_L = Gr_L * Pr
        return Ra_L

    def Re_D(rho, V, D, mu):
        Re = rho * V * D / mu
        return Re

    def Re_L(rho, V, L, mu):
        Re = rho * V * L / mu
        return Re
    # Make a first guess for shellside outlet temperature
    shellside_temp_out_guess = shellside_temp_in_C * 1.5

    # Geometry Calculations
    S_L = math.cos(angle_between_tubes * math.pi / 180) * distance_between_tubes
    S_T = math.sin(angle_between_tubes * math.pi / 180) * distance_between_tubes * 2
    nr_horizontal_tubes = tube_row_length / S_T
    nr_tube_rows = nr_tubes / nr_horizontal_tubes
    width_of_tube_package = 2 * S_L * nr_tube_rows + S_L
    free_area = (distance_between_tubes - tube_diameter_mm) / 1000 * (tube_length_mm / 1000) * nr_horizontal_tubes / (nr_baffles + 1)
    shell_side_media_velocity = shellside_flow_l_s / 1000 / free_area
    tubeside_heat_transfer_area = nr_tubes * tube_diameter_mm * math.pi / 1000 * tube_length_mm / 1000

    heat_transfer_area_1 = tube_diameter_mm * math.pi / 1000 * tube_length_mm * nr_heat_exchangers * nr_HEX_passes / 1000 * nr_tubes / (1 + extra_area_procent)
    heat_transfer_area_2 = (tube_diameter_mm - (tube_diameter_mm - 2 * tube_thickness_mm)) / math.log(tube_diameter_mm / (tube_diameter_mm - 2 * tube_thickness_mm)) * math.pi / 1000 * tube_length_mm * nr_heat_exchangers * nr_HEX_passes / 1000 * nr_tubes / (1 + extra_area_procent)
    heat_transfer_area_ln = (heat_transfer_area_1 - heat_transfer_area_2) / (math.log(heat_transfer_area_1 / heat_transfer_area_2))
    active_area = heat_transfer_area_ln / parts

    # Create an empty list to store the calculated shellside temperatures
    shellside_tempC = []

    guesses = 0
    shellside_tempC_final = 0
    shellside_tempC = []

    while (abs(shellside_temp_in_C - shellside_tempC_final) / shellside_temp_in_C > tolerance):

        tubeside_velocity = (tubeside_flow_l_s / 1000) / (math.pi / 4 * (tube_diameter_mm / 1000 - 2 * tube_thickness_mm / 1000) ** 2 * nr_tubes)

        if cocurrent_or_countercurrent_flow in ["counter_current", "counter", "countercurrent", "CounterCurrent"]:
            shellside_velocity = (shellside_flow_l_s / 1000) / (math.pi / 4 * (shell_diameter_mm / 1000) ** 2 / nr_HEX_passes - 3.14 / 4 * (tube_diameter_mm / 1000) ** 2 * nr_tubes)
        else:
            shellside_velocity = shell_side_media_velocity

        result = [["Del av växlare", "tubeside_tempC", "mu_dynamic_tubeside_fluid", "rho_tubeside_fluid", "Cp_tubeside_fluid", "k_tubeside_fluid", "mu_kinematic_tubeside_fluid", "Re_tubeside", "Pr_tubeside", "tubeside_h",
                   "shellside_tempC", "mu_dynamic_shellside_fluid", "rho_shellside_fluid", "Cp_shellside_fluid", "k_shellside_fluid", "mu_kinematic_shellside_fluid", "Re_shellside", "Pr_shellside", "shellside_h", "active_area", "U_value", "my_v", "P"]]

        if not shellside_tempC:
            shellside_tempC.append(shellside_temp_out_guess)
        else:
            shellside_temp_out_guess = shellside_temp_out_guess + 0.1 * (shellside_temp_in_C - shellside_tempC[-1])
            shellside_tempC.append(shellside_temp_out_guess)

        i = 0
        tubeside_tempC = tubeside_temp_in_C

        while i <= parts:
            # ST_HEX: This function is for calculating the shell and tube heat ex.
            # This model takes into consideration...

            tubeside_pressure_Pa = tubeside_pressure_in_Pa
            shellside_pressure_Pa = shellside_pressure_in_Pa

            # Use specific libraries if the chemical is water
            if tubeside_fluid_CAS == "7732-18-5" or tubeside_fluid_common_name == "water":
                # rho = Mass density of water, [kg/m^3]
                rho_tubeside_fluid = densitet_Orjan_vatten(tubeside_tempC)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_tubeside_fluid = Cp_Orjan_vatten(tubeside_tempC)
                # k = Thermal Capacity, [W/mC]
                k_tubeside_fluid = k_Orjan_vatten(tubeside_tempC)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_tubeside_fluid = dynamisk_visk_Orjan_vatten(tubeside_tempC)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_tubeside_fluid = kinematisk_visk_Orjan_vatten(tubeside_tempC)
                # Velocity calculations based on the mass flux
            # Add other fluids if needed
            elif tubeside_fluid_CAS == "107-21-1" or tubeside_fluid_common_name == "etylenglykol":
                # rho = Mass density of ethylene glycol, [kg/m^3]
                rho_tubeside_fluid = densitet_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_tubeside_fluid = Cp_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_tubeside_fluid = k_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_tubeside_fluid = dynamisk_visk_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_tubeside_fluid = kinematisk_visk_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
            elif tubeside_fluid_CAS == "57-55-6" or tubeside_fluid_common_name == "propylenglykol":
                # rho = Mass density of propylene glycol, [kg/m^3]
                rho_tubeside_fluid = densitet_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_tubeside_fluid = Cp_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_tubeside_fluid = k_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_tubeside_fluid = dynamisk_visk_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_tubeside_fluid = kinematisk_visk_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
            if shellside_fluid_CAS == "7732-18-5" or shellside_fluid_common_name == "water":
                # rho = Mass density of water, [kg/m^3]
                rho_shellside_fluid = densitet_Orjan_vatten(shellside_tempC[-1])
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_shellside_fluid = Cp_Orjan_vatten(shellside_tempC[-1])
                # k = Thermal Capacity, [W/mC]
                k_shellside_fluid = k_Orjan_vatten(shellside_tempC[-1])
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_shellside_fluid = dynamisk_visk_Orjan_vatten(shellside_tempC[-1])
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_shellside_fluid = kinematisk_visk_Orjan_vatten(shellside_tempC[-1])
                # Velocity calculations based on the mass flux
            elif shellside_fluid_CAS == "107-21-1" or shellside_fluid_common_name == "etylenglykol"or shellside_fluid_common_name == "ehtyleneglycol":
                # rho = Mass density of ethylene glycol, [kg/m^3]
                rho_shellside_fluid = densitet_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_shellside_fluid = Cp_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_shellside_fluid = k_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_shellside_fluid = dynamisk_visk_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_shellside_fluid = kinematisk_visk_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
            elif shellside_fluid_CAS == "57-55-6" or shellside_fluid_common_name == "propylenglykol":
                # rho = Mass density of propylene glycol, [kg/m^3]
                rho_shellside_fluid = densitet_Orjan_propylenglykol(shellside_tempC[-1])
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_shellside_fluid = Cp_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_shellside_fluid = k_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_shellside_fluid = dynamisk_visk_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_shellside_fluid = kinematisk_visk_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
            
            # Add other fluids if needed

            # Calculate other properties for different fluids as needed

            Re_tubeside = Re_D(rho_tubeside_fluid, tubeside_velocity, (tube_diameter_mm - 2 * tube_thickness_mm) / 1000, mu_dynamic_tubeside_fluid)
            Re_shellside = Re_D(rho_shellside_fluid, shellside_velocity, tube_diameter_mm / 1000, mu_dynamic_shellside_fluid)
            Pr_tubeside = Pr(mu_dynamic_tubeside_fluid, Cp_tubeside_fluid * 1000, k_tubeside_fluid)
            Pr_shellside = Pr(mu_dynamic_shellside_fluid, Cp_shellside_fluid * 1000, k_shellside_fluid)

            tubeside_h = k_tubeside_fluid / ((tube_diameter_mm - 2 * tube_thickness_mm) / 1000) * 0.023 * Re_tubeside ** 0.8 * Pr_tubeside ** 0.4
            C, n = tube_orientation(S_T / 1000, S_L / 1000, tube_diameter_mm / 1000, zig_zag_straight)

            if cocurrent_or_countercurrent_flow in ["counter_current", "counter", "countercurrent", "CounterCurrent"]:
                shellside_h = k_shellside_fluid / (tube_diameter_mm / 1000) * 0.036 * Re_shellside ** 0.8 * Re_shellside ** 0.33
            else:
                shellside_h = k_shellside_fluid / (tube_diameter_mm / 1000) * C * Re_shellside ** n

            U_value = 1 / (1 / shellside_h + 1 / tubeside_h + (tube_thickness_mm / 1000) / k_tube_material + (tube_fouling_thickness_inner_mm / 1000) / k_fouling_inner + (tube_fouling_thickness_outer_mm / 1000) / k_fouling_outer)
            my_v = tubeside_tempC - U_value / tubeside_h * (tubeside_tempC - shellside_tempC[-1])
            P = tubeside_h * active_area * (tubeside_tempC - my_v) / 1000

            tubeside_tempC -= P / (tubeside_flow_l_s * Cp_tubeside_fluid)
            shellside_tempC[-1] -= P / (shellside_flow_l_s * Cp_shellside_fluid)

            result.append([i, tubeside_tempC, mu_dynamic_tubeside_fluid, rho_tubeside_fluid, Cp_tubeside_fluid, k_tubeside_fluid,
                           mu_kinematic_tubeside_fluid, Re_tubeside, Pr_tubeside, tubeside_h, shellside_tempC[-1],
                           mu_dynamic_shellside_fluid, rho_shellside_fluid, Cp_shellside_fluid, k_shellside_fluid,
                           mu_kinematic_shellside_fluid, Re_shellside, Pr_shellside, shellside_h, active_area, U_value, my_v, P])

            i += 1

        shellside_tempC_final = shellside_tempC[-1]
        guesses += 1

    return result


    

class OutputFrame(tk.Frame):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.__create_widgets()

    def __create_widgets(self):
        self.resultbutton = tk.Button(self, text="Generate Results", width=30, height=5, command=self.result_button_click)
        self.resultbutton.grid(column=0, row=0, padx=5, pady=10)

        self.warninglabel = tk.Label(self, text="");
        self.warninglabel.grid(column=0, row=1, padx=5, pady=10);

    def result_button_click(self):
        failures = 0
        for data in self.master.shared_data:
            if self.master.shared_data[data].get():
                pass
            elif data=="tube_fouling_thickness_inner_mm" or data =="tube_fouling_thickness_outer_mm" or data == "extra_area_procent": #add in or data=="" for any other variable that should be allowed to be 0
                pass
            else:
                failures += 1
                print(self.master.shared_data[data])
                print(failures)
                self.master.shared_data[data].set(0)
        if failures > 0:
            self.warninglabel['text'] = "Failure, unfilled field found!"
            self.warninglabel['fg'] = "red"
        else:
            self.master.calculate()

    def print_result(self, dim, cost):
        self.result_dim_label['text'] = "The cheapest is: " + str(1000 * dim) + "mm"
        self.result_cost_label['text'] = "The cost is: " + '{:0,.2f}'.format(cost).replace(",", "X").replace(".", ",").replace("X", " ") + "kr"

class InputFrame(tk.Frame):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=5)
        self.__create_widgets()

    def __create_widgets(self):
        def create_widget_pair(column, row, text, var):
            label = tk.Label(self, text=text)
            label.grid(column=column, row=row, sticky=tk.W, padx=5, pady=(10, 0))
            entry = tk.Entry(self, width=25, textvariable=self.master.shared_data[var])
            entry.grid(column=column, row=row + 1, sticky=tk.W, padx=5, pady=(5, 10))

        # Add more input widgets here as needed for heat exchanger data
        # For example:
        create_widget_pair(column=0, row=0, text='Tubeside Flow (L/s):', var="tubeside_flow_l_s")
        create_widget_pair(column=0, row=2, text='Tubeside Inlet Temp (C):', var="tubeside_temp_in_C")
        create_widget_pair(column=0, row=4, text='Tubeside Outlet Temp (C):', var="tubeside_temp_out")
        create_widget_pair(column=0, row=6, text='Shellside Flow (L/s):', var="shellside_flow_l_s")
        create_widget_pair(column=0, row=8, text='Shellside Inlet Temp (C):', var="shellside_temp_in_C")
        create_widget_pair(column=0, row=10, text='Shellside Outlet Temp (C):', var="shellside_temp_out")
        create_widget_pair(column=0, row=12, text='Tubeside Pressure (Pa):', var="tubeside_pressure_in_Pa")
        create_widget_pair(column=0, row=14, text='Shellside Pressure (Pa):', var="shellside_pressure_in_Pa")
        create_widget_pair(column=0, row=16, text='Shell Length (mm):', var="shell_length_mm")
        create_widget_pair(column=0, row=18, text='Shell Diameter (mm):', var="shell_diameter_mm")
        create_widget_pair(column=1, row=0, text='Tube Length (mm):', var="tube_length_mm")
        create_widget_pair(column=1, row=2, text='Tube Diameter (mm):', var="tube_diameter_mm")
        create_widget_pair(column=1, row=4, text='Tube Thickness (mm):', var="tube_thickness_mm")
        create_widget_pair(column=1, row=6, text='Tube Fouling Thickness Inner (mm):', var="tube_fouling_thickness_inner_mm")
        create_widget_pair(column=1, row=8, text='Tube Fouling Thickness Outer (mm):', var="tube_fouling_thickness_outer_mm")
        create_widget_pair(column=1, row=10, text='K Fouling Inner:', var="k_fouling_inner")
        create_widget_pair(column=1, row=12, text='K Fouling Outer:', var="k_fouling_outer")
        create_widget_pair(column=1, row=14, text='K Tube Material:', var="k_tube_material")
        create_widget_pair(column=1, row=16, text='Nr Baffles:', var="nr_baffles")
        create_widget_pair(column=1, row=18, text='Nr Tubes:', var="nr_tubes")
        create_widget_pair(column=2, row=0, text='Distance Between Tubes (mm):', var="distance_between_tubes")
        create_widget_pair(column=2, row=2, text='Angle Between Tubes (degrees):', var="angle_between_tubes")
        create_widget_pair(column=2, row=4, text='Tube Row Length (mm):', var="tube_row_length")
        create_widget_pair(column=2, row=6, text='Tubeside Fluid CAS:', var="tubeside_fluid_CAS")
        create_widget_pair(column=2, row=8, text='Shellside Fluid CAS:', var="shellside_fluid_CAS")
        create_widget_pair(column=2, row=10, text='Tubeside Fluid Common Name:', var="tubeside_fluid_common_name")
        create_widget_pair(column=2, row=12, text='Shellside Fluid Common Name:', var="shellside_fluid_common_name")
        create_widget_pair(column=2, row=14, text='Parts:', var="parts")
        create_widget_pair(column=2, row=16, text='Tolerance:', var="tolerance")
        create_widget_pair(column=2, row=18, text='Co-Current or Counter-Current Flow:', var="cocurrent_or_countercurrent_flow")
        create_widget_pair(column=3, row=0, text='Nr HEX Passes:', var="nr_HEX_passes")
        create_widget_pair(column=3, row=2, text='Nr Heat Exchangers:', var="nr_heat_exchangers")
        create_widget_pair(column=3, row=4, text='Extra Area Procent:', var="extra_area_procent")
        create_widget_pair(column=3, row=6, text='Zig Zag Straight:', var="zig_zag_straight")

        # Add more input widgets as needed

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        self.shared_data ={
            "tubeside_flow_l_s": tk.DoubleVar(),
            "tubeside_temp_in_C": tk.DoubleVar(),
            "tubeside_temp_out": tk.DoubleVar(),
            "shellside_flow_l_s": tk.DoubleVar(),
            "shellside_temp_in_C": tk.DoubleVar(),
            "shellside_temp_out": tk.DoubleVar(),
            "tubeside_pressure_in_Pa": tk.DoubleVar(),
            "shellside_pressure_in_Pa": tk.DoubleVar(),
            "shell_length_mm": tk.DoubleVar(),
            "shell_diameter_mm": tk.DoubleVar(),
            "tube_length_mm": tk.DoubleVar(),
            "tube_diameter_mm": tk.DoubleVar(),
            "tube_thickness_mm": tk.DoubleVar(),
            "tube_fouling_thickness_inner_mm": tk.DoubleVar(),
            "tube_fouling_thickness_outer_mm": tk.DoubleVar(),
            "k_fouling_inner": tk.DoubleVar(),
            "k_fouling_outer": tk.DoubleVar(),
            "k_tube_material": tk.DoubleVar(),
            "nr_baffles": tk.DoubleVar(),
            "nr_tubes": tk.DoubleVar(),
            "distance_between_tubes": tk.DoubleVar(),
            "angle_between_tubes": tk.DoubleVar(),
            "tube_row_length": tk.DoubleVar(),
            "tubeside_fluid_CAS": tk.StringVar(),
            "shellside_fluid_CAS": tk.StringVar(),
            "tubeside_fluid_common_name": tk.StringVar(),
            "shellside_fluid_common_name": tk.StringVar(),
            "parts": tk.DoubleVar(),
            "tolerance": tk.DoubleVar(),
            "cocurrent_or_countercurrent_flow": tk.StringVar(),
            "nr_HEX_passes": tk.DoubleVar(),
            "nr_heat_exchangers": tk.DoubleVar(),
            "extra_area_procent": tk.DoubleVar(),
            "zig_zag_straight": tk.StringVar()
        }
        self.__set_default_values()
        self.__create_widgets()
    def __set_default_values(self):
        self.shared_data["tubeside_flow_l_s"].set(317.0)
        self.shared_data["tubeside_temp_in_C"].set(84)
        self.shared_data["tubeside_temp_out"].set(76.5)
        self.shared_data["shellside_flow_l_s"].set(83)
        self.shared_data["shellside_temp_in_C"].set(49)
        self.shared_data["shellside_temp_out"].set(77.7)
        self.shared_data["tubeside_pressure_in_Pa"].set(201325)
        self.shared_data["shellside_pressure_in_Pa"].set(201325)
        self.shared_data["shell_length_mm"].set(7000)
        self.shared_data["shell_diameter_mm"].set(1440)
        self.shared_data["tube_length_mm"].set(5500)
        self.shared_data["tube_diameter_mm"].set(25)
        self.shared_data["tube_thickness_mm"].set(1)
        self.shared_data["tube_fouling_thickness_inner_mm"].set(0)
        self.shared_data["tube_fouling_thickness_outer_mm"].set(0)
        self.shared_data["k_fouling_inner"].set(0.9)
        self.shared_data["k_fouling_outer"].set(0.9)
        self.shared_data["k_tube_material"].set(16)
        self.shared_data["nr_baffles"].set(22)
        self.shared_data["nr_tubes"].set(240)
        self.shared_data["distance_between_tubes"].set(31.8)
        self.shared_data["angle_between_tubes"].set(30)
        self.shared_data["tube_row_length"].set(1520)
        self.shared_data["tubeside_fluid_CAS"].set("7732-18-5")
        self.shared_data["shellside_fluid_CAS"].set("7732-18-5")
        self.shared_data["tubeside_fluid_common_name"].set("water")
        self.shared_data["shellside_fluid_common_name"].set("water")
        self.shared_data["parts"].set(40)
        self.shared_data["tolerance"].set(0.005)
        self.shared_data["cocurrent_or_countercurrent_flow"].set("counter_current")
        self.shared_data["nr_HEX_passes"].set(2)
        self.shared_data["nr_heat_exchangers"].set(1)
        self.shared_data["extra_area_procent"].set(0)
        self.shared_data["zig_zag_straight"].set("straight")
    def __create_widgets(self):
        self.input_frame = InputFrame(self)
        self.input_frame.grid(column=0, row=0)

        self.output_frame = OutputFrame(self)
        self.output_frame.grid(column=1, row=0)
    def calculate(self):
        # Call the heat exchanger calculation function with the provided data   
        tubeside_flow_l_s= self.shared_data['tubeside_flow_l_s'].get()
        tubeside_temp_in_C=self.shared_data['tubeside_temp_in_C'].get()
        tubeside_temp_out=self.shared_data['tubeside_temp_out'].get()
        shellside_flow_l_s=self.shared_data['shellside_flow_l_s'].get()
        shellside_temp_in_C=self.shared_data['shellside_temp_in_C'].get()
        shellside_temp_out=self.shared_data['shellside_temp_out'].get()
        tubeside_pressure_in_Pa=self.shared_data['tubeside_pressure_in_Pa'].get()
        shellside_pressure_in_Pa=self.shared_data['shellside_pressure_in_Pa'].get()
        shell_length_mm=self.shared_data['shell_length_mm'].get()
        shell_diameter_mm=self.shared_data['shell_diameter_mm'].get()
        tube_length_mm=self.shared_data['tube_length_mm'].get()
        tube_diameter_mm=self.shared_data['tube_diameter_mm'].get()
        tube_thickness_mm=self.shared_data['tube_thickness_mm'].get()
        tube_fouling_thickness_inner_mm=self.shared_data['tube_fouling_thickness_inner_mm'].get()
        tube_fouling_thickness_outer_mm=self.shared_data['tube_fouling_thickness_outer_mm'].get()
        k_fouling_inner=self.shared_data['k_fouling_inner'].get()
        k_fouling_outer=self.shared_data['k_fouling_outer'].get()
        k_tube_material=self.shared_data['k_tube_material'].get()
        nr_baffles=self.shared_data['nr_baffles'].get()
        nr_tubes=self.shared_data['nr_tubes'].get()
        distance_between_tubes=self.shared_data['distance_between_tubes'].get()
        angle_between_tubes=self.shared_data['angle_between_tubes'].get()
        tube_row_length=self.shared_data['tube_row_length'].get()
        tubeside_fluid_CAS=self.shared_data['tubeside_fluid_CAS'].get()
        shellside_fluid_CAS=self.shared_data['shellside_fluid_CAS'].get()
        tubeside_fluid_common_name=self.shared_data['tubeside_fluid_common_name'].get()
        shellside_fluid_common_name=self.shared_data['shellside_fluid_common_name'].get()
        parts=self.shared_data['parts'].get()
        tolerance=self.shared_data['tolerance'].get()
        cocurrent_or_countercurrent_flow=self.shared_data['cocurrent_or_countercurrent_flow'].get()
        nr_HEX_passes=self.shared_data['nr_HEX_passes'].get()
        nr_heat_exchangers=self.shared_data['nr_heat_exchangers'].get()
        extra_area_procent=self.shared_data['extra_area_procent'].get()
        zig_zag_straight=self.shared_data['zig_zag_straight'].get()

        self.results = ST_HEX_solver_Orjan(
            tubeside_flow_l_s,
            tubeside_temp_in_C,
            tubeside_temp_out,
            shellside_flow_l_s,
            shellside_temp_in_C,
            shellside_temp_out,
            tubeside_pressure_in_Pa,
            shellside_pressure_in_Pa,
            shell_length_mm,
            shell_diameter_mm,
            tube_length_mm,
            tube_diameter_mm,
            tube_thickness_mm,
            tube_fouling_thickness_inner_mm,
            tube_fouling_thickness_outer_mm,
            k_fouling_inner,
            k_fouling_outer,
            k_tube_material,
            nr_baffles,
            nr_tubes,
            distance_between_tubes,
            angle_between_tubes,
            tube_row_length,
            tubeside_fluid_CAS,
            shellside_fluid_CAS,
            tubeside_fluid_common_name,
            shellside_fluid_common_name,
            parts,
            tolerance,
            cocurrent_or_countercurrent_flow,
            nr_HEX_passes,
            nr_heat_exchangers,
            extra_area_procent,
            zig_zag_straight
        )

        # Extract the relevant columns from the result for plotting and saving as CSV
        T = pd.DataFrame(self.results[1:], columns=self.results[0])
        T_data = T[["Del av växlare", "tubeside_tempC", "shellside_tempC"]]
        self.plot_graph(T_data)

        # Save the result as a CSV file
        T_data.to_csv(csv_file_path, index=False)
        # Save the result as an Excel file
        T_data.to_excel(excel_file_path, sheet_name='sheet1', index=False)
    
    def plot_graph(self, data):
        import matplotlib.pyplot as plt

        x_data = data["Del av växlare"]
        y_data_1 = data["tubeside_tempC"]
        y_data_2 = data["shellside_tempC"]

        tubeside_inlet_temp = float(self.shared_data["tubeside_temp_in_C"].get())
        shellside_inlet_temp = float(self.shared_data["shellside_temp_in_C"].get())

        if tubeside_inlet_temp > shellside_inlet_temp:
            warm_data, cold_data = y_data_1, y_data_2
            warm_label = "Tube Side (Warm Side)"
            cold_label = "Shell Side (Cold Side)"
        else:
            warm_data, cold_data = y_data_2, y_data_1
            warm_label = "Shell Side (Warm Side)"
            cold_label = "Tube Side (Cold Side)"

        flow_type = self.shared_data["cocurrent_or_countercurrent_flow"].get()
        print(flow_type)
        plt.plot(x_data, warm_data, 'r', label=warm_label)
        plt.plot(x_data, cold_data, 'b', label=cold_label)

        arrow_spacing = len(x_data) // 10
        arrow_dx = (x_data.iloc[-1] - x_data.iloc[0]) / 10
        arrow_properties = dict(shape='full', lw=0, length_includes_head=True, head_width=2, head_length=1)

        for i in range(0, len(x_data) - arrow_spacing, arrow_spacing):
            # Warm side arrow follows the line
            dx_warm = x_data[i + arrow_spacing] - x_data[i]
            dy_warm = warm_data[i + arrow_spacing] - warm_data[i]
            plt.arrow(x_data[i], warm_data[i], dx_warm, dy_warm, color='r', **arrow_properties)
            
            # Compute the dx and dy for the cold side regardless of the flow type
            dx_cold = x_data[i + arrow_spacing] - x_data[i]
            dy_cold = cold_data[i + arrow_spacing] - cold_data[i]

            # Check flow type for cold side
            if flow_type in ["counter_current", "counter", "countercurrent", "CounterCurrent"]:
                # Cold side arrow starts from the future point and points back for counter-current flow
                plt.arrow(x_data[i + arrow_spacing], cold_data[i + arrow_spacing], -dx_cold, -dy_cold, color='b', **arrow_properties)
            else:
                # Cold side arrow follows the line for co-current flow
                plt.arrow(x_data[i], cold_data[i], dx_cold, dy_cold, color='b', **arrow_properties)
        # Annotate the starting temperatures
        # Left Side warm text
        plt.annotate(f"{warm_data.iloc[0]:.2f}°C", 
                    (x_data.iloc[0], warm_data.iloc[0]),
                    textcoords="offset points", 
                    xytext=(-40,-7), 
                    ha='center',
                    color='r')
        # Left Side cold text
        plt.annotate(f"{cold_data.iloc[0]:.2f}°C", 
                    (x_data.iloc[0], cold_data.iloc[0]),
                    textcoords="offset points", 
                    xytext=(-40,-5), 
                    ha='center',
                    color='b')

        # Annotate the end temperatures
        # Right Side warm text
        plt.annotate(f"{warm_data.iloc[-1]:.2f}°C", 
                    (x_data.iloc[-1], warm_data.iloc[-1]),
                    textcoords="offset points", 
                    xytext=(40,-10), 
                    ha='center',
                    color='r')
        # Right Side cold text
        plt.annotate(f"{cold_data.iloc[-1]:.2f}°C", 
                    (x_data.iloc[-1], cold_data.iloc[-1]),
                    textcoords="offset points", 
                    xytext=(40,10), 
                    ha='center',
                    color='b')
        plt.title("S&T HEX Temperature Profile")
        plt.xlabel("Slice of Heat Exchanger")
        plt.ylabel("Temperature C")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    app = App()
    app.mainloop()

```

# apps/Economics_Kostnadsoptimering_Ror.py

```py
#import re
import xml.etree.ElementTree as ET
import math
import fluids
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import tkinter as tk
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

class Calculations:
    def velocity(d,q):
        return (q/3600)/(((d/2)**2)*math.pi);
        
    #friction coeffecient for laminar flow
    def laminar(re):
        f = 64/re;
        return f;

    def reynolds_number(v,den,dim,dyn_vis):
        rey = den*v*dim/dyn_vis;
        return rey;
        
    #calculate friction coeffecient for turbulent flow using Mileikovskyi method, no iteration needed
    def Mileikovskyi(re,rr): 
        A0 = -0.79638*math.log((rr/8.298)+(7.3357/re));
        A1 = re*rr+9.3120665*A0;
        f=((8.128943+A1)/(8.128943*A0-0.86859209*A1*math.log(A1/(3.7099535*re))))**2;
        return f;

    #use fluids to calculate headloss from bends
    def bend_calc(d,f,v,sys_length,bends,den):
        K = fluids.fittings.entrance_sharp();
        for n in range(math.floor(bends)):
            K += fluids.fittings.bend_rounded(Di=d,angle=90,fd=f);
        K += fluids.fittings.exit_normal();
        K += fluids.core.K_from_f(fd=f, L=sys_length, D=d)
        loss = fluids.core.dP_from_K(K=K,rho=den,V=v); # returns pressure loss in Pa
        return loss;

    #yearly energy cost
    def calc_en_cost(head,q,pump_eff,en_cost,yearly_h,year,endev):
        kw= q*head/(3599000*pump_eff);
       # print("kw: " + str(kw));
        return [en_cost*((1+endev)**year)*kw*yearly_h,kw];


    def calc_con_cost(mcost,d,sys_length,spots_w,speed_w,sal_w,sal_i,sal_a,time_i,time_a,price_i,price_a,work_eff,scaff,thic_m,thic_i):
        con_cost = scaff;
        con_cost += mcost * sys_length;
        con_cost += (d+2*thic_m)*math.pi*spots_w*speed_w*sal_w/work_eff;
        con_cost += time_i*sal_i/work_eff + (d+2*thic_m+2*thic_i)*math.pi*sys_length*price_i;
        con_cost += time_a*sal_a/work_eff + (d+2*thic_m+2*thic_i)*math.pi*sys_length*price_a;
        
        #print("con cost: " + str(con_cost))
        return con_cost

 


class OutputFrame(tk.Frame):
    def __init__(self,master):
        super().__init__();
        self.master = master;
        self.columnconfigure(0, weight=1);
        self.columnconfigure(1, weight=3);
        self.__create_widgets();
        
    def __create_widgets(self):
        self.resultbutton = tk.Button(self, text="Generera resultat",width=30, height=5, command = self.result_button_click);
        self.resultbutton.grid(column=0, row=0, padx=5, pady=10);
        
        self.warninglabel = tk.Label(self, text="");
        self.warninglabel.grid(column=0, row=1, padx=5, pady=10);
        
        self.result_dim_label = tk.Label(self, text="", font=25);
        self.result_dim_label.grid(column=0, row=1, padx=5, pady=10);
        self.result_cost_label = tk.Label(self, text="", font=25);
        self.result_cost_label.grid(column=0, row=2, padx=5, pady=10);
        
        
    def result_button_click(self):
        #Check that all fields are filld in or it wont execute the calculations and instead give a red warning text
        failures = 0;
        for data in self.master.shared_data:
            if self.master.shared_data[data].get():
                pass
            #    print("success");
            elif data=="endevvar": #add in or data=="" for any other variable that should be allowed to be 0
                pass
            else:
                failures +=1;
              #  print("Missing: " + data);
                self.master.shared_data[data].set(0)
        if failures > 0:
             self.warninglabel['text'] = "Misslyckande, icke ifyllt fält funnet!";
             self.warninglabel['fg'] = "red";
        else:
           self.master.calculate()
           
    def print_result(self, dim,cost):
        self.result_dim_label['text'] = "Den billigaste är: " + str(1000*dim) + "mm";
        self.result_cost_label['text'] = "Den kostar: " + '{:0,.2f}'.format(cost).replace(",","X").replace(".",",").replace("X"," ") + "kr"
            
class InputFrame(tk.Frame):
    def __init__(self,master):
        super().__init__();
        
        self.columnconfigure(0, weight=1);
        self.columnconfigure(1, weight=3);
        self.columnconfigure(2, weight=5);
        self.__create_widgets();
          
    def __create_widgets(self):
        
        ##Column 0
        
        flowlabel = tk.Label(self, text='Flöde [m³/h]');
        flowlabel.grid(column=0,row=0,sticky=tk.W,padx=5,pady=(10,0));
        flowentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["flowvar"]);
        flowentry.grid(column=0,row=1,sticky=tk.W,padx=5,pady=(5,10));
        
        label1 = tk.Label(self, text='Densitet [kg/m³]');
        label1.grid(column=0,row=2,sticky=tk.W,padx=5,pady=(10,0));
        entry1 = tk.Entry(self, width=25, textvariable=self.master.shared_data["denvar"]);
        entry1.grid(column=0,row=3,sticky=tk.W,padx=5,pady=(5,10));
        
        dynvislabel = tk.Label(self, text='Dynamisk viskositet [Pa*s]');
        dynvislabel.grid(column=0,row=4,sticky=tk.W,padx=5,pady=(10,0));
        dynvisentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["dynvisvar"]);
        dynvisentry.grid(column=0,row=5,sticky=tk.W,padx=5,pady=(5,10));
        
        minvlabel = tk.Label(self, text='Min hastighet [m/s]');
        minvlabel.grid(column=0,row=6,sticky=tk.W,padx=5,pady=(10,0));
        minventry = tk.Entry(self, width=25, textvariable=self.master.shared_data["minvvar"]);
        minventry.grid(column=0,row=7,sticky=tk.W,padx=5,pady=(5,10));
        
        maxvlabel = tk.Label(self, text='Max hastighet [m/s]');
        maxvlabel.grid(column=0,row=8,sticky=tk.W,padx=5,pady=(10,0));
        maxventry = tk.Entry(self, width=25, textvariable=self.master.shared_data["maxvvar"]);
        maxventry.grid(column=0,row=9,sticky=tk.W,padx=5,pady=(5,10));
        
        pumpefflabel = tk.Label(self, text='Pump verkningsgrad [0-1]');
        pumpefflabel.grid(column=0,row=10,sticky=tk.W,padx=5,pady=(10,0));
        pumpeffentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["pumpeffvar"]);
        pumpeffentry.grid(column=0,row=11,sticky=tk.W,padx=5,pady=(5,10));
        
        potheadlabel = tk.Label(self, text='Pump uppfodringshöjd [m]');
        potheadlabel.grid(column=0,row=12,sticky=tk.W,padx=5,pady=(10,0));
        potheadentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["potheadvar"]);
        potheadentry.grid(column=0,row=13,sticky=tk.W,padx=5,pady=(5,10));
        
        heightlabel = tk.Label(self, text='Höjdskillnad [m]');
        heightlabel.grid(column=0,row=14,sticky=tk.W,padx=5,pady=(10,0));
        heightentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["heightvar"]);
        heightentry.grid(column=0,row=15,sticky=tk.W,padx=5,pady=(5,10));
        
        yearlylabel = tk.Label(self, text='Årsförbrukning [h]');
        yearlylabel.grid(column=0,row=16,sticky=tk.W,padx=5,pady=(10,0));
        yearlyentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["yearlyhvar"]);
        yearlyentry.grid(column=0,row=17,sticky=tk.W,padx=5,pady=(5,10));
        
        encostlabel = tk.Label(self, text='Elkostnad [kr/kwh]');
        encostlabel.grid(column=0,row=18,sticky=tk.W,padx=5,pady=(10,0));
        encostentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["encostvar"]);
        encostentry.grid(column=0,row=19,sticky=tk.W,padx=5,pady=(5,10));
        
        
        ##Column 1
        
        
        lengthlabel = tk.Label(self, text='Rörlängd [m]');
        lengthlabel.grid(column=1,row=8,sticky=tk.W,padx=5,pady=(10,0));
        lengthentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["lengthvar"]);
        lengthentry.grid(column=1,row=9,sticky=tk.W,padx=5,pady=(5,10));
        
        scafflabel = tk.Label(self, text='Ställningskostnad [kr]');
        scafflabel.grid(column=1,row=0,sticky=tk.W,padx=5,pady=(10,0));
        scaffentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["scaffvar"]);
        scaffentry.grid(column=1,row=1,sticky=tk.W,padx=5,pady=(5,10));
        
        salwlabel = tk.Label(self, text='Lön svets [kr/h]');
        salwlabel.grid(column=1,row=2,sticky=tk.W,padx=5,pady=(10,0));
        salwentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["salwvar"]);
        salwentry.grid(column=1,row=3,sticky=tk.W,padx=5,pady=(5,10));
        
        salilabel = tk.Label(self, text='Lön isolering [kr/h]');
        salilabel.grid(column=1,row=4,sticky=tk.W,padx=5,pady=(10,0));
        salientry = tk.Entry(self, width=25, textvariable=self.master.shared_data["salivar"]);
        salientry.grid(column=1,row=5,sticky=tk.W,padx=5,pady=(5,10));
        
        salalabel = tk.Label(self, text='Lön isolerskal [kr/h]');
        salalabel.grid(column=1,row=6,sticky=tk.W,padx=5,pady=(10,0));
        salaentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["salavar"]);
        salaentry.grid(column=1,row=7,sticky=tk.W,padx=5,pady=(5,10));
        
        speedwlabel = tk.Label(self, text='Svetshastighet [m/s]');
        speedwlabel.grid(column=1,row=10,sticky=tk.W,padx=5,pady=(10,0));
        speedwentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["speedwvar"]);
        speedwentry.grid(column=1,row=11,sticky=tk.W,padx=5,pady=(5,10));
        
        spotswlabel = tk.Label(self, text='Mängden fogar [n]');
        spotswlabel.grid(column=1,row=12,sticky=tk.W,padx=5,pady=(10,0));
        spotswentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["spotswvar"]);
        spotswentry.grid(column=1,row=13,sticky=tk.W,padx=5,pady=(5,10));
        
        mthicclabel = tk.Label(self, text='Godstjocklek [m]');
        mthicclabel.grid(column=1,row=14,sticky=tk.W,padx=5,pady=(10,0));
        mthiccentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["mthiccvar"]);
        mthiccentry.grid(column=1,row=15,sticky=tk.W,padx=5,pady=(5,10));
        
        ithicclabel = tk.Label(self, text='Isolering tjocklek [m]');
        ithicclabel.grid(column=1,row=16,sticky=tk.W,padx=5,pady=(10,0));
        ithiccentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["ithiccvar"]);
        ithiccentry.grid(column=1,row=17,sticky=tk.W,padx=5,pady=(5,10));
        
        atimelabel = tk.Label(self, text='Isolerskalstid [h]');
        atimelabel.grid(column=1,row=18,sticky=tk.W,padx=5,pady=(10,0));
        atimeentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["atimevar"]);
        atimeentry.grid(column=1,row=19,sticky=tk.W,padx=5,pady=(5,10));
    
        
        
        
    
        ##Column2
    
        apricelabel = tk.Label(self, text='Isolerskalskostnad [kr/m²]');
        apricelabel.grid(column=2,row=0,sticky=tk.W,padx=5,pady=(10,0));
        apriceentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["apricevar"]);
        apriceentry.grid(column=2,row=1,sticky=tk.W,padx=5,pady=(5,10));
        
        ipricelabel = tk.Label(self, text='Isoleringskostnad [kr/m²]');
        ipricelabel.grid(column=2,row=2,sticky=tk.W,padx=5,pady=(10,0));
        ipriceentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["ipricevar"]);
        ipriceentry.grid(column=2,row=3,sticky=tk.W,padx=5,pady=(5,10));
        
        workefflabel = tk.Label(self, text='Arbetseffektivitet [0-1]');
        workefflabel.grid(column=2,row=4,sticky=tk.W,padx=5,pady=(10,0));
        workeffentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["workeffvar"]);
        workeffentry.grid(column=2,row=5,sticky=tk.W,padx=5,pady=(5,10));
        
        bendlabel = tk.Label(self, text='Mängdböjar [n] (antas 90 grader)');
        bendlabel.grid(column=2,row=6,sticky=tk.W,padx=5,pady=(10,0));
        bendentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["bendvar"]);
        bendentry.grid(column=2,row=7,sticky=tk.W,padx=5,pady=(5,10));
        
        roughlabel = tk.Label(self, text='Absolut ytojämnhet [m]');
        roughlabel.grid(column=2,row=8,sticky=tk.W,padx=5,pady=(10,0));
        roughentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["roughvar"]);
        roughentry.grid(column=2,row=9,sticky=tk.W,padx=5,pady=(5,10));
        
        rentlabel = tk.Label(self, text='Kalkylränta [%]');
        rentlabel.grid(column=2,row=10,sticky=tk.W,padx=5,pady=(10,0));
        rententry = tk.Entry(self, width=25, textvariable=self.master.shared_data["rentvar"]);
        rententry.grid(column=2,row=11,sticky=tk.W,padx=5,pady=(5,10));
        
        endevlabel = tk.Label(self, text='Elpris utveckling [%]');
        endevlabel.grid(column=2,row=12,sticky=tk.W,padx=5,pady=(10,0));
        endeventry = tk.Entry(self, width=25, textvariable=self.master.shared_data["endevvar"]);
        endeventry.grid(column=2,row=13,sticky=tk.W,padx=5,pady=(5,10));

        lifespanlabel = tk.Label(self, text='Livslängd [år]');
        lifespanlabel.grid(column=2,row=14,sticky=tk.W,padx=5,pady=(10,0));
        lifespanentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["lifespanvar"]);
        lifespanentry.grid(column=2,row=15,sticky=tk.W,padx=5,pady=(5,10));
        
        itimelabel = tk.Label(self, text='Isoleringstid [h]');
        itimelabel.grid(column=2,row=16,sticky=tk.W,padx=5,pady=(10,0));
        itimeentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["itimevar"]);
        itimeentry.grid(column=2,row=17,sticky=tk.W,padx=5,pady=(5,10));
        
    
    
class App(tk.Tk):
    def __init__(self):
        super().__init__();
       
        self.columnconfigure(0, weight=1);
        self.columnconfigure(1, weight=3);
        
        #Data that will be shared throughout the entire programm, from input window to calculations.
        self.shared_data ={
            "flowvar" : tk.DoubleVar(),
            "denvar" : tk.DoubleVar(),
            "dynvisvar" : tk.DoubleVar(),
            "minvvar" : tk.DoubleVar(),
            "maxvvar" : tk.DoubleVar(),
            "pumpeffvar" : tk.DoubleVar(),
            "potheadvar" : tk.DoubleVar(),
            "heightvar" : tk.DoubleVar(),
            "yearlyhvar" : tk.IntVar(),
            "encostvar" : tk.DoubleVar(),
            "lifespanvar" : tk.IntVar(),
            "lengthvar" : tk.DoubleVar(),
            "scaffvar" : tk.IntVar(),
            "salwvar" : tk.DoubleVar(),
            "salivar" : tk.DoubleVar(),
            "salavar" : tk.DoubleVar(),
            "speedwvar" : tk.DoubleVar(),
            "spotswvar" : tk.DoubleVar(),
            "mthiccvar" : tk.DoubleVar(),
            "ithiccvar" : tk.DoubleVar(),
            "atimevar" : tk.DoubleVar(),
            "itimevar" : tk.DoubleVar(),
            "apricevar" : tk.DoubleVar(),
            "ipricevar" : tk.DoubleVar(),
            "workeffvar" : tk.DoubleVar(),
            "bendvar" : tk.IntVar(),
            "roughvar" : tk.DoubleVar(),
            "rentvar" : tk.DoubleVar(),
            "endevvar" : tk.DoubleVar()
            }
        self.__set_default_values();
        self.__create_widgets();
    
    def __set_default_values(self):
        self.shared_data["denvar"].set(997);
        self.shared_data["dynvisvar"].set(0.001);
        self.shared_data["roughvar"].set(0.002);
        self.shared_data["pumpeffvar"].set(0.7);
        self.shared_data["encostvar"].set(0.5);
        self.shared_data["lifespanvar"].set(30);
        self.shared_data["mthiccvar"].set(0.005);
        self.shared_data["ithiccvar"].set(0.02);
        self.shared_data["apricevar"].set(10000);
        self.shared_data["ipricevar"].set(30000);
        self.shared_data["minvvar"].set(1);
        self.shared_data["maxvvar"].set(6);
        self.shared_data["yearlyhvar"].set(8520);
        self.shared_data["salavar"].set(300);
        self.shared_data["salivar"].set(300);
        self.shared_data["salwvar"].set(300);
        self.shared_data["speedwvar"].set(0.005);
        self.shared_data["flowvar"].set(300);
        self.shared_data["workeffvar"].set(0.65);
        self.shared_data["scaffvar"].set(150000);
        self.shared_data["spotswvar"].set(90);
        self.shared_data["bendvar"].set(12);
        self.shared_data["lengthvar"].set(70);
        self.shared_data["heightvar"].set(5);
        self.shared_data["potheadvar"].set(50);
        self.shared_data["itimevar"].set(30);
        self.shared_data["atimevar"].set(25);
        self.shared_data["salivar"].set(300);
        self.shared_data["rentvar"].set(0.11);
        self.shared_data["endevvar"].set(0.04);
    
    def __create_widgets(self):
        self.input_frame = InputFrame(self);
        self.input_frame.grid(column=0, row=0);
        
        self.output_frame = OutputFrame(self);
        self.output_frame.grid(column=1, row=0);
        
    def calculate(self):
        
        #set variables using input gui data(this can be reworked to use gui data directly but not worth the few bytes in ram usage it saves)
        q= self.shared_data["flowvar"].get();
        den = self.shared_data["denvar"].get();
        dyn_vis = self.shared_data["dynvisvar"].get();
        min_v = self.shared_data["minvvar"].get();
        max_v = self.shared_data["maxvvar"].get();
        pump_eff = self.shared_data["pumpeffvar"].get();
        pot_head = self.shared_data["potheadvar"].get();
        diff_head = self.shared_data["heightvar"].get();
        yearly_h = self.shared_data["yearlyhvar"].get();
        en_cost = self.shared_data["encostvar"].get();
        sys_length = self.shared_data["lengthvar"].get();
        lifespan = self.shared_data["lifespanvar"].get();
        scaff = self.shared_data["scaffvar"].get();
        sal_w = self.shared_data["salwvar"].get();
        sal_i = self.shared_data["salivar"].get();
        sal_a = self.shared_data["salavar"].get();
        speed_w = self.shared_data["speedwvar"].get();
        spots_w = self.shared_data["spotswvar"].get();
        thic_m = self.shared_data["mthiccvar"].get();
        thic_i = self.shared_data["ithiccvar"].get();
        time_i = self.shared_data["itimevar"].get();
        time_a = self.shared_data["atimevar"].get();
        price_i = self.shared_data["ipricevar"].get();
        price_a = self.shared_data["apricevar"].get();
        work_eff = self.shared_data["workeffvar"].get();
        bends = self.shared_data["bendvar"].get();
        rough = self.shared_data["roughvar"].get();
        rent = self.shared_data["rentvar"].get();
        endev = self.shared_data["endevvar"].get();
        
        tree = ET.parse(os.path.join(current_directory,'/apps/ror_dim.xml'));
        root = tree.getroot();

        #arrays for storing data from all the materials
        dim = [];
        mcost = [];
        vel = [];
        con_cost = [];
        total_cost = [];
        yearly_cost = [];
        energy_cost = [];
        energy_cost_final = [];
        functional = [];
        pump_kw = [];
        
        for child in root: #find all dims and their metercost from xml file
            dim.append(float(child.find('dim').text));
            mcost.append(float(child.find('mcost').text));


        n= 0; #keeps track of which dim we are on
        min_cost= None; 
        min_dim = None;
        
        for dimension in dim:
            #vel
            v = Calculations.velocity(dim[n],q);
            vel.append(v);
           # print("Velocity: " + str(v))
            if v<= max_v and v>= min_v:
                functional.append(True);
            else:
                functional.append(False);

            #head   
            re = Calculations.reynolds_number(v,den,sys_length,dyn_vis);
            if re >= 2300:
                f = Calculations.Mileikovskyi(re,rough/dim[n]);
            else:
                f = Calculations.laminar(re);

            head_bend = Calculations.bend_calc(dim[n],f,v,sys_length,bends,den);

           
    
            
            diff_head_loss = diff_head*den*9.81 #converting to pascal
            
            h_loss = diff_head_loss+head_bend; #real world height diff + friction loss + bend loss

            if h_loss > (pot_head*den*9.81) and functional[n] != False: #if the loss is greater than pump head then invalidate it if its not already invalidated
                functional[n] = False;
                
            pump_kw.append(Calculations.calc_en_cost(h_loss,q,pump_eff,en_cost,yearly_h,0,0)[1]);  
            
            yearly_energy_cost_rent = [];   
                
            yearly_cost.append(Calculations.calc_en_cost(h_loss,q,pump_eff,en_cost,yearly_h,0,0)[0]);
            con_cost.append(Calculations.calc_con_cost(mcost[n],dim[n],sys_length,spots_w,speed_w,sal_w,sal_i,sal_a,time_i,time_a,price_i,price_a,work_eff,scaff,thic_m,thic_i));
            
            if endev: #if we have increasing electricity prices we will do a different calculation of the yearly cost
                for year in range(lifespan+1):
                    yearly_energy_cost_rent.append(Calculations.calc_en_cost(h_loss,q,pump_eff,en_cost,yearly_h,year,endev)[0]/((1+rent)**year));
            else:
                for year in range(lifespan+1):
                    yearly_energy_cost_rent.append(yearly_cost[n]/((1+rent)**year));
                
            energy_cost.append(yearly_energy_cost_rent); #Puts a list of all the years and their energy cost into a list of all dims (2d matrix)
            energy_cost_final.append(sum(energy_cost[n][0:lifespan])); #Gives life cycle cost for energy by summing up all the years
            
            total_cost.append(energy_cost_final[n]+con_cost[n]);
            
           # print("Total cost for "  + str(dim[n]) + ": " + str(total_cost[n]));
            
            if functional[n] == True and (min_cost == None or total_cost[n] < min_cost): #if its functional and cheaper than the current cheapest it becomes our cheapest
                min_cost = total_cost[n];
                min_dim = dim[n];
    
            n+=1;
        
      
        
        dim_txt = []; #convert to text so that matplotlib does not interpret the dim as a value axis and scales it to that
        for val in dim:
            dim_txt.append(str(math.floor(val*1000)));
            
        valid_dim = [];
        valid_cost = [];
        valid_kw = [];
        col = [];
        x_time = []; # list of time lists will be (1ifespan x amount of valids) matrix
        y_cost = []; # same as above but energy cost for a certain year per valid
        nr=0; #what dim we are currently on
        j=0; # how many valids we have
        OutputFrame.print_result(self.output_frame,min_dim,min_cost);
        
        #export to excel
        df = DataFrame({'Dimension' : dim_txt, 'Totalcost' : total_cost, 'Functional' : functional, 'Construction cost' : con_cost, 'Energy cost' : energy_cost_final});
        df.to_excel('ror_dim.xlsx', sheet_name='sheet1', index=False);
      
        #go through and mark all the valid dims
        for valid in functional:
            if valid:
                col.append('green'); 
                valid_dim.append(dim_txt[nr]);
                valid_cost.append(total_cost[nr]/1000000);
                valid_kw.append(pump_kw[nr]);
                time =[];
                cost = [];
                for i in range(lifespan+1): #this is to calculate a cost for each year for the valid dims
                    time.append(i);
                    cost.append((sum(energy_cost[nr][0:i]) + con_cost[nr])/1000000);
                x_time.append(time);
                y_cost.append(cost);
                j +=1
            else:
                col.append('red');
            nr +=1;

        #matplot lib setup
        fig,ax = plt.subplots(nrows=2,ncols=3);
        
        nr=0;
        for cost in total_cost:
            total_cost[nr] = cost/1000000;
            nr+=1;
        
        #graph for all dimmensions
        ax[0,0].bar(dim_txt,total_cost, color=col);
        ax[0,0].set_xlabel("Rör dim mm");
        ax[0,0].set_ylabel("Livscykelkostnad MSek");
        ax[0,0].set_ylim(top=(min_cost/1000000)*4);
        

        #graph for only valid dimmensions
        ax[0,1].bar(valid_dim,valid_cost, color='green');
        ax[0,1].set_xlabel("Rör dim mm");
        ax[0,1].set_ylabel("Livscykelkostnad MSek");
        for i,v in enumerate(valid_cost):
            ax[0,1].text(i, 0.1, str(round(v,2)), fontsize=10,color="black",ha='center',va='center')

        nr=0;
        y1=[];
        y2=[];
        for cost in total_cost:
            y1.append(energy_cost_final[nr]/(total_cost[nr]*1000000));
            y2.append(con_cost[nr]/(total_cost[nr]*1000000));
            nr+=1;
        ax[0,2].bar(dim_txt,y1, color='orange', label="Energy");
        ax[0,2].bar(dim_txt,y2, bottom=y1, color='purple', label="Construction");
        ax[0,2].set_xlabel("Dim mm");
        ax[0,2].set_ylabel("Livscykelkostnad %");
        ax[0,2].legend();

        #plot one line for each valid dimension
        nr = 0
        for lists in x_time:
            ax[1,0].plot(lists,y_cost[nr], label=valid_dim[nr])
            nr +=1
        ax[1,0].legend();
        ax[1,0].set_ylim(bottom=0);
        ax[1,0].set_xlabel("tid år");
        ax[1,0].set_ylabel("Livscykelkostnad MSek");
        

        normalised_speeds=[];
        normalised_speeds_dims=[];
        nr = 0
        for v in vel:
            if v < max_v*2:
                normalised_speeds.append(v);
                normalised_speeds_dims.append(dim_txt[nr]);
            
            nr +=1


        ax[1,1].bar(normalised_speeds_dims,normalised_speeds)
        ax[1,1].set_xlabel("Rör dim mm");
        ax[1,1].set_ylabel("Mediahastighet m/s");
        for i,v in enumerate(normalised_speeds):
            ax[1,1].text(i, 0.5, str(round(v,2)), fontsize=10,color="black",ha='center',va='center')
            
        ax[1,2].bar(valid_dim,valid_kw);
        ax[1,2].set_xlabel("Rör dim mm");
        ax[1,2].set_ylabel("Energiförbrukning kW");
        for i,v in enumerate(valid_kw):
            ax[1,2].text(i,1, str(round(v,2)), fontsize=10,color="black",ha='center',va='center')
            
        plt.show();
  
        

if __name__=="__main__":
    app = App()
    app.mainloop()

```

# apps/Economics_Pipe_Cost_Optimization.py

```py
#import re
import xml.etree.ElementTree as ET
import math
import fluids
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import tkinter as tk
import os
import datetime

# Assuming current_directory is defined as:
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'results' directory
results_dir = os.path.join(current_directory, 'results')

# Check if 'results' directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Generate the filename based on the current date including seconds
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"Pipe_Cost_Results_{current_time}.xlsx"
file_path = os.path.join(results_dir, filename)

class Calculations:
    def velocity(d,q):
        return (q/3600)/(((d/2)**2)*math.pi);
        
    #friction coeffecient for laminar flow
    def laminar(re):
        f = 64/re;
        return f;

    def reynolds_number(v,den,dim,dyn_vis):
        rey = den*v*dim/dyn_vis;
        return rey;
        
    #calculate friction coeffecient for turbulent flow using Mileikovskyi method, no iteration needed
    def Mileikovskyi(re,rr): 
        A0 = -0.79638*math.log((rr/8.298)+(7.3357/re));
        A1 = re*rr+9.3120665*A0;
        f=((8.128943+A1)/(8.128943*A0-0.86859209*A1*math.log(A1/(3.7099535*re))))**2;
        return f;

    #use fluids to calculate headloss from bends
    def bend_calc(d,f,v,sys_length,bends,den):
        K = fluids.fittings.entrance_sharp();
        for n in range(math.floor(bends)):
            K += fluids.fittings.bend_rounded(Di=d,angle=90,fd=f);
        K += fluids.fittings.exit_normal();
        K += fluids.core.K_from_f(fd=f, L=sys_length, D=d)
        loss = fluids.core.dP_from_K(K=K,rho=den,V=v); # returns pressure loss in Pa
        return loss;

    #yearly energy cost
    def calc_en_cost(head,q,pump_eff,en_cost,yearly_h,year,endev):
        kw= q*head/(3599000*pump_eff);
       # print("kw: " + str(kw));
        return [en_cost*((1+endev)**year)*kw*yearly_h,kw];


    def calc_con_cost(mcost,d,sys_length,spots_w,speed_w,sal_w,sal_i,sal_a,time_i,time_a,price_i,price_a,work_eff,scaff,thic_m,thic_i):
        con_cost = scaff;
        con_cost += mcost * sys_length;
        con_cost += (d+2*thic_m)*math.pi*spots_w*speed_w*sal_w/work_eff;
        con_cost += time_i*sal_i/work_eff + (d+2*thic_m+2*thic_i)*math.pi*sys_length*price_i;
        con_cost += time_a*sal_a/work_eff + (d+2*thic_m+2*thic_i)*math.pi*sys_length*price_a;
        
        #print("con cost: " + str(con_cost))
        return con_cost

 


class OutputFrame(tk.Frame):
    def __init__(self,master):
        super().__init__();
        self.master = master;
        self.columnconfigure(0, weight=1);
        self.columnconfigure(1, weight=3);
        self.__create_widgets();
        
    def __create_widgets(self):
        self.resultbutton = tk.Button(self, text="Generate Results",width=30, height=5, command = self.result_button_click);
        self.resultbutton.grid(column=0, row=0, padx=5, pady=10);
        
        self.warninglabel = tk.Label(self, text="");
        self.warninglabel.grid(column=0, row=1, padx=5, pady=10);
        
        self.result_dim_label = tk.Label(self, text="", font=25);
        self.result_dim_label.grid(column=0, row=1, padx=5, pady=10);
        self.result_cost_label = tk.Label(self, text="", font=25);
        self.result_cost_label.grid(column=0, row=2, padx=5, pady=10);
        
        
    def result_button_click(self):
        #Check that all fields are filld in or it wont execute the calculations and instead give a red warning text
        failures = 0;
        for data in self.master.shared_data:
            if self.master.shared_data[data].get():
                pass
            #    print("success");
            elif data=="endevvar": #add in or data=="" for any other variable that should be allowed to be 0
                pass
            else:
                failures +=1;
              #  print("Missing: " + data);
                self.master.shared_data[data].set(0)
        if failures > 0:
             self.warninglabel['text'] = "Error, fill all fields!";
             self.warninglabel['fg'] = "red";
        else:
           self.master.calculate()
           
    def print_result(self, dim,cost):
        self.result_dim_label['text'] = "The cheapest is: " + str(1000*dim) + "mm";
        self.result_cost_label['text'] = "Price: " + '{:0,.2f}'.format(cost).replace(",","X").replace(".",",").replace("X"," ") + "$"
            
class InputFrame(tk.Frame):
    def __init__(self,master):
        super().__init__();
        
        self.columnconfigure(0, weight=1);
        self.columnconfigure(1, weight=3);
        self.columnconfigure(2, weight=5);
        self.__create_widgets();
          
    def __create_widgets(self):
        
        ##Column 0
        
        flowlabel = tk.Label(self, text='Flow [m³/h]');
        flowlabel.grid(column=0,row=0,sticky=tk.W,padx=5,pady=(10,0));
        flowentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["flowvar"]);
        flowentry.grid(column=0,row=1,sticky=tk.W,padx=5,pady=(5,10));
        
        label1 = tk.Label(self, text='Density [kg/m³]');
        label1.grid(column=0,row=2,sticky=tk.W,padx=5,pady=(10,0));
        entry1 = tk.Entry(self, width=25, textvariable=self.master.shared_data["denvar"]);
        entry1.grid(column=0,row=3,sticky=tk.W,padx=5,pady=(5,10));
        
        dynvislabel = tk.Label(self, text='Dynamic Viscosity [Pa*s]');
        dynvislabel.grid(column=0,row=4,sticky=tk.W,padx=5,pady=(10,0));
        dynvisentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["dynvisvar"]);
        dynvisentry.grid(column=0,row=5,sticky=tk.W,padx=5,pady=(5,10));
        
        minvlabel = tk.Label(self, text='Min velocity [m/s]');
        minvlabel.grid(column=0,row=6,sticky=tk.W,padx=5,pady=(10,0));
        minventry = tk.Entry(self, width=25, textvariable=self.master.shared_data["minvvar"]);
        minventry.grid(column=0,row=7,sticky=tk.W,padx=5,pady=(5,10));
        
        maxvlabel = tk.Label(self, text='Max velocity [m/s]');
        maxvlabel.grid(column=0,row=8,sticky=tk.W,padx=5,pady=(10,0));
        maxventry = tk.Entry(self, width=25, textvariable=self.master.shared_data["maxvvar"]);
        maxventry.grid(column=0,row=9,sticky=tk.W,padx=5,pady=(5,10));
        
        pumpefflabel = tk.Label(self, text='Pump Efficiency [0-1]');
        pumpefflabel.grid(column=0,row=10,sticky=tk.W,padx=5,pady=(10,0));
        pumpeffentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["pumpeffvar"]);
        pumpeffentry.grid(column=0,row=11,sticky=tk.W,padx=5,pady=(5,10));
        
        potheadlabel = tk.Label(self, text='Pump Head [m]');
        potheadlabel.grid(column=0,row=12,sticky=tk.W,padx=5,pady=(10,0));
        potheadentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["potheadvar"]);
        potheadentry.grid(column=0,row=13,sticky=tk.W,padx=5,pady=(5,10));
        
        heightlabel = tk.Label(self, text='Height Difference [m]');
        heightlabel.grid(column=0,row=14,sticky=tk.W,padx=5,pady=(10,0));
        heightentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["heightvar"]);
        heightentry.grid(column=0,row=15,sticky=tk.W,padx=5,pady=(5,10));
        
        yearlylabel = tk.Label(self, text='Operational Hours [h]');
        yearlylabel.grid(column=0,row=16,sticky=tk.W,padx=5,pady=(10,0));
        yearlyentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["yearlyhvar"]);
        yearlyentry.grid(column=0,row=17,sticky=tk.W,padx=5,pady=(5,10));
        
        encostlabel = tk.Label(self, text='Electric Cost [$/kwh]');
        encostlabel.grid(column=0,row=18,sticky=tk.W,padx=5,pady=(10,0));
        encostentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["encostvar"]);
        encostentry.grid(column=0,row=19,sticky=tk.W,padx=5,pady=(5,10));
        
        
        ##Column 1
        
        
        lengthlabel = tk.Label(self, text='Pipe Length [m]');
        lengthlabel.grid(column=1,row=8,sticky=tk.W,padx=5,pady=(10,0));
        lengthentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["lengthvar"]);
        lengthentry.grid(column=1,row=9,sticky=tk.W,padx=5,pady=(5,10));
        
        scafflabel = tk.Label(self, text='Skaffolding Cost [$]');
        scafflabel.grid(column=1,row=0,sticky=tk.W,padx=5,pady=(10,0));
        scaffentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["scaffvar"]);
        scaffentry.grid(column=1,row=1,sticky=tk.W,padx=5,pady=(5,10));
        
        salwlabel = tk.Label(self, text='Welder Salary [$/h]');
        salwlabel.grid(column=1,row=2,sticky=tk.W,padx=5,pady=(10,0));
        salwentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["salwvar"]);
        salwentry.grid(column=1,row=3,sticky=tk.W,padx=5,pady=(5,10));
        
        salilabel = tk.Label(self, text='Pipe Insulator Salary [$/h]');
        salilabel.grid(column=1,row=4,sticky=tk.W,padx=5,pady=(10,0));
        salientry = tk.Entry(self, width=25, textvariable=self.master.shared_data["salivar"]);
        salientry.grid(column=1,row=5,sticky=tk.W,padx=5,pady=(5,10));
        
        salalabel = tk.Label(self, text='Pipe Insulator Shell Salary [$/h]');
        salalabel.grid(column=1,row=6,sticky=tk.W,padx=5,pady=(10,0));
        salaentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["salavar"]);
        salaentry.grid(column=1,row=7,sticky=tk.W,padx=5,pady=(5,10));
        
        speedwlabel = tk.Label(self, text='Welding speed [m/s]');
        speedwlabel.grid(column=1,row=10,sticky=tk.W,padx=5,pady=(10,0));
        speedwentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["speedwvar"]);
        speedwentry.grid(column=1,row=11,sticky=tk.W,padx=5,pady=(5,10));
        
        spotswlabel = tk.Label(self, text='Number of Weld joints [n]');
        spotswlabel.grid(column=1,row=12,sticky=tk.W,padx=5,pady=(10,0));
        spotswentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["spotswvar"]);
        spotswentry.grid(column=1,row=13,sticky=tk.W,padx=5,pady=(5,10));
        
        mthicclabel = tk.Label(self, text='Material Thickness [m]');
        mthicclabel.grid(column=1,row=14,sticky=tk.W,padx=5,pady=(10,0));
        mthiccentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["mthiccvar"]);
        mthiccentry.grid(column=1,row=15,sticky=tk.W,padx=5,pady=(5,10));
        
        ithicclabel = tk.Label(self, text='Insulation Thickness [m]');
        ithicclabel.grid(column=1,row=16,sticky=tk.W,padx=5,pady=(10,0));
        ithiccentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["ithiccvar"]);
        ithiccentry.grid(column=1,row=17,sticky=tk.W,padx=5,pady=(5,10));
        
        atimelabel = tk.Label(self, text='Insulation Shelling Time [h]');
        atimelabel.grid(column=1,row=18,sticky=tk.W,padx=5,pady=(10,0));
        atimeentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["atimevar"]);
        atimeentry.grid(column=1,row=19,sticky=tk.W,padx=5,pady=(5,10));
    
        
        
        
    
        ##Column2
    
        apricelabel = tk.Label(self, text='Price of Insulation Shell [$/m²]');
        apricelabel.grid(column=2,row=0,sticky=tk.W,padx=5,pady=(10,0));
        apriceentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["apricevar"]);
        apriceentry.grid(column=2,row=1,sticky=tk.W,padx=5,pady=(5,10));
        
        ipricelabel = tk.Label(self, text='Price of Insulation [$/m²]');
        ipricelabel.grid(column=2,row=2,sticky=tk.W,padx=5,pady=(10,0));
        ipriceentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["ipricevar"]);
        ipriceentry.grid(column=2,row=3,sticky=tk.W,padx=5,pady=(5,10));
        
        workefflabel = tk.Label(self, text='Work Efficiency [0-1]');
        workefflabel.grid(column=2,row=4,sticky=tk.W,padx=5,pady=(10,0));
        workeffentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["workeffvar"]);
        workeffentry.grid(column=2,row=5,sticky=tk.W,padx=5,pady=(5,10));
        
        bendlabel = tk.Label(self, text='90 bends [n] (antas 90 grader)');
        bendlabel.grid(column=2,row=6,sticky=tk.W,padx=5,pady=(10,0));
        bendentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["bendvar"]);
        bendentry.grid(column=2,row=7,sticky=tk.W,padx=5,pady=(5,10));
        
        roughlabel = tk.Label(self, text='Absolute Roughness [m]');
        roughlabel.grid(column=2,row=8,sticky=tk.W,padx=5,pady=(10,0));
        roughentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["roughvar"]);
        roughentry.grid(column=2,row=9,sticky=tk.W,padx=5,pady=(5,10));
        
        rentlabel = tk.Label(self, text='Interest Rate [%]');
        rentlabel.grid(column=2,row=10,sticky=tk.W,padx=5,pady=(10,0));
        rententry = tk.Entry(self, width=25, textvariable=self.master.shared_data["rentvar"]);
        rententry.grid(column=2,row=11,sticky=tk.W,padx=5,pady=(5,10));
        
        endevlabel = tk.Label(self, text='Price development electricity [%]');
        endevlabel.grid(column=2,row=12,sticky=tk.W,padx=5,pady=(10,0));
        endeventry = tk.Entry(self, width=25, textvariable=self.master.shared_data["endevvar"]);
        endeventry.grid(column=2,row=13,sticky=tk.W,padx=5,pady=(5,10));

        lifespanlabel = tk.Label(self, text='Life Cycle [years]');
        lifespanlabel.grid(column=2,row=14,sticky=tk.W,padx=5,pady=(10,0));
        lifespanentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["lifespanvar"]);
        lifespanentry.grid(column=2,row=15,sticky=tk.W,padx=5,pady=(5,10));
        
        itimelabel = tk.Label(self, text='Isulation Time [h]');
        itimelabel.grid(column=2,row=16,sticky=tk.W,padx=5,pady=(10,0));
        itimeentry = tk.Entry(self, width=25, textvariable=self.master.shared_data["itimevar"]);
        itimeentry.grid(column=2,row=17,sticky=tk.W,padx=5,pady=(5,10));
        
    
    
class App(tk.Tk):
    def __init__(self):
        super().__init__();
       
        self.columnconfigure(0, weight=1);
        self.columnconfigure(1, weight=3);
        
        #Data that will be shared throughout the entire programm, from input window to calculations.
        self.shared_data ={
            "flowvar" : tk.DoubleVar(),
            "denvar" : tk.DoubleVar(),
            "dynvisvar" : tk.DoubleVar(),
            "minvvar" : tk.DoubleVar(),
            "maxvvar" : tk.DoubleVar(),
            "pumpeffvar" : tk.DoubleVar(),
            "potheadvar" : tk.DoubleVar(),
            "heightvar" : tk.DoubleVar(),
            "yearlyhvar" : tk.IntVar(),
            "encostvar" : tk.DoubleVar(),
            "lifespanvar" : tk.IntVar(),
            "lengthvar" : tk.DoubleVar(),
            "scaffvar" : tk.IntVar(),
            "salwvar" : tk.DoubleVar(),
            "salivar" : tk.DoubleVar(),
            "salavar" : tk.DoubleVar(),
            "speedwvar" : tk.DoubleVar(),
            "spotswvar" : tk.DoubleVar(),
            "mthiccvar" : tk.DoubleVar(),
            "ithiccvar" : tk.DoubleVar(),
            "atimevar" : tk.DoubleVar(),
            "itimevar" : tk.DoubleVar(),
            "apricevar" : tk.DoubleVar(),
            "ipricevar" : tk.DoubleVar(),
            "workeffvar" : tk.DoubleVar(),
            "bendvar" : tk.IntVar(),
            "roughvar" : tk.DoubleVar(),
            "rentvar" : tk.DoubleVar(),
            "endevvar" : tk.DoubleVar()
            }
        self.__set_default_values();
        self.__create_widgets();
    
    def __set_default_values(self):
        self.shared_data["denvar"].set(997);
        self.shared_data["dynvisvar"].set(0.001);
        self.shared_data["roughvar"].set(0.002);
        self.shared_data["pumpeffvar"].set(0.7);
        self.shared_data["encostvar"].set(0.5);
        self.shared_data["lifespanvar"].set(30);
        self.shared_data["mthiccvar"].set(0.005);
        self.shared_data["ithiccvar"].set(0.02);
        self.shared_data["apricevar"].set(10000);
        self.shared_data["ipricevar"].set(30000);
        self.shared_data["minvvar"].set(1);
        self.shared_data["maxvvar"].set(6);
        self.shared_data["yearlyhvar"].set(8520);
        self.shared_data["salavar"].set(300);
        self.shared_data["salivar"].set(300);
        self.shared_data["salwvar"].set(300);
        self.shared_data["speedwvar"].set(0.005);
        self.shared_data["flowvar"].set(300);
        self.shared_data["workeffvar"].set(0.65);
        self.shared_data["scaffvar"].set(150000);
        self.shared_data["spotswvar"].set(90);
        self.shared_data["bendvar"].set(12);
        self.shared_data["lengthvar"].set(70);
        self.shared_data["heightvar"].set(5);
        self.shared_data["potheadvar"].set(50);
        self.shared_data["itimevar"].set(30);
        self.shared_data["atimevar"].set(25);
        self.shared_data["salivar"].set(300);
        self.shared_data["rentvar"].set(0.11);
        self.shared_data["endevvar"].set(0.04);
    
    def __create_widgets(self):
        self.input_frame = InputFrame(self);
        self.input_frame.grid(column=0, row=0);
        
        self.output_frame = OutputFrame(self);
        self.output_frame.grid(column=1, row=0);
        
    def calculate(self):
        
        #set variables using input gui data(this can be reworked to use gui data directly but not worth the few bytes in ram usage it saves)
        q= self.shared_data["flowvar"].get();
        den = self.shared_data["denvar"].get();
        dyn_vis = self.shared_data["dynvisvar"].get();
        min_v = self.shared_data["minvvar"].get();
        max_v = self.shared_data["maxvvar"].get();
        pump_eff = self.shared_data["pumpeffvar"].get();
        pot_head = self.shared_data["potheadvar"].get();
        diff_head = self.shared_data["heightvar"].get();
        yearly_h = self.shared_data["yearlyhvar"].get();
        en_cost = self.shared_data["encostvar"].get();
        sys_length = self.shared_data["lengthvar"].get();
        lifespan = self.shared_data["lifespanvar"].get();
        scaff = self.shared_data["scaffvar"].get();
        sal_w = self.shared_data["salwvar"].get();
        sal_i = self.shared_data["salivar"].get();
        sal_a = self.shared_data["salavar"].get();
        speed_w = self.shared_data["speedwvar"].get();
        spots_w = self.shared_data["spotswvar"].get();
        thic_m = self.shared_data["mthiccvar"].get();
        thic_i = self.shared_data["ithiccvar"].get();
        time_i = self.shared_data["itimevar"].get();
        time_a = self.shared_data["atimevar"].get();
        price_i = self.shared_data["ipricevar"].get();
        price_a = self.shared_data["apricevar"].get();
        work_eff = self.shared_data["workeffvar"].get();
        bends = self.shared_data["bendvar"].get();
        rough = self.shared_data["roughvar"].get();
        rent = self.shared_data["rentvar"].get();
        endev = self.shared_data["endevvar"].get();
        
        tree = ET.parse(os.path.join(current_directory,'pipe_cost_optimization_defaults.xml'));
        root = tree.getroot();

        #arrays for storing data from all the materials
        dim = [];
        mcost = [];
        vel = [];
        con_cost = [];
        total_cost = [];
        yearly_cost = [];
        energy_cost = [];
        energy_cost_final = [];
        functional = [];
        pump_kw = [];
        
        for child in root: #find all dims and their metercost from xml file
            dim.append(float(child.find('dim').text));
            mcost.append(float(child.find('mcost').text));


        n= 0; #keeps track of which dim we are on
        min_cost= None; 
        min_dim = None;
        
        for dimension in dim:
            #vel
            v = Calculations.velocity(dim[n],q);
            vel.append(v);
           # print("Velocity: " + str(v))
            if v<= max_v and v>= min_v:
                functional.append(True);
            else:
                functional.append(False);

            #head   
            re = Calculations.reynolds_number(v,den,sys_length,dyn_vis);
            if re >= 2300:
                f = Calculations.Mileikovskyi(re,rough/dim[n]);
            else:
                f = Calculations.laminar(re);

            head_bend = Calculations.bend_calc(dim[n],f,v,sys_length,bends,den);

           
    
            
            diff_head_loss = diff_head*den*9.81 #converting to pascal
            
            h_loss = diff_head_loss+head_bend; #real world height diff + friction loss + bend loss

            if h_loss > (pot_head*den*9.81) and functional[n] != False: #if the loss is greater than pump head then invalidate it if its not already invalidated
                functional[n] = False;
                
            pump_kw.append(Calculations.calc_en_cost(h_loss,q,pump_eff,en_cost,yearly_h,0,0)[1]);  
            
            yearly_energy_cost_rent = [];   
                
            yearly_cost.append(Calculations.calc_en_cost(h_loss,q,pump_eff,en_cost,yearly_h,0,0)[0]);
            con_cost.append(Calculations.calc_con_cost(mcost[n],dim[n],sys_length,spots_w,speed_w,sal_w,sal_i,sal_a,time_i,time_a,price_i,price_a,work_eff,scaff,thic_m,thic_i));
            
            if endev: #if we have increasing electricity prices we will do a different calculation of the yearly cost
                for year in range(lifespan+1):
                    yearly_energy_cost_rent.append(Calculations.calc_en_cost(h_loss,q,pump_eff,en_cost,yearly_h,year,endev)[0]/((1+rent)**year));
            else:
                for year in range(lifespan+1):
                    yearly_energy_cost_rent.append(yearly_cost[n]/((1+rent)**year));
                
            energy_cost.append(yearly_energy_cost_rent); #Puts a list of all the years and their energy cost into a list of all dims (2d matrix)
            energy_cost_final.append(sum(energy_cost[n][0:lifespan])); #Gives life cycle cost for energy by summing up all the years
            
            total_cost.append(energy_cost_final[n]+con_cost[n]);
            
           # print("Total cost for "  + str(dim[n]) + ": " + str(total_cost[n]));
            
            if functional[n] == True and (min_cost == None or total_cost[n] < min_cost): #if its functional and cheaper than the current cheapest it becomes our cheapest
                min_cost = total_cost[n];
                min_dim = dim[n];
    
            n+=1;
        
      
        
        dim_txt = []; #convert to text so that matplotlib does not interpret the dim as a value axis and scales it to that
        for val in dim:
            dim_txt.append(str(math.floor(val*1000)));
            
        valid_dim = [];
        valid_cost = [];
        valid_kw = [];
        col = [];
        x_time = []; # list of time lists will be (1ifespan x amount of valids) matrix
        y_cost = []; # same as above but energy cost for a certain year per valid
        nr=0; #what dim we are currently on
        j=0; # how many valids we have
        OutputFrame.print_result(self.output_frame,min_dim,min_cost);
        
        #export to excel
        df = DataFrame({'Dimension' : dim_txt, 'Totalcost' : total_cost, 'Functional' : functional, 'Construction cost' : con_cost, 'Energy cost' : energy_cost_final});
        df.to_excel(file_path, sheet_name='sheet1', index=False)
      
        #go through and mark all the valid dims
        for valid in functional:
            if valid:
                col.append('green'); 
                valid_dim.append(dim_txt[nr]);
                valid_cost.append(total_cost[nr]/1000000);
                valid_kw.append(pump_kw[nr]);
                time =[];
                cost = [];
                for i in range(lifespan+1): #this is to calculate a cost for each year for the valid dims
                    time.append(i);
                    cost.append((sum(energy_cost[nr][0:i]) + con_cost[nr])/1000000);
                x_time.append(time);
                y_cost.append(cost);
                j +=1
            else:
                col.append('red');
            nr +=1;

        #matplot lib setup
        fig,ax = plt.subplots(nrows=2,ncols=3);
        
        nr=0;
        for cost in total_cost:
            total_cost[nr] = cost/1000000;
            nr+=1;
        
        #graph for all dimmensions
        ax[0,0].bar(dim_txt,total_cost, color=col);
        ax[0,0].set_xlabel("Pipe dim mm");
        ax[0,0].set_ylabel("Life Cycle Cost mil$");
        ax[0,0].set_ylim(top=(min_cost/1000000)*4);
        

        #graph for only valid dimmensions
        ax[0,1].bar(valid_dim,valid_cost, color='green');
        ax[0,1].set_xlabel("Pipe Dimension mm");
        ax[0,1].set_ylabel("Life Cycle Cost mil$");
        for i,v in enumerate(valid_cost):
            ax[0,1].text(i, 0.1, str(round(v,2)), fontsize=10,color="black",ha='center',va='center')

        nr=0;
        y1=[];
        y2=[];
        for cost in total_cost:
            y1.append(energy_cost_final[nr]/(total_cost[nr]*1000000));
            y2.append(con_cost[nr]/(total_cost[nr]*1000000));
            nr+=1;
        ax[0,2].bar(dim_txt,y1, color='orange', label="Energy");
        ax[0,2].bar(dim_txt,y2, bottom=y1, color='purple', label="Construction");
        ax[0,2].set_xlabel("Dim mm");
        ax[0,2].set_ylabel("Life Cycle Cost %");
        ax[0,2].legend();

        #plot one line for each valid dimension
        nr = 0
        for lists in x_time:
            ax[1,0].plot(lists,y_cost[nr], label=valid_dim[nr])
            nr +=1
        ax[1,0].legend();
        ax[1,0].set_ylim(bottom=0);
        ax[1,0].set_xlabel("Time [years]");
        ax[1,0].set_ylabel("Life Cycle Cost mil$");
        

        normalised_speeds=[];
        normalised_speeds_dims=[];
        nr = 0
        for v in vel:
            if v < max_v*2:
                normalised_speeds.append(v);
                normalised_speeds_dims.append(dim_txt[nr]);
            
            nr +=1


        ax[1,1].bar(normalised_speeds_dims,normalised_speeds)
        ax[1,1].set_xlabel("Pipe dim mm");
        ax[1,1].set_ylabel("Average Media Velocity m/s");
        for i,v in enumerate(normalised_speeds):
            ax[1,1].text(i, 0.5, str(round(v,2)), fontsize=10,color="black",ha='center',va='center')
            
        ax[1,2].bar(valid_dim,valid_kw);
        ax[1,2].set_xlabel("Pipe dim mm");
        ax[1,2].set_ylabel("Energy Consumption kW");
        for i,v in enumerate(valid_kw):
            ax[1,2].text(i,1, str(round(v,2)), fontsize=10,color="black",ha='center',va='center')
            
        plt.show();
  
        

if __name__=="__main__":
    app = App()
    app.mainloop()

```

# apps/Environmental_Impact.py

```py
import tkinter as tk
from tkinter import ttk
# Assuming you have the chemicals library installed
import chemicals.environment

def retrieve_data():
    CASRN_value = CASRN_entry.get()

    GWP_methods = chemicals.environment.GWP_methods(CASRN_value)
    GWP_values = {method: chemicals.environment.GWP(CASRN_value, method=method) for method in GWP_methods}

    ODP_methods = chemicals.environment.ODP_methods(CASRN_value)
    ODP_values = {method: chemicals.environment.ODP(CASRN_value, method=method) for method in ODP_methods}

    logP_methods = chemicals.environment.logP_methods(CASRN_value)
    logP_values = {method: chemicals.environment.logP(CASRN_value, method=method) for method in logP_methods}

    result_text.set("Results for CASRN {}: \n\n".format(CASRN_value))
    result_text.set(result_text.get() + "GWP values:\n")
    for method, value in GWP_values.items():
        result_text.set(result_text.get() + "{}: {}\n".format(method, value))
    
    result_text.set(result_text.get() + "\nODP values:\n")
    for method, value in ODP_values.items():
        result_text.set(result_text.get() + "{}: {}\n".format(method, value))
    
    result_text.set(result_text.get() + "\nlogP values:\n")
    for method, value in logP_values.items():
        result_text.set(result_text.get() + "{}: {}\n".format(method, value))

app = tk.Tk()
app.title("Environmental Properties Retriever")

frame = ttk.Frame(app, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

CASRN_label = ttk.Label(frame, text="Enter CASRN:")
CASRN_label.grid(row=0, column=0, sticky=tk.W, pady=5)

CASRN_entry = ttk.Entry(frame, width=20)
CASRN_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

submit_button = ttk.Button(frame, text="Retrieve Data", command=retrieve_data)
submit_button.grid(row=1, column=0, columnspan=2, pady=10)

result_label = ttk.Label(frame, text="Results:")
result_label.grid(row=2, column=0, sticky=tk.W, pady=5)

result_text = tk.StringVar()
result_display = ttk.Label(frame, textvariable=result_text)
result_display.grid(row=3, column=0, columnspan=2, sticky=tk.W)

app.mainloop()

```

# apps/gas_solvers/solve_BWR.py

```py
import numpy as np
from scipy.optimize import root

def BWR_EoS(V, T, P, Tc, omega, a, b):
    """
    Calculates the molar volume for a mixture of components using the Benedict-Webb-Rubin (BWR) Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): BWR equation of state parameter of components.
    b (array): BWR equation of state parameter of components.

    Returns:
    float: Residual function for the BWR equation of state.
    """
    def f_BWR(V, T, Tc, omega, a, b):
        """
        Benedict-Webb-Rubin equation of state for a single component.
        """
        Tr = T / Tc
        return (a / V**2) * (1 + (2 * omega - 1) * np.sqrt(Tr) + (1 - omega) * Tr) - P / (R * T)

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_BWR(V, T, Tc[i], omega[i], a[i], b[i])
    return np.sum(Z)

def BWR_EoS_solver(T, P, Tc, omega, a, b):
    """
    Solves the Benedict-Webb-Rubin (BWR) Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): BWR equation of state parameter of components.
    b (array): BWR equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the BWR equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(BWR_EoS, V0, args=(T, P, Tc, omega, a, b)).x[0]
    return V
"""
This code defines two functions BWR_EoS and BWR_EoS_solver.
The BWR_EoS function calculates the residual function for the BWR equation of state,
while the BWR_EoS_solver function solves the BWR equation of state for a mixture of components.
 The input parameters for the BWR_EoS_solver function include the temperature, pressure,
critical temperature, acentric factor, and BWR equation of state parameters for each component.
The function returns the molar volume that satisfies the BWR equation of state
for the given conditions.
To use the `BWR_EoS_solver` function,
provide the necessary input parameters and call the function.
The function will return the molar volume that satisfies the BWR equation of state
for the given conditions.
"""

```

# apps/gas_solvers/solve_CubicExcessVolume.py

```py
import numpy as np
from scipy.optimize import root

def cubic_EoS_excess_volume(V, T, P, Tc, omega, a, b, k, d):
    """
    Calculates the molar volume for a mixture of components using a Cubic Equation of State with Excess Volume.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): Cubic equation of state parameter of components.
    b (array): Cubic equation of state parameter of components.
    k (array): Excess volume parameter of components.
    d (array): Excess volume parameter of components.

    Returns:
    float: Residual function for the cubic equation of state with excess volume.
    """
    def f_cubic_EoS_excess_volume(V, T, Tc, omega, a, b, k, d):
        """
        Cubic equation of state with excess volume for a single component.
        """
        Tr = T / Tc
        return (a / V**2) * (1 + (k + d * Tr) * (V / b - 1)) - P / (R * T)

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_cubic_EoS_excess_volume(V, T, Tc[i], omega[i], a[i], b[i], k[i], d[i])
    return np.sum(Z)

def cubic_EoS_excess_volume_solver(T, P, Tc, omega, a, b, k, d):
    """
    Solves a Cubic Equation of State with Excess Volume for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): Cubic equation of state parameter of components.
    b (array): Cubic equation of state parameter of components.
    k (array): Excess volume parameter of components.
    d (array): Excess volume parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the cubic equation of state with excess volume for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(cubic_EoS_excess_volume, V0, args=(T, P, Tc, omega, a, b, k, d)).x[0]
    return V
"""
This code defines two functions cubic_EoS_excess_volume and cubic_EoS_excess_volume_solver.
 The cubic_EoS_excess_volume function calculates the residual
function for the cubic equation of state with excess volume, 
while the cubic_EoS_excess_volume_solver function solves the cubic equation of state
 with excess volume for a mixture of components. 
The input parameters for the cubic_EoS_excess_volume_solver function include the temperature,
 pressure, critical temperature, acentric factor, cubic equation of state parameters,
 and excess volume parameters for each component. 
The function returns the molar volume that satisfies the cubic equation of state
 with excess volume for the given conditions.

To use the code, you would provide the necessary input parameters 
for the cubic_EoS_excess_volume_solver function and call the function. 
The function will then return the molar volume that satisfies the cubic equation of state
 with excess volume for the given conditions. 
It is important to note that the cubic equation of state parameters a, b, k, and d
 need to be calculated beforehand using methods such as 
regression analysis or the modified corresponding states (MCS) method.
"""


```

# apps/gas_solvers/solve_CubicPlusAssociation.py

```py
import numpy as np
from scipy.optimize import root

def CPA_EoS(V, T, P, a, b, k_ij):
    """
    Calculates the molar volume for a mixture of components using the Cubic-Plus-Association Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    a (array): Cubic equation of state parameter of components.
    b (array): Cubic equation of state parameter of components.
    k_ij (matrix): Association parameter of components.

    Returns:
    float: Residual function for the Cubic-Plus-Association equation of state.
    """
    def f_CPA_EoS(V, T, a, b, k_ij):
        """
        Cubic-Plus-Association equation of state for a single component.
        """
        N = len(a)
        f_CPA = np.zeros(N)
        for i in range(N):
            for j in range(i, N):
                f_CPA[i] += k_ij[i][j] * V**(1/3) * (V**(1/3) - b[j])**2
        return (R * T) / V - P + np.sum(a * V**(2/3) / b**2 - 2 * f_CPA)

    N = len(a)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_CPA_EoS(V, T, a[i], b[i], k_ij)
    return np.sum(Z)

def CPA_EoS_solver(T, P, a, b, k_ij):
    """
    Solves the Cubic-Plus-Association Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    a (array): Cubic equation of state parameter of components.
    b (array): Cubic equation of state parameter of components.
    k_ij (matrix): Association parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the Cubic-Plus-Association equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(CPA_EoS, V0, args=(T, P, a, b, k_ij)).x[0]
    return V
"""
This code defines two functions CPA_EoS and CPA_EoS_solver. The CPA_EoS function calculates the residual function for the Cubic-Plus-Association equation of state, while the CPA_EoS_solver function solves the Cubic-Plus-Association equation of state for a mixture of components. The input parameters for the CPA_EoS_solver function include the temperature, pressure, cubic
equation of state parameters a and b, and association parameter k_ij for each component. The function returns the molar volume that satisfies the Cubic-Plus-Association equation of state for the given conditions.

In this code, the root function from the scipy.optimize module is used to solve for the molar volume that satisfies the Cubic-Plus-Association equation of state. The root function uses a numerical method to find the root of a function, in this case, the residual function for the Cubic-Plus-Association equation of state. The V0 value is set as the initial estimate for the molar volume and is calculated as the molar volume that would correspond to an ideal gas at the given temperature and pressure. The V value returned by the root function is the molar volume that satisfies the Cubic-Plus-Association equation of state for the given conditions.

It is important to note that the parameters a, b, and k_ij in the Cubic-Plus-Association equation of state need to be calculated beforehand using methods such as regression analysis or experimental data. Additionally, the Cubic-Plus-Association equation of state is an empirical model that has been developed for specific types of systems and may not provide accurate results for all systems.
"""

```

# apps/gas_solvers/solve_GeneralizedCubic.py

```py
import numpy as np
from scipy.optimize import root

def GC_EoS(V, T, P, a, b, k):
    """
    Calculates the molar volume for a mixture of components using the Generalized Cubic Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    a (array): Cubic equation of state parameter of components.
    b (array): Cubic equation of state parameter of components.
    k (array): Generalized cubic equation of state parameter of components.

    Returns:
    float: Residual function for the Generalized Cubic equation of state.
    """
    N = len(a)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = (R * T) / V - P + a[i] * (1 - np.exp(-k[i] * (V / b[i] - 1))) / (k[i] * V / b[i])
    return np.sum(Z)

def GC_EoS_solver(T, P, a, b, k):
    """
    Solves the Generalized Cubic Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    a (array): Cubic equation of state parameter of components.
    b (array): Cubic equation of state parameter of components.
    k (array): Generalized cubic equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the Generalized Cubic equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(GC_EoS, V0, args=(T, P, a, b, k)).x[0]
    return V
"""
This code defines two functions GC_EoS and GC_EoS_solver. The GC_EoS function calculates the residual function for the Generalized Cubic equation of state, while the GC_EoS_solver function solves the Generalized Cubic equation of state for a mixture of components. The input parameters for the GC_EoS_solver function include the temperature, pressure, cubic equation of state parameters a and b, and the generalized cubic equation of state parameter k for each component. The function returns the molar volume that satisfies the Generalized Cubic equation of state for the given conditions.

In this code, the root function from the scipy.optimize module is used to solve for the molar volume that satisfies the Generalized Cubic equation of state. The root function uses a numerical method to find the root of a function, in this case, the residual function for the Generalized Cubic equation of state. The V0 value is set as the initial estimate for the molar volume and is calculated as the molar volume that would correspond to an ideal gas at the given temperature and pressure. The V value returned by the `root
"""


```

# apps/gas_solvers/solve_peng_robinson_iter.py

```py
import numpy as np

def peng_robinson(P, T, z, w, Tr, Pr):
    # Constants
    R = 8.314 # J/mol*K
    a = (0.45724 * R**2 * Tr**2) / Pr
    b = 0.0778 * R * Tr / Pr

    # Cubic equation coefficients
    A = a * P / (R * T)**2
    B = b * P / (R * T)

    # Cubic equation
    coeffs = [1, -1, -A + B - B**2, -A * B + B**2]
    roots = np.roots(coeffs)

    # Find the real root
    for root in roots:
        if np.isreal(root):
            v = np.real(root)
            break

    # Ideal gas contribution
    alpha = (1 + (0.37464 + 1.54226 * w - 0.26992 * w**2) * (1 - (T / Tr)**0.5))**2
    ideal = z - 1 - np.log(z - B)

    # Residual contribution
    residual = -(2 * np.sqrt(a * B) / (R * T)) * np.log((v + (1 + np.sqrt(2)) * B) / (v + (1 - np.sqrt(2)) * B))

    return ideal + residual * alpha

def solve_peng_robinson(P, T, z, w, Tr, Pr, tolerance=1e-6):
    # Initial estimate for v
    v = 0.001

    # Iterate until the desired tolerance is reached
    while True:
        f = peng_robinson(P, T, z, w, Tr, Pr)
        df = (peng_robinson(P, T, z, w, Tr, Pr, v + tolerance) - f) / tolerance
        v_new = v - f / df

        if abs(v_new - v) < tolerance:
            break

        v = v_new

    return v

```

# apps/gas_solvers/solve_peng_robinson.py

```py
import numpy as np
from scipy.optimize import root

def PR_EoS(V, T, P, Tc, omega, k, a, b):
    """
    Calculates the molar volume for a mixture of components using the Peng-Robinson (PR) Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): PR equation of state parameter of components.
    a (array): PR equation of state parameter of components.
    b (array): PR equation of state parameter of components.

    Returns:
    float: Residual function for the PR equation of state.
    """
    def f_PR(V, T, Tc, omega, k, a, b):
        """
        Peng-Robinson equation of state for a single component.
        """
        Tr = T / Tc
        return (a / V**2) * (1 - k * (1 - np.sqrt(Tr))) - P / (R * T) + (b * P) / (V * R * T)

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_PR(V, T, Tc[i], omega[i], k[i], a[i], b[i])
    return np.sum(Z)

def PR_EoS_solver(T, P, Tc, omega, k, a, b):
    """
    Solves the Peng-Robinson (PR) Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): PR equation of state parameter of components.
    a (array): PR equation of state parameter of components.
    b (array): PR equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the PR equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(PR_EoS, V0, args=(T, P, Tc, omega, k, a, b)).x[0]
    return V

```

# apps/gas_solvers/solve_RK.py

```py
import numpy as np
from scipy.optimize import root

def RK_EoS(V, T, P, Tc, omega, a, b):
    """
    Calculates the molar volume for a mixture of components using the Redlich-Kwong (RK) Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): RK equation of state parameter of components.
    b (array): RK equation of state parameter of components.

    Returns:
    float: Residual function for the RK equation of state.
    """
    def f_RK(V, T, Tc, omega, a, b):
        """
        Redlich-Kwong equation of state for a single component.
        """
        Tr = T / Tc
        return (a / V**2) * (1 + (2 * omega - 1) * np.sqrt(Tr) + (1 - omega) * Tr) - P / (R * T) + (b * P) / (V * R * T)

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_RK(V, T, Tc[i], omega[i], a[i], b[i])
    return np.sum(Z)

def RK_EoS_solver(T, P, Tc, omega, a, b):
    """
    Solves the Redlich-Kwong (RK) Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): RK equation of state parameter of components.
    b (array): RK equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the RK equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(RK_EoS, V0, args=(T, P, Tc, omega, a, b)).x[0]
    return V
"""
This code defines two functions RK_EoS and RK_EoS_solver. 
The RK_EoS function calculates the residual function for the RK equation of state,
 while the RK_EoS_solver function solves the RK equation of state for a mixture of components.
 The input parameters for the RK_EoS_solver function include the 
temperature, pressure, critical temperature, acentric factor, and RK equation of state parameters
 for each component. The function returns the molar volume that satisfies the RK equation of state
 for the given conditions
"""

```

# apps/gas_solvers/solve_SRK_association.py

```py
import numpy as np
from scipy.optimize import root

def SRK_EoS_association(V, T, P, Tc, omega, a, b, c, d):
    """
    Calculates the molar volume for a mixture of components using the Soave-Redlich-Kwong (SRK) Equation of State with Association.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): SRK equation of state parameter of components.
    b (array): SRK equation of state parameter of components.
    c (array): Association parameter of components.
    d (array): Association parameter of components.

    Returns:
    float: Residual function for the SRK equation of state with association.
    """
    def f_SRK_association(V, T, Tc, omega, a, b, c, d):
        """
        Soave-Redlich-Kwong equation of state with association for a single component.
        """
        Tr = T / Tc
        return (a / V**2) * (1 + c * np.sqrt(Tr) + d * Tr) + (b * P) / (V * R * T) - P / (R * T)

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_SRK_association(V, T, Tc[i], omega[i], a[i], b[i], c[i], d[i])
    return np.sum(Z)

def SRK_EoS_association_solver(T, P, Tc, omega, a, b, c, d):
    """
    Solves the Soave-Redlich-Kwong (SRK) Equation of State with Association for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): SRK equation of state parameter of components.
    b (array): SRK equation of state parameter of components.
    c (array): Association parameter of components.
    d (array): Association parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the SRK equation of state with association for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(SRK_EoS_association, V0, args=(T, P, Tc, omega, a, b, c, d)).x[0]
    return V
"""
This code defines two functions SRK_EoS_association and SRK_EoS_association_solver.
The `SRK_EoS_associationfunction calculates the residual function for the SRK equation of state
 with association, while theSRK_EoS_association_solverfunction solves the SRK equation of state 
with association for a mixture of components. The input parameters for theSRK_EoS_association_solver`
 function include the temperature, pressure, critical temperature, acentric factor, 
SRK equation of state parameters, and association parameters for each component.
 The function returns the molar volume that satisfies the SRK equation of state 
with association for the given conditions.
To use the code, you would provide the necessary input parameters for the
 SRK_EoS_association_solver function and call the function. The function will then
 return the molar volume that satisfies the SRK equation of state with association
 for the given conditions. It is important to note that the SRK equation of state parameters
 a, b, c, and d need to be calculated beforehand using methods such as regression analysis or
 the modified corresponding states (MCS) method.
"""


```

# apps/gas_solvers/solve_SRK_iter.py

```py
import numpy as np

def soave_redlich_kwong(P, T, z, w, Tr, Pr):
    # Constants
    R = 8.314 # J/mol*K
    a = (0.42747 * R**2 * Tr**2) / Pr
    b = 0.08664 * R * Tr / Pr
    # Cubic equation coefficients
    A = a * P / (R * T)**2
    B = b * P / (R * T)
    # Cubic equation
    coeffs = [1, -1, -A + B - B**2, -A * B + B**2]
    roots = np.roots(coeffs)
    # Find the real root
    for root in roots:
        if np.isreal(root):
            v = np.real(root)
            break
    # Ideal gas contribution
    ideal = z - 1 - np.log(z - B)
    # Residual contribution
    residual = -(2 * np.sqrt(a * B) / (R * T)) * np.log((v + (1 + np.sqrt(2)) * B) / (v + (1 - np.sqrt(2)) * B))
    return ideal + residual

def solve_soave_redlich_kwong(P, T, z, w, Tr, Pr, tolerance=1e-6):
    # Initial estimate for v
    v = 0.001
    # Iterate until the desired tolerance is reached
    while True:
        f = soave_redlich_kwong(P, T, z, w, Tr, Pr)
        df = (soave_redlich_kwong(P, T, z, w, Tr, Pr, v + tolerance) - f) / tolerance
        v_new = v - f / df
        if abs(v_new - v) < tolerance:
            break
        v = v_new
    return v

```

# apps/gas_solvers/solve_SRK.py

```py
import numpy as np
from scipy.optimize import root

def SRK_EoS(V, T, P, Tc, omega, a, b):
    """
    Calculates the molar volume for a mixture of components using the Soave-Redlich-Kwong (SRK) Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): SRK equation of state parameter of components.
    b (array): SRK equation of state parameter of components.

    Returns:
    float: Residual function for the SRK equation of state.
    """
    def f_SRK(V, T, Tc, omega, a, b):
        """
        Soave-Redlich-Kwong equation of state for a single component.
        """
        Tr = T / Tc
        return (a / V**2) * (1 + (0.48 + 1.574 * omega - 0.176 * omega**2) * np.sqrt(Tr)) - P / (R * T) + (b * P) / (V * R * T)

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_SRK(V, T, Tc[i], omega[i], a[i], b[i])
    return np.sum(Z)

def SRK_EoS_solver(T, P, Tc, omega, a, b):
    """
    Solves the Soave-Redlich-Kwong (SRK) Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    a (array): SRK equation of state parameter of components.
    b (array): SRK equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the SRK equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(SRK_EoS, V0, args=(T, P, Tc, omega, a, b)).x[0]
    return V

"""
This code defines two functions SRK_EoS and SRK_EoS_solver.
 The SRK_EoS function calculates the residual function for the SRK equation of state, 
while the SRK_EoS_solver function solves the SRK equation of state for a mixture of components. 
The input parameters for the SRK_EoS_solver function include the 
temperature, pressure, critical temperature, acentric factor, 
and SRK equation of state parameters for each component. 
The function returns the molar volume that satisfies the SRK equation of state for the given conditions.

To use the code, you would provide the necessary input parameters for the SRK_EoS_solver function and call the function. The function will then return the molar volume that satisfies the SRK equation of state for the given conditions. It is important to note that the SRK equation of state parameters a and b need to be calculated beforehand using methods such as regression analysis or the modified corresponding states (MCS) method.
"""

"""

```

# apps/gas_solvers/solve_van_der_waals_molar_volume_iter.py

```py
import numpy as np

def van_der_waals(P, V, a, b):
    """
    Calculates the pressure for a substance using the Van der Waals equation of state.

    Parameters:
    P (float): Pressure (Pa).
    V (float): Molar volume (m^3/mol).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.

    Returns:
    float: Pressure (Pa).
    """
    return (P + a / V**2) * (V - b) - 8.314 * 298.15 / V

def solve_van_der_waals_molar_volume(P, a, b, tolerance=1e-6):
    """
    Solves for the molar volume using the Van der Waals equation of state.

    Parameters:
    P (float): Pressure (Pa).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.
    tolerance (float, optional): Tolerance for the iteration (default is 1e-6).

    Returns:
    float: Molar volume (m^3/mol).
    """
    # Initial estimate for molar volume
    V = 0.1
    # Iterate until the desired tolerance is reached
    while True:
        f = van_der_waals(P, V, a, b)
        df = (van_der_waals(P, V + tolerance, a, b) - f) / tolerance
        V_new = V - f / df
        if abs(V_new - V) < tolerance:
            break
        V = V_new
    return V

```

# apps/gas_solvers/solve_van_der_waals_pressure_iter.py

```py
import numpy as np

def van_der_waals(P, V, a, b):
    """
    Calculates the pressure for a substance using the Van der Waals equation of state.

    Parameters:
    P (float): Pressure (Pa).
    V (float): Volume (m^3).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.

    Returns:
    float: Pressure (Pa).
    """
    return (P + a / V**2) * (V - b) - 8.314 * 298.15 / V

def solve_van_der_waals(V, a, b, tolerance=1e-6):
    """
    Solves the Van der Waals equation of state for a substance.

    Parameters:
    V (float): Volume (m^3).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.
    tolerance (float, optional): Tolerance for the iteration (default is 1e-6).

    Returns:
    float: Pressure (Pa).
    """
    # Initial estimate for pressure
    P = 1e5
    # Iterate until the desired tolerance is reached
    while True:
        f = van_der_waals(P, V, a, b)
        df = (van_der_waals(P + tolerance, V, a, b) - f) / tolerance
        P_new = P - f / df
        if abs(P_new - P) < tolerance:
            break
        P = P_new
    return P

```

# apps/gas_solvers/solve_VDW.py

```py
import numpy as np
from scipy.optimize import root

def vdW_EoS(V, T, P, a, b):
    """
    Calculates the molar volume for a mixture of components using the Van der Waals Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    a (array): Van der Waals equation of state parameter of components.
    b (array): Van der Waals equation of state parameter of components.

    Returns:
    float: Residual function for the Van der Waals equation of state.
    """
    def f_vdW_EoS(V, T, a, b):
        """
        Van der Waals equation of state for a single component.
        """
        return (R * T) / (V - b) - a / (V**2) - P / (R * T)

    N = len(a)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_vdW_EoS(V, T, a[i], b[i])
    return np.sum(Z)

def vdW_EoS_solver(T, P, a, b):
    """
    Solves the Van der Waals Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    a (array): Van der Waals equation of state parameter of components.
    b (array): Van der Waals equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the Van der Waals equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(vdW_EoS, V0, args=(T, P, a, b)).x[0]
    return V

"""
This code defines two functions vdW_EoS and vdW_EoS_solver. 
The vdW_EoS function calculates the residual function for the Van der Waals equation of state,
 while the vdW_EoS_solver function solves the Van der Waals equation of state for a mixture
 of components. 
The input parameters for the vdW_EoS_solver function include the temperature,
 pressure, and Van der Waals equation of state parameters a and b for each component.
 The function returns the molar volume that satisfies the Van der Waals equation
 of state for the given conditions.

To use the code, you would provide the necessary input parameters for the vdW_EoS_solver function
 and call the function.
 The function will then return the molar volume that satisfies the Van der Waals
 equation of state for the given conditions. It is important to note that the Van der Waals 
equation of state parameters a and b need to be calculated beforehand 
using methods such as regression analysis or the modified corresponding states (MCS) method.

In this code, the root function from the scipy.optimize module is used to solve for the 
molar volume that satisfies the Van der Waals equation of state. 
The root function uses a numerical method to find the root of a function, in this case, 
the residual function for the Van der Waals equation of state. 
The V0 value is set as the initial estimate for the molar volume and is 
calculated as the molar volume that would correspond to an ideal gas at the given temperature 
and pressure. The V value returned by the root function is the molar volume that satisfies
 the Van der Waals equation of state for the given conditions.
"""


```

# apps/gas_solvers/solve_virial_iter.py

```py
import numpy as np

def virial(P, T, z, B0, B1, B2, B3):
    # Cubic equation coefficients
    A = B0 + B1 / T + B2 / T**2 + B3 / T**3
    B = B1 + 2 * B2 / T + 3 * B3 / T**2
    C = B2 + 3 * B3 / T
    D = B3
    # Cubic equation
    coeffs = [1, -1, A - B * z + C * z**2 - D * z**3, -A * z + (B - C * z + D * z**2) * z - D * z**3]
    roots = np.roots(coeffs)
    # Find the real root
    for root in roots:
        if np.isreal(root):
            v = np.real(root)
            break
    return v

def solve_virial(P, T, z, B0, B1, B2, B3, tolerance=1e-6):
    # Initial estimate for v
    v = 0.001
    # Iterate until the desired tolerance is reached
    while True:
        f = virial(P, T, z, B0, B1, B2, B3)
        df = (virial(P, T, z, B0, B1, B2, B3, v + tolerance) - f) / tolerance
        v_new = v - f / df
        if abs(v_new - v) < tolerance:
            break
        v = v_new
    return v

```

# apps/gas_solvers/solve_virial.py

```py
import numpy as np
from scipy.optimize import root

def virial_EoS(V, T, P, B, C):
    """
    Calculates the molar volume for a mixture of components using the Virial Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    B (array): Second virial coefficient of components.
    C (array): Third virial coefficient of components.

    Returns:
    float: Residual function for the Virial equation of state.
    """
    def f_virial_EoS(V, T, B, C):
        """
        Virial equation of state for a single component.
        """
        return (R * T) / V - P + B / V**2 + C / V**3

    N = len(B)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_virial_EoS(V, T, B[i], C[i])
    return np.sum(Z)

def virial_EoS_solver(T, P, B, C):
    """
    Solves the Virial Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    B (array): Second virial coefficient of components.
    C (array): Third virial coefficient of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the Virial equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(virial_EoS, V0, args=(T, P, B, C)).x[0]
    return V
"""
This code defines two functions virial_EoS and virial_EoS_solver. The virial_EoS function calculates the residual function for the Virial equation of state, while the virial_EoS_solver function solves the Virial equation of state for a mixture of components. The input parameters for the virial_EoS_solver function include the temperature, pressure, and second and third virial coefficients B and C for each component. The function returns the molar volume that satisfies the Virial equation of state for the given conditions.

To use the code, you would provide the necessary input parameters for the virial_EoS_solver function and call the function. The function will then return the molar volume that satisfies the Virial equation of state for the given conditions. It is important to note that the virial coefficients B and C need to be calculated beforehand using methods such as regression analysis or experimental data.
In this code, the root function from the scipy.optimize module is used to solve for the molar volume that satisfies the Virial equation of state. The root function uses a numerical method to find the root of a function, in this case, the residual function for the Virial equation of state. The V0 value is set as the initial estimate for the molar volume and is calculated as the molar volume that would correspond to an ideal gas at the given temperature and pressure. The V value returned by the root function is the molar volume that satisfies the Virial equation of state for the given conditions.

It is important to note that the Virial equation of state is a simplified model that only considers the first three terms in the virial expansion. The higher order terms in the virial expansion become increasingly important at higher pressures and lower temperatures. As a result, the Virial equation of state is only applicable for a limited range of conditions and may not provide accurate results for all systems.
"""


```

# apps/heat_exchangers/background.jpg

This is a binary file of the type: Image

# apps/heat_exchangers/icon.png

This is a binary file of the type: Image

# apps/heat_exchangers/LMTD.py

```py
import math

def LMTD(T1_in, T1_out, T2_in, T2_out):
    delta_T1 = T1_out - T2_in
    delta_T2 = T1_in - T2_out
    return (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)


T1_in = 100.0 # hot fluid inlet temperature in °C
T1_out = 80.0 # hot fluid outlet temperature in °C
T2_in = 20.0 # cold fluid inlet temperature in °C
T2_out = 30.0 # cold fluid outlet temperature in °C

lmtd = LMTD(T1_in, T1_out, T2_in, T2_out)

print("The LMTD is:", lmtd, "°C")

```

# apps/heat_exchangers/pygame_heat_exchanger_v1_1.py

```py
import pygame

# Constants and calculation function

# ...

# Pygame initialization
pygame.init()
pygame.display.set_caption("Heat Transfer Coefficient Calculator")

# Set up the Pygame window
#size = (800, 600)
#screen = pygame.display.set_mode(size)
#bg_color = (255, 255, 255)
#font = pygame.font.SysFont(None, 30)

# Set up the Pygame window
width, height = 800, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
bg_color = (255, 255, 255)
font = pygame.font.SysFont(None, 30)

# Create input fields

# Create input fields
labels = ["Thermal conductivity of tube wall (W/mK): ", "Diameter of tube (mm): ", "Length of tube (mm): ",
          "Density of vapor (kg/m^3): ", "Viscosity of vapor (Pa.s): ", "Density of liquid (kg/m^3): ",
          "Viscosity of liquid (Pa.s): ", "Surface tension (N/m): ", "Acceleration due to gravity (m/s^2): "]
inputs = []
for i in range(len(labels)):
    input_box = pygame.Rect(350, 50 + i * 50, 200, 32)
    inputs.append({"rect": input_box, "color": pygame.Color("white"), "text": "", "label": labels[i]})

# Add images and icons
icon = pygame.image.load("icon.png")
background = pygame.image.load("background.jpg")

# Add a "Calculate" button
button = pygame.Rect(350, 500, 100, 50)
button_color = pygame.Color("blue")
button_text = font.render("Calculate", True, (255, 255, 255))

# Add a "Quit" button
quit_button = pygame.Rect(650, 500, 100, 50)
quit_button_color = pygame.Color("red")
quit_button_text = font.render("Quit", True, (255, 255, 255))


# Function to calculate the convective heat transfer coefficient
def calculate_h():
    global k_tube, d_tube, l_tube, rho_v, mu_v, rho_l,mu_l, sigma, g
h = 100
# Main loop
while True:
    # Get user inputs, calculate the heat transfer coefficient, and draw the UI
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            for inp in inputs:
                if inp["rect"].collidepoint(event.pos):
                    inp["color"] = pygame.Color("yellow")
                else:
                    inp["color"] = pygame.Color("white")
            if button.collidepoint(event.pos):
                # Calculate the heat transfer coefficient
                k_tube = float(inputs[0]["text"])
                d_tube = float(inputs[1]["text"])
                l_tube = float(inputs[2]["text"])
                rho_v = float(inputs[3]["text"])
                mu_v = float(inputs[4]["text"])
                rho_l = float(inputs[5]["text"])
                mu_l = float(inputs[6]["text"])
                sigma = float(inputs[7]["text"])
                g = float(inputs[8]["text"])
                calculate_h()
            elif quit_button.collidepoint(event.pos):
                pygame.quit()
                quit()

        if event.type == pygame.KEYDOWN:
            for inp in inputs:
                if inp["color"] == pygame.Color("yellow"):
                    if event.key == pygame.K_BACKSPACE:
                        inp["text"] = inp["text"][:-1]
                    else:
                        inp["text"] += event.unicode

    # Draw the UI
    screen.fill((255, 255, 255))  # Set the background color to white
    #screen.blit(icon, (100, 50))
    for inp in inputs:
        label = font.render(inp["label"], True, (0, 0, 0))
        screen.blit(label, (50, inp["rect"].y))
        pygame.draw.rect(screen, inp["color"], inp["rect"], 2)
        text_surface = font.render(inp["text"], True, (0, 0, 0))
        screen.blit(text_surface, (inp["rect"].x + 5, inp["rect"].y + 5))
    pygame.draw.rect(screen, button_color, button)
    screen.blit(button_text, (button.x + 15, button.y + 15))
    pygame.draw.rect(screen, quit_button_color, quit_button)
    screen.blit(quit_button_text, (quit_button.x + 15, quit_button.y + 15))

    # Display the result
    if h > 0:
        result_text = font.render("The convective heat transfer coefficient is %.2f W/m^2K." % h, True, (0, 0, 0))
        screen.blit(result_text, (50, 450))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()

```

# apps/heat_exchangers/pygame_heat_exchanger_v1_2.py.py

```py
import pygame

# Constants and calculation function

h = 100

# Pygame initialization
pygame.init()
pygame.display.set_caption("Heat Transfer Coefficient Calculator")

# Set up the Pygame window
width, height = 800, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
bg_color = (255, 255, 255)
font = pygame.font.SysFont(None, 30)

# Create input fields and create the "Calculate" button
# Create input fields
labels = ["Thermal conductivity of tube wall (W/mK): ", "Diameter of tube (mm): ", "Length of tube (mm): ",
          "Density of vapor (kg/m^3): ", "Viscosity of vapor (Pa.s): ", "Density of liquid (kg/m^3): ",
          "Viscosity of liquid (Pa.s): ", "Surface tension (N/m): ", "Acceleration due to gravity (m/s^2): "]
inputs = []
for i in range(len(labels)):
    input_box = pygame.Rect(350, 50 + i * 50, 200, 32)
    inputs.append({"rect": input_box, "color": pygame.Color("white"), "text": "", "label": labels[i]})

# Add images and icons
icon = pygame.image.load("icon.png")
background = pygame.image.load("background.jpg")

# Add a "Calculate" button
button = pygame.Rect(350, 500, 100, 50)
button_color = pygame.Color("blue")
button_text = font.render("Calculate", True, (255, 255, 255))


# Add a "Quit" button
quit_button = pygame.Rect(650, 500, 100, 50)
quit_button_color = pygame.Color("red")
quit_button_text = font.render("Quit", True, (255, 255, 255))

# Main loop
# Main loop
while True:
    # Get user inputs, calculate the heat transfer coefficient, and draw the UI
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            for inp in inputs:
                if inp["rect"].collidepoint(event.pos):
                    inp["color"] = pygame.Color("yellow")
                else:
                    inp["color"] = pygame.Color("white")
            if button.collidepoint(event.pos):
                # Calculate the heat transfer coefficient
                k_tube = float(inputs[0]["text"])
                d_tube = float(inputs[1]["text"])
                l_tube = float(inputs[2]["text"])
                rho_v = float(inputs[3]["text"])
                mu_v = float(inputs[4]["text"])
                rho_l = float(inputs[5]["text"])
                mu_l = float(inputs[6]["text"])
                sigma = float(inputs[7]["text"])
                g = float(inputs[8]["text"])
                calculate_h()
            elif quit_button.collidepoint(event.pos):
                pygame.quit()
                quit()

        if event.type == pygame.KEYDOWN:
            for inp in inputs:
                if inp["color"] == pygame.Color("yellow"):
                    if event.key == pygame.K_BACKSPACE:
                        inp["text"] = inp["text"][:-1]
                    else:
                        inp["text"] += event.unicode

        if event.type == pygame.VIDEORESIZE:
            # Resize the window and the UI elements
            width, height = event.size
            screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            bg_color = (255, 255, 255)
            font_size = min(int(height / 20), 40)
            font = pygame.font.SysFont(None, font_size)
            button = pygame.Rect(50, int(height / 2.5), 150, int(height / 15))
            quit_button = pygame.Rect(int(width / 1.3), int(height / 1.2), int(width / 8), int(height / 15))
            for i, inp in enumerate(inputs):
                inp["rect"] = pygame.Rect(50, int(height / 10) + i * int(height / 15), width - 100, int(height / 20))

    # Draw the UI
    screen.fill(bg_color)
    for inp in inputs:
        label = font.render(inp["label"], True, (0, 0, 0))
        screen.blit(label, (50, inp["rect"].y))
        pygame.draw.rect(screen, (0, 0, 0), inp["rect"], 2)
        pygame.draw.rect(screen, inp["color"], inp["rect"], 0)
        text_surface = font.render(inp["text"], True, (0, 0, 0))
        screen.blit(text_surface, (inp["rect"].x + 5, inp["rect"].y + 5))
    pygame.draw.rect(screen, button_color, button)
    screen.blit(button_text, (button.x + 15, button.y + 15))
    pygame.draw.rect(screen, quit_button_color, quit_button)
    screen.blit(quit_button_text, (quit_button.x + 15, quit_button.y + 15))

    # Display the result
    if h > 0:
        result_text = font.render("The convective heat transfer coefficient is %.2f W/m^2K." % h, True, (0, 0, 0))
        screen.blit(result_text, (50, int(height / 1.5)))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()

```

# apps/heat_exchangers/pygame_heat_exchanger.py

```py
import pygame

# Constants
k_tube = 0.0
d_tube = 0.0
l_tube = 0.0
rho_v = 0.0
mu_v = 0.0
rho_l = 0.0
mu_l = 0.0
sigma = 0.0
g = 0.0

# Pygame initialization
pygame.init()
pygame.display.set_caption("Heat Transfer Coefficient Calculator")

# Set up the Pygame window
size = (800, 600)
screen = pygame.display.set_mode(size)
bg_color = (255, 255, 255)
font = pygame.font.SysFont(None, 30)

# Create input fields
labels = ["Thermal conductivity of tube wall (W/mK): ", "Diameter of tube (mm): ", "Length of tube (mm): ",
          "Density of vapor (kg/m^3): ", "Viscosity of vapor (Pa.s): ", "Density of liquid (kg/m^3): ",
          "Viscosity of liquid (Pa.s): ", "Surface tension (N/m): ", "Acceleration due to gravity (m/s^2): "]
inputs = []
for i in range(len(labels)):
    input_box = pygame.Rect(350, 50 + i * 50, 200, 32)
    inputs.append({"rect": input_box, "color": pygame.Color("white"), "text": "", "label": labels[i]})

# Add images and icons
icon = pygame.image.load("icon.png")
background = pygame.image.load("background.jpg")

# Add a "Calculate" button
button = pygame.Rect(350, 500, 100, 50)
button_color = pygame.Color("blue")
button_text = font.render("Calculate", True, (255, 255, 255))

# Function to calculate the convective heat transfer coefficient
def calculate_h():
    global k_tube, d_tube, l_tube, rho_v, mu_v, rho_l,mu_l, sigma, g

# Get user inputs
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        quit()
    if event.type == pygame.MOUSEBUTTONDOWN:
        for inp in inputs:
            if inp["rect"].collidepoint(event.pos):
                inp["color"] = pygame.Color("yellow")
            else:
                inp["color"] = pygame.Color("white")
        if button.collidepoint(event.pos):
            k_tube = float(inputs[0]["text"])
            d_tube = float(inputs[1]["text"])
            l_tube = float(inputs[2]["text"])
            rho_v = float(inputs[3]["text"])
            mu_v = float(inputs[4]["text"])
            rho_l = float(inputs[5]["text"])
            mu_l = float(inputs[6]["text"])
            sigma = float(inputs[7]["text"])
            g = float(inputs[8]["text"])
            calculate_h()

    if event.type == pygame.KEYDOWN:
        for inp in inputs:
            if inp["color"] == pygame.Color("yellow"):
                if event.key == pygame.K_BACKSPACE:
                    inp["text"] = inp["text"][:-1]
                else:
                    inp["text"] += event.unicode

# Draw the UI
screen.blit(background, (0, 0))
screen.blit(icon, (100, 50))
for inp in inputs:
    label = font.render(inp["label"], True, (0, 0, 0))
    screen.blit(label, (50, inp["rect"].y))
    pygame.draw.rect(screen, inp["color"], inp["rect"], 2)
    text_surface = font.render(inp["text"], True, (0, 0, 0))
    screen.blit(text_surface, (inp["rect"].x + 5, inp["rect"].y + 5))
pygame.draw.rect(screen, button_color, button)
screen.blit(button_text, (button.x + 15, button.y + 15))

h = 100
# Display the result
if h > 0:
    result_text = font.render("The convective heat transfer coefficient is %.2f W/m^2K." % h, True, (0, 0, 0))
    screen.blit(result_text, (50, 450))

# Update the display
pygame.display.update()

#Quit pygame
pygame.quit()

```

# apps/liquid_solvers/solve_CPA.py

```py
import numpy as np
from scipy.optimize import root
#Cubic-Plus-Association (CPA) Equation of State for a mixture of components
def CPA_EoS(V, T, P, Zc, Tc, omega, k, m, a, b):
    """
    Calculates the molar volume for a mixture of components using the Cubic-Plus-Association (CPA) Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Zc (array): Critical compressibility factor of components.
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): Association constant of components.
    m (array): Association parameter of components.
    a (array): CPA cubic equation of state parameter of components.
    b (array): CPA cubic equation of state parameter of components.

    Returns:
    float: Residual function for the CPA equation of state.
    """
    def f_cubic(V, T, a, b):
        """
        Cubic equation of state for a single component.
        """
        return (a / (V - b)) - (8 * T / (3 * V**2))

    def f_association(V, T, Zc, Tc, omega, k, m):
        """
        Association term for a single component.
        """
        Tr = T / Tc
        return np.exp(-m * (1 - np.sqrt(Tr))**2) * (Zc - 1 - k * V / (T * Zc))

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = root(f_cubic, Zc[i], args=(T, a[i], b[i])).x[0] + f_association(V, T, Zc[i], Tc[i], omega[i], k[i], m[i])
    return np.sum(Z) - P * V / (R * T)

def CPA_EoS_solver(T, P, Zc, Tc, omega, k, m, a, b):
    """
    Solves the Cubic-Plus-Association (CPA) Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Zc (array): Critical compressibility factor of components.
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): Association constant of components.
    m (array): Association parameter of components.
    a (array): CPA cubic equation of state parameter of components.
    b (array): CPA cubic equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the CPA equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) /P
    V = root(CPA_EoS, V0, args=(T, P, Zc, Tc, omega, k, m, a, b)).x[0]
    return V
"""
This code defines two functions `CPA_EoS` and `CPA_EoS_solver`. 
The `CPA_EoS` function calculates the residual function for the CPA equation of state,
The `CPA_EoS_solver` function solves the CPA equation of state for a mixture of components. 
The input parameters for the `CPA_EoS_solver` function include:
temperature, pressure, critical compressibility factor, critical temperature, acentric factor, 
association constant, association parameter, and CPA cubic equation of state parameters 
for each component. The function returns the molar volume that satisfies the CPA equation of state 
#for the given conditions.
To use the `CPA_EoS_solver` function, you would provide the necessary input parameters and call the function. The function will return the molar volume that satisfies the CPA equation of state for the given conditions.
"""

```

# apps/liquid_solvers/solve_Flory_Huggins.py

```py
import numpy as np

def FloryHuggins(T, x, chi):
    """
    Calculates the activity coefficients for a binary mixture using the Flory-Huggins Model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    chi (array): Flory-Huggins interaction parameter.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    activity = np.zeros(2)
    activity[0] = np.exp(-chi * x[1])
    activity[1] = np.exp(-chi * x[0])
    return activity

```

# apps/liquid_solvers/solve_HenrysLawModel.py

```py
import numpy as np

def HenrysLawModel(T, x, z, H1, H2):
    """
    Calculates the activity coefficients for a binary mixture using the Henry's Law Model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    z (array): Overall molality of components in the mixture.
    H1 (array): Henry's Law constant for component 1.
    H2 (array): Henry's Law constant for component 2.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    activity = np.zeros(2)
    for i in range(2):
        activity[i] = H1[i] * x[i] / (H2[i] * (1 - x[i]))
    return activity


```

# apps/liquid_solvers/solve_NRTL.py

```py
import numpy as np

def NRTL(x, T, gamma, alpha, tau):
    """
    Calculates the activity coefficients for a binary mixture using the NRTL equation.

    Parameters:
    x (list or numpy array): Composition of the mixture (mol fraction).
    T (float): Temperature (K).
    gamma (numpy array): Matrix of binary interaction parameters.
    alpha (numpy array): Matrix of temperature-dependent parameters.
    tau (numpy array): Matrix of tau values.

    Returns:
    numpy array: Activity coefficients.
    """
    n = len(x)
    ln_gamma = np.zeros(n)
    for i in range(n):
        for j in range(n):
            ln_gamma[i] += x[j] * (gamma[i, j] + alpha[i, j] * (1 - np.exp(-tau[i, j] * (1 / T - 1 / 298.15))))
    return np.exp(ln_gamma)

def solve_NRTL(x, T, gamma, alpha, tau, tolerance=1e-6):
    """
    Solves the NRTL equation for a binary mixture.

    Parameters:
    x (list or numpy array): Composition of the mixture (mol fraction).
    T (float): Temperature (K).
    gamma (numpy array): Matrix of binary interaction parameters.
    alpha (numpy array): Matrix of temperature-dependent parameters.
    tau (numpy array): Matrix of tau values.
    tolerance (float, optional): Tolerance for the iteration (default is 1e-6).

    Returns:
    numpy array: Activity coefficients.
    """
    # Initial estimate for activity coefficients
    gamma_new = np.ones(len(x))
    # Iterate until the desired tolerance is reached
    while True:
        gamma = gamma_new
        gamma_new = NRTL(x, T, gamma, alpha, tau)
        if np.linalg.norm(gamma_new - gamma) < tolerance:
            break
    return gamma_new

```

# apps/liquid_solvers/solve_StirredTank.py

```py
import numpy as np

def StirredTank(T, x, z, Tc, Vc, omega, k1, k2):
    """
    Calculates the activity coefficients for a binary mixture using the Stirred Tank Model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    z (array): Overall molality of components in the mixture.
    Tc (array): Critical temperature (K) of components.
    Vc (array): Critical volume (m^3/mol) of components.
    omega (array): Acentric factor of components.
    k1 (array): Stirred Tank binary interaction parameters for component 1.
    k2 (array): Stirred Tank binary interaction parameters for component 2.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    activity = np.zeros(2)
    for i in range(2):
        activity[i] = np.exp(np.sum(x * np.log(z * (1 + k1[i] * x + k2[i] * x**2))))
    return activity

```

# apps/liquid_solvers/solve_UNIFAC.py

```py
import numpy as np

def UNIFAC(T, x, z, Tc, Vc, omega, q1, q2):
    """
    Calculates the activity coefficients for a binary mixture using the UNIFAC model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    z (array): Overall molality of components in the mixture.
    Tc (array): Critical temperature (K) of components.
    Vc (array): Critical volume (m^3/mol) of components.
    omega (array): Acentric factor of components.
    q1 (array): UNIFAC binary interaction parameters for component 1.
    q2 (array): UNIFAC binary interaction parameters for component 2.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    k = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + k * (1 - np.sqrt(T / Tc)))**2
    a = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            a[i, j] = alpha[i] * alpha[j] * np.exp((q1[i] * q2[j]) / (1 + k[i] * k[j]))
    activity = np.zeros(2)
    for i in range(2):
        activity[i] = np.exp(np.sum(x * np.log(z * a[i, :])))
    return activity

```

# apps/liquid_solvers/solve_UNIQUAC.py

```py
import numpy as np

def UNIQUAC(T, x, z, Tc, Vc, omega, q1, q2):
    """
    Calculates the activity coefficients for a binary mixture using the UNIQUAC model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    z (array): Overall molality of components in the mixture.
    Tc (array): Critical temperature (K) of components.
    Vc (array): Critical volume (m^3/mol) of components.
    omega (array): Acentric factor of components.
    q1 (array): UNIQUAC binary interaction parameters for component 1.
    q2 (array): UNIQUAC binary interaction parameters for component 2.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    k = 0.48 + 1.574 * omega - 0.176 * omega**2
    alpha = (1 + k * (1 - np.sqrt(T / Tc)))**2
    a = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            a[i, j] = alpha[i] * alpha[j] * np.exp((q1[i] * q2[j]) / (1 + k[i] * k[j]))
    activity = np.zeros(2)
    for i in range(2):
        activity[i] = np.exp(np.sum(x * np.log(z * a[i, :])))
    return activity

```

# apps/Mass_Energy_Balance_NOx_Module.py

```py
def compute_dry_to_wet_sample(Fukthalt, C_viktsprocent_torr, H_viktsprocent_torr, O_viktsprocent_torr, N_viktsprocent_torr, S_viktsprocent_torr, Aska_viktsprocent_torr):
    # Omräkning från torrt till fuktigt prov
    c_viktsprocent_fuktigt = (1 - Fukthalt / 100) * C_viktsprocent_torr
    h_viktsprocent_fuktigt = (1 - Fukthalt / 100) * H_viktsprocent_torr
    o_viktsprocent_fuktigt = (1 - Fukthalt / 100) * O_viktsprocent_torr
    n_viktsprocent_fuktigt = (1 - Fukthalt / 100) * N_viktsprocent_torr
    s_viktsprocent_fuktigt = (1 - Fukthalt / 100) * S_viktsprocent_torr
    aska_viktsprocent_fuktigt = (1 - Fukthalt / 100) * Aska_viktsprocent_torr
    
    return c_viktsprocent_fuktigt, h_viktsprocent_fuktigt, o_viktsprocent_fuktigt, n_viktsprocent_fuktigt, s_viktsprocent_fuktigt, aska_viktsprocent_fuktigt

def compute_dry_to_wet_oil_sample(fukt_OE, c_vp_torr_OE, h_vp_torr_OE, o_vp_torr_OE, n_vp_torr_OE, s_vp_torr_OE, aska_vp_torr_OE):
    # Omräkning från torrt till fuktigt prov olja
    c_vp_fukt_OE = (1 - fukt_OE / 100) * c_vp_torr_OE
    h_vp_fukt_OE = (1 - fukt_OE / 100) * h_vp_torr_OE
    o_vp_fukt_OE = (1 - fukt_OE / 100) * o_vp_torr_OE
    n_vp_fukt_OE = (1 - fukt_OE / 100) * n_vp_torr_OE
    s_vp_fukt_OE = (1 - fukt_OE / 100) * s_vp_torr_OE
    aska_vp_fukt_OE = (1 - fukt_OE / 100) * aska_vp_torr_OE
    
    return c_vp_fukt_OE, h_vp_fukt_OE, o_vp_fukt_OE, n_vp_fukt_OE, s_vp_fukt_OE, aska_vp_fukt_OE

# More functions here, following the same style...

def compute_outputs(Panna_drift, Gasanalys_fel, torrt_rokgasflode, torrt_rokgasflode_OE, bransleflode, Olja, verklig_rokgasmangd_torr_medluft, verklig_rokgas_vat, effekt_anga, effekt_inkl_rokgas, effekt_inkl_rokgas_OE, RökgasNOx):
    # Utgångar
    if Panna_drift and not Gasanalys_fel:
        Torrt_rökgasflöde = limit(0, torrt_rokgasflode + torrt_rokgasflode_OE, 100)
        Bränsleflöde = limit(0, bransleflode + Olja, 100)
        Verklig_rökgasmängd_torr = limit(0, verklig_rokgasmangd_torr_medluft, 100)
        Verklig_rökgasmängd_våt = limit(0, verklig_rokgas_vat, 100)
        Verkningsgrad = limit(0, (effekt_anga / 1000) / (effekt_inkl_rokgas + effekt_inkl_rokgas_OE) * 100, 100)
        Nyttiggjord_energi = limit(0, effekt_anga / 1000, 100)
        Tillförd_energi = limit(0, effekt_inkl_rokgas + effekt_inkl_rokgas_OE, 100)
        NOx_flöde = limit(0, RökgasNOx * 2.05 * (torrt_rokgasflode + torrt_rokgasflode_OE) * 3.6 / 1000, 100)
        NOx_emission = limit(0, RökgasNOx * 2.05 * (torrt_rokgasflode + torrt_rokgasflode_OE) / (effekt_inkl_rokgas + effekt_inkl_rokgas_OE), 100)
    else:
        Torrt_rökgasflöde, Bränsleflöde, Verklig_rökgasmängd_torr, Verklig_rökgasmängd_våt, Verkningsgrad, Nyttiggjord_energi, Tillförd_energi, NOx_flöde, NOx_emission = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return Torrt_rökgasflöde, Bränsleflöde, Verklig_rökgasmängd_torr, Verklig_rökgasmängd_våt, Verkningsgrad, Nyttiggjord_energi, Tillförd_energi, NOx_flöde, NOx_emission

def limit(min_val, val, max_val):
    return max(min_val, min(val, max_val))

# You'll have to use these functions in your main execution context to get the desired results.
# ... previous functions ...

def compute_energy_content(c_fuktigt, h_fuktigt, o_fuktigt, s_fuktigt, aska_fuktigt):
    # Beräkning av energiinnehåll
    effekt_fuktigt = 33950 * c_fuktigt + 117200 * h_fuktigt - 24420 * o_fuktigt - 2760 * s_fuktigt - 23720 * aska_fuktigt
    return effekt_fuktigt

def compute_energy_content_oil(c_fukt_OE, h_fukt_OE, o_fukt_OE, s_fukt_OE, aska_fukt_OE):
    # Beräkning av energiinnehåll olja
    effekt_fukt_OE = 33950 * c_fukt_OE + 117200 * h_fukt_OE - 24420 * o_fukt_OE - 2760 * s_fukt_OE - 23720 * aska_fukt_OE
    return effekt_fukt_OE

def compute_dry_gas_flow(torr_rokgas_viktsprocent, rokgas_torr_medluft, rokgas_vat):
    # Beräkning av torrt rökgasflöde
    verklig_rokgasmangd_torr_medluft = rokgas_torr_medluft / (1 + torr_rokgas_viktsprocent / 100)
    verklig_rokgas_vat = rokgas_vat / (1 + torr_rokgas_viktsprocent / 100)
    return verklig_rokgasmangd_torr_medluft, verklig_rokgas_vat

# ... the rest of the pseudocode ...

# You'll have to use these functions in your main execution context to get the desired results.
# ... previous functions ...

def compute_heat_loss(water_content, energy_content):
    # Beräkning av värmeförluster
    heat_loss = water_content * energy_content * 0.2  # 0.2 is an arbitrary factor; adjust as necessary.
    return heat_loss

def calculate_emissions(co2_content, nox_content):
    # Beräkning av emissioner
    total_emissions = co2_content + nox_content
    return total_emissions

def determine_efficiency(energy_input, energy_output):
    # Beräkning av effektivitet
    efficiency = energy_output / energy_input * 100
    return efficiency

def adjust_for_conditions(temperature, pressure, value):
    # Justering för specifika förhållanden
    adjusted_value = value * (1 + 0.01 * temperature) * (1 - 0.01 * pressure)  # Here we assume that for every degree of temperature increase, the value increases by 1% and for every unit of pressure increase, the value decreases by 1%. Adjust the factors as necessary.
    return adjusted_value

# ... potential additional pseudocode ...

# When implementing, make sure you call these functions from your main logic, providing the required parameters, and use the results as appropriate.
# ... previous functions ...

def determine_operating_mode(temperature):
    # Bestäm driftläge baserat på temperatur
    if temperature < 0:
        return "WINTER_MODE"
    elif temperature < 20:
        return "SPRING_FALL_MODE"
    else:
        return "SUMMER_MODE"

def log_system_state(state):
    # Logga systemets aktuella tillstånd
    # Note: This is a pseudocode. In actual implementation, this could be writing to a file, database or even displaying on a dashboard.
    print(f"System is currently in {state} state")

def check_safety_parameters(parameters):
    # Kontrollera säkerhetsparametrar
    if parameters["pressure"] > 100 or parameters["temperature"] > 80:  # arbitrary thresholds
        return False
    return True

def alert_operator(message):
    # Larma operatören
    # This would typically notify a human operator in the real-world scenario.
    print(f"ALERT: {message}")

def initiate_shutdown():
    # Initiera avstängning
    # Here, we might gracefully shut down processes, turn off equipment, etc.
    print("System shutting down...")

def optimize_for_demand(demand, supply):
    # Optimering för efterfrågan
    if demand > supply:
        adjust_supply = supply + (demand - supply) * 0.5  # Adjusting supply to meet demand. 0.5 is an arbitrary factor.
        return adjust_supply
    return supply

# ... more potential pseudocode ...

def main_logic():
    # The main logic where we'll orchestrate all the functions
    current_temperature = get_temperature()  # Assuming there's a function to get the temperature
    current_pressure = get_pressure()  # Assuming there's a function to get the pressure
    current_mode = determine_operating_mode(current_temperature)
    log_system_state(current_mode)

    safety_parameters = {
        "temperature": current_temperature,
        "pressure": current_pressure
    }

    if not check_safety_parameters(safety_parameters):
        alert_operator("Safety parameters exceeded!")
        initiate_shutdown()
        return  # End the program
    
    demand = get_demand()  # Assuming there's a function to retrieve the current energy demand
    supply = get_supply()  # Assuming there's a function to get the current energy supply
    optimized_supply = optimize_for_demand(demand, supply)
    set_supply(optimized_supply)  # Assuming there's a function to set the energy supply

    # ... other main logic ...

# When implementing, be sure to actually create functions like get_temperature, get_pressure, get_demand, get_supply, and set_supply based on your system's actual requirements.


```

# apps/PRESET_pipe_cost_optimization_defaults.xml

```xml
<?xml version ="1.0"?>
<data>
	<ror name="40mm">
		<dim>0.040</dim>
		<mcost>950</mcost>
	</ror>
	<ror name="50mm">
		<dim>0.050</dim>
		<mcost>1000</mcost>
	</ror>
	<ror name="65mm">
		<dim>0.065</dim>
		<mcost>1400</mcost>
	</ror>
	<ror name="100mm">
		<dim>0.100</dim>
		<mcost>1700</mcost>
	</ror>
	<ror name="125mm">
		<dim>0.125</dim>
		<mcost>2400</mcost>
	</ror>
	<ror name="150mm">
		<dim>0.150</dim>
		<mcost>3000</mcost>
	</ror>
	<ror name="200mm">
		<dim>0.200</dim>
		<mcost>3500</mcost>
	</ror>
	<ror name="250mm">
		<dim>0.250</dim>
		<mcost>4500</mcost>
	</ror>
	<ror name="300mm">
		<dim>0.300</dim>
		<mcost>5500</mcost>
	</ror>
	<ror name="350mm">
		<dim>0.350</dim>
		<mcost>6500</mcost>
	</ror>
	<ror name="400mm">
		<dim>0.400</dim>
		<mcost>7500</mcost>
	</ror>
</data>

```

# apps/Safety_Chemical_Safety.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from fpdf import FPDF
import chemicals
import pandas as pd

# Read the AFS data
#afs_data = pd.read_excel("AFS2018merged_data.xlsx")
class App:
    def __init__(self, root):
        self.root = root
        root.title("Chemicals Safety Data")

        # Layout elements
        ttk.Label(root, text="Enter CAS Number:").pack(pady=10)
        self.cas_entry = ttk.Entry(root)
        self.cas_entry.pack(pady=10)

        ttk.Label(root, text="Or Enter Common Name:").pack(pady=10)
        self.common_name_entry = ttk.Entry(root)
        self.common_name_entry.pack(pady=10)

        ttk.Button(root, text="Fetch Data", command=self.fetch_data).pack(pady=10)

        # Results frame
        self.results_frame = ttk.LabelFrame(root, text="Results", padding="10")
        self.results_frame.pack(pady=20, padx=20, fill='both', expand=True)

        # To show data
        self.data_tree = ttk.Treeview(self.results_frame, columns=("Parameter", "Value"), show="headings")
        self.data_tree.heading("Parameter", text="Parameter")
        self.data_tree.heading("Value", text="Value")
        self.data_tree.pack(fill='both', expand=True)

        ttk.Button(root, text="Export to PDF", command=self.export_pdf).pack(pady=10)

    def fetch_data(self):
        CASRN = self.cas_entry.get().strip()
        
        # If CAS is empty, try to fetch CAS from common name
        if not CASRN:
            common_name = self.common_name_entry.get().strip()
            if not common_name:
                messagebox.showerror("Error", "Please enter either a CAS number or a common name.")
                return

            try:
                CASRN = chemicals.identifiers.CAS_from_any(common_name)
            except Exception:
                messagebox.showerror("Error", "Unable to identify the CAS number for the provided name.")
                return
        
        # Clear previous entries
        for row in self.data_tree.get_children():
            self.data_tree.delete(row)

        # Check in AFS Data
        #afs_row = afs_data[afs_data['CAS­nr'] == CASRN]
        #if not afs_row.empty:
        #    self.data_tree.insert("", "end", values=("Chemical Name", afs_row.iloc[0]['Chemical Name']))
        #    self.data_tree.insert("", "end", values=("TWA (ppm)", afs_row.iloc[0]['TWA (ppm)']))
        #    self.data_tree.insert("", "end", values=("TWA (mg/m^3)", afs_row.iloc[0]['TWA (mg/m^3)']))
        #    self.data_tree.insert("", "end", values=("STEL (ppm)", afs_row.iloc[0]['STEL (ppm)']))
        #    self.data_tree.insert("", "end", values=("STEL (mg/m3)", afs_row.iloc[0]['STEL (mg/m3)']))
        #    self.data_tree.insert("", "end", values=("Anm.", afs_row.iloc[0]['Anm.']))
        #    self.data_tree.insert("", "end", values=("Notes", afs_row.iloc[0]['Notes']))

        # Fetch and populate data
        # STEL data
        stel_methods = chemicals.safety.STEL_methods(CASRN)
        for method in stel_methods:
            stel_value = chemicals.safety.STEL(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"STEL ({method})", stel_value))
        
        # TWA data
        twa_methods = chemicals.safety.TWA_methods(CASRN)
        for method in twa_methods:
            twa_value = chemicals.safety.TWA(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"TWA ({method})", twa_value))

        # Ceiling data
        ceiling_methods = chemicals.safety.Ceiling_methods(CASRN)
        for method in ceiling_methods:
            ceiling_value = chemicals.safety.Ceiling(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Ceiling ({method})", ceiling_value))

        # Skin data
        skin_methods = chemicals.safety.Skin_methods(CASRN)
        for method in skin_methods:
            skin_value = chemicals.safety.Skin(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Skin ({method})", skin_value))

        # Carcinogen data
        carcinogen_methods = chemicals.safety.Carcinogen_methods(CASRN)
        for method in carcinogen_methods:
            carcinogen_value = chemicals.safety.Carcinogen(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Carcinogen ({method})", carcinogen_value))

        # Flash temperature data
        t_flash_methods = chemicals.safety.T_flash_methods(CASRN)
        for method in t_flash_methods:
            t_flash_value = chemicals.safety.T_flash(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Flash Temp. ({method})", t_flash_value))

        # Autoignition temperature data
        t_autoignition_methods = chemicals.safety.T_autoignition_methods(CASRN)
        for method in t_autoignition_methods:
            t_autoignition_value = chemicals.safety.T_autoignition(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Autoignition Temp. ({method})", t_autoignition_value))

        # The other methods, such as those related to LFL, UFL, etc. require additional parameters 
        # (e.g., Hc, atoms). For this simple GUI, these parameters would either need to be hardcoded, 
        # or additional input fields would be required.


    def export_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        for row in self.data_tree.get_children():
            parameter, value = self.data_tree.item(row, "values")
            pdf.cell(200, 10, f"{parameter}: {value}", ln=True)

        pdf.output("chemical_data.pdf")
        messagebox.showinfo("Success", "Data exported to chemical_data.pdf")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

```

# apps/Thermodynamic_Cycle_water.py

```py
import tkinter as tk
from tkinter import ttk
from thermo import FlashPureVLS, IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations

def calculate_thermal_efficiency():
    T1 = float(temp_entry.get()) + 273.15
    P1 = float(pressure_entry.get()) * 1e5
    P2 = float(pressure2_entry.get()) * 1e5
    VF3 = 0
    P3 = P2
    P4 = P1

    liquid = IAPWS95Liquid(T=T1, P=P1, zs=[1])
    gas = IAPWS95Gas(T=T1, P=P1, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    stage_1 = flasher.flash(P=P1, T=T1)
    stage_2 = flasher.flash(P=P2, S=stage_1.S())
    stage_3 = flasher.flash(VF=VF3, P=P3)
    stage_4 = flasher.flash(P=P4, S=stage_3.S())

    expander_duty = stage_2.H() - stage_1.H()
    pump_duty = stage_4.H() - stage_3.H()
    heating_duty = stage_1.H() - stage_4.H()

    eta_th = -expander_duty/heating_duty
    result_var.set(f"Thermal Efficiency: {eta_th*100:.2f}%")

app = tk.Tk()
app.title("Thermal Efficiency Calculator")

# Input fields
ttk.Label(app, text="Temperature (°C):").grid(row=0, column=0, padx=10, pady=5)
temp_entry = ttk.Entry(app)
temp_entry.insert(0, '350') # default value
temp_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(app, text="Pressure 1 (bar):").grid(row=1, column=0, padx=10, pady=5)
pressure_entry = ttk.Entry(app)
pressure_entry.insert(0, '100') # default value
pressure_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(app, text="Pressure 2 (bar):").grid(row=2, column=0, padx=10, pady=5)
pressure2_entry = ttk.Entry(app)
pressure2_entry.insert(0, '1') # default value
pressure2_entry.grid(row=2, column=1, padx=10, pady=5)

# Calculate button
ttk.Button(app, text="Calculate Thermal Efficiency", command=calculate_thermal_efficiency).grid(row=3, columnspan=2, pady=10)

# Output label
result_var = tk.StringVar()
result_label = ttk.Label(app, textvariable=result_var)
result_label.grid(row=4, columnspan=2, padx=10, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Air_Cooler_Designer.py

```py

import tkinter as tk
import platform
from tkinter import ttk, DoubleVar, StringVar, IntVar
import ht.air_cooler
class AirCoolerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Air Cooler Design Tool")

        self.vars = {}

        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        self.container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.create_geometry_input_section()
        self.create_temp_correction_section()
        self.create_noise_estimation_section()
        self.create_pressure_drop_section()
        self.create_heat_transfer_coeff_section()
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=500, height=500)
        self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._on_mousewheel))
        self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=500, height=500)
        self.container.bind("<Map>", lambda e: self._on_canvas_configure(None))
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.container.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Cross-platform mouse wheel scrolling
        if platform.system() == "Windows":
            self.canvas.bind('<MouseWheel>', self._on_mousewheel)
            self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._on_mousewheel))
            self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
        elif platform.system() == "Linux":
            self.canvas.bind('<Button-4>', self._on_mousewheel)
            self.canvas.bind('<Button-5>', self._on_mousewheel)
            self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<Button-4><Button-5>', self._on_mousewheel))
            self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<Button-4><Button-5>'))
        elif platform.system() == "Darwin":
            self.canvas.bind('<MouseWheel>', self._on_mousewheel)
            self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._on_mousewheel))
            self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
            
        self.container.bind("<Configure>", self._on_frame_configure)
        self._on_canvas_configure(None)

    def _on_frame_configure(self, event):
        """Update the scrollbars to match the size of the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        if platform.system() == "Windows" or platform.system() == "Darwin":
            self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
        elif platform.system() == "Linux":
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

    def _on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def _create_frame(self, text, row):
        frame = ttk.LabelFrame(self.container, text=text, padding=(10, 5))
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        return frame

    def _create_input_fields(self, frame, labels):
        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w")
            self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

    def create_geometry_input_section(self):
        frame = self._create_frame("Geometry Input", 0)
        labels = ["Surface Area (A) [m^2]", "Tube Diameter [m]", "A_min [m^2]", "A_increase", "A_fin [m^2]",
                  "A_tube_showing [m^2]", "Fin Diameter [m]", "Fin Thickness [m]", "Bare Length [m]",
                  "Pitch Parallel [m]", "Pitch Normal [m]", "Tube Rows", "Density [kg/m^3]", "Heat Capacity [J/kg/K]",
                  "Viscosity [Pa*s]", "Thermal Conductivity [W/m/K]", "Fin Thermal Conductivity [W/m/K]"]

        self._create_input_fields(frame, labels)

    def create_temp_correction_section(self):
        frame = self._create_frame("Temperature Correction Factor", 1)
        labels = ["Thi", "Tho", "Tci", "Tco", "Ntp", "Rows"]
        self._create_input_fields(frame, labels)

        row_end = len(labels)
        ttk.Button(frame, text="Calculate", command=self.calc_temp_correction).grid(row=row_end, columnspan=2)
        self.temp_correction_result_var = StringVar()
        ttk.Label(frame, text="Result").grid(row=row_end + 1, column=0)
        ttk.Label(frame, textvariable=self.temp_correction_result_var).grid(row=row_end + 1, column=1)

    def calc_temp_correction(self):
        result = ht.air_cooler.Ft_aircooler(
            self.vars["Thi"].get(),
            self.vars["Tho"].get(),
            self.vars["Tci"].get(),
            self.vars["Tco"].get(),
            self.vars["Ntp"].get(),
            self.vars["Rows"].get()
        )
        self.temp_correction_result_var.set(result)

    def create_noise_estimation_section(self):
        frame = ttk.LabelFrame(self.container, text="Noise Estimation", padding=(10, 5))
        frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        labels = ["Tip Speed", "Power", "Fan Diameter"]
        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0)
            self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

        self.noise_method_var = StringVar(value="GPSA")
        ttk.Radiobutton(frame, text="GPSA", variable=self.noise_method_var, value="GPSA").grid(row=3, column=0)
        ttk.Radiobutton(frame, text="Mukherjee", variable=self.noise_method_var, value="Mukherjee").grid(row=3, column=1)

        self.induced_var = IntVar()
        ttk.Checkbutton(frame, text="Induced", variable=self.induced_var).grid(row=4, column=0)

        ttk.Button(frame, text="Calculate Noise", command=self.calc_noise).grid(row=5, columnspan=2)

        self.noise_result_var = StringVar()
        ttk.Label(frame, text="Noise Result").grid(row=6, column=0)
        ttk.Label(frame, textvariable=self.noise_result_var).grid(row=6, column=1)

    def calc_noise(self):
        if self.noise_method_var.get() == "GPSA":
            result = ht.air_cooler.air_cooler_noise_GPSA(self.vars["Tip Speed"].get(), self.vars["Power"].get())
        else:
            result = ht.air_cooler.air_cooler_noise_Mukherjee(self.vars["Tip Speed"].get(), self.vars["Power"].get(),
                                                              self.vars["Fan Diameter"].get(), bool(self.induced_var.get()))
        self.noise_result_var.set(result)

   
    def create_pressure_drop_section(self):
        frame = ttk.LabelFrame(self.container, text="Pressure Drop Estimation", padding=(10, 5))
        frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        labels = ["Mass Flow Rate (m)", "A_min", "A_increase", "Flow Area Contraction Ratio", 
                  "Tube Diameter", "Fin Height (Only for Low Fin)", "Bare Length", "Pitch Parallel",
                  "Pitch Normal", "Tube Rows", "Density", "Viscosity"]

        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0)
            if label not in self.vars:
                self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

        self.pressure_drop_method_var = StringVar(value="High Fin")
        ttk.Radiobutton(frame, text="High Fin", variable=self.pressure_drop_method_var, value="High Fin").grid(row=len(labels), column=0)
        ttk.Radiobutton(frame, text="Low Fin", variable=self.pressure_drop_method_var, value="Low Fin").grid(row=len(labels), column=1)

        ttk.Button(frame, text="Calculate Pressure Drop", command=self.calc_pressure_drop).grid(row=len(labels) + 1, columnspan=2)

        self.pressure_drop_result_var = StringVar()
        ttk.Label(frame, text="Pressure Drop Result").grid(row=len(labels) + 2, column=0)
        ttk.Label(frame, textvariable=self.pressure_drop_result_var).grid(row=len(labels) + 2, column=1)

    def calc_pressure_drop(self):
        if self.pressure_drop_method_var.get() == "High Fin":
            result = ht.air_cooler.dP_ESDU_high_fin(
                self.vars["Mass Flow Rate (m)"].get(),
                self.vars["A_min"].get(),
                self.vars["A_increase"].get(),
                self.vars["Flow Area Contraction Ratio"].get(),
                self.vars["Tube Diameter"].get(),
                self.vars["Pitch Parallel"].get(),
                self.vars["Pitch Normal"].get(),
                self.vars["Tube Rows"].get(),
                self.vars["Density"].get(),
                self.vars["Viscosity"].get()
            )
        else:
            result = ht.air_cooler.dP_ESDU_low_fin(
                self.vars["Mass Flow Rate (m)"].get(),
                self.vars["A_min"].get(),
                self.vars["A_increase"].get(),
                self.vars["Flow Area Contraction Ratio"].get(),
                self.vars["Tube Diameter"].get(),
                self.vars["Fin Height (Only for Low Fin)"].get(),
                self.vars["Bare Length"].get(),
                self.vars["Pitch Parallel"].get(),
                self.vars["Pitch Normal"].get(),
                self.vars["Tube Rows"].get(),
                self.vars["Density"].get(),
                self.vars["Viscosity"].get()
            )
        self.pressure_drop_result_var.set(result)

    def create_heat_transfer_coeff_section(self):
        frame = ttk.LabelFrame(self.container, text="Heat Transfer Coefficient Estimation", padding=(10, 5))
        frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        labels = ["Mass Flow Rate (m)", "A", "A_min", "A_increase", "A_fin", "A_tube_showing",
                  "Tube Diameter", "Fin Diameter", "Fin Thickness", "Bare Length", "Pitch Parallel",
                  "Pitch Normal", "Tube Rows", "Density", "Cp", "Viscosity", "Thermal Conductivity (k)",
                  "Fin Conductivity (k_fin)"]

        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0)
            if label not in self.vars:
                self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

        methods = ["Briggs Young", "ESDU High Fin", "ESDU Low Fin", "Ganguli VDI"]
        self.htc_method_var = StringVar(value=methods[0])
        
        for i, method in enumerate(methods):
            ttk.Radiobutton(frame, text=method, variable=self.htc_method_var, value=method).grid(row=len(labels) + i, column=0, columnspan=2)

        ttk.Button(frame, text="Calculate Heat Transfer Coefficient", command=self.calc_htc).grid(row=len(labels) + len(methods), columnspan=2)

        self.htc_result_var = StringVar()
        ttk.Label(frame, text="Heat Transfer Coefficient Result").grid(row=len(labels) + len(methods) + 1, column=0)
        ttk.Label(frame, textvariable=self.htc_result_var).grid(row=len(labels) + len(methods) + 1, column=1)

    def calc_htc(self):
        method = self.htc_method_var.get()
        # Variables that might be used by multiple methods, fetched once for efficiency
        m = self.vars["Mass Flow Rate (m)"].get()
        A = self.vars["A"].get()
        A_min = self.vars["A_min"].get()
        A_increase = self.vars["A_increase"].get()
        A_fin = self.vars["A_fin"].get()
        A_tube_showing = self.vars["A_tube_showing"].get()
        Tube_Diameter = self.vars["Tube Diameter"].get()
        Fin_Diameter = self.vars["Fin Diameter"].get()
        Fin_Thickness = self.vars["Fin Thickness"].get()
        Bare_Length = self.vars["Bare Length"].get()
        Density = self.vars["Density"].get()
        Cp = self.vars["Cp"].get()
        Viscosity = self.vars["Viscosity"].get()
        k = self.vars["Thermal Conductivity (k)"].get()
        k_fin = self.vars["Fin Conductivity (k_fin)"].get()
        # Assuming all methods require these parameters, but you can adjust accordingly.

        if method == "Briggs Young":
            result = ht.air_cooler.h_Briggs_Young(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Density, Cp, Viscosity, k, k_fin
            )

        elif method == "ESDU High Fin":
            result = ht.air_cooler.h_ESDU_high_fin(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Density, Cp, Viscosity, k, k_fin
            )

        elif method == "ESDU Low Fin":
            # Assuming the ESDU low fin method requires an extra parameter, say "Fin_Height", but you can adjust accordingly.
            Fin_Height = self.vars["Fin Height (Only for Low Fin)"].get()
            result = ht.air_cooler.h_ESDU_low_fin(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Fin_Height, Density, Cp, Viscosity, k, k_fin
            )

        elif method == "Ganguli VDI":
            Pitch_Parallel = self.vars["Pitch Parallel"].get()
            Pitch_Normal = self.vars["Pitch Normal"].get()
            Tube_Rows = self.vars["Tube Rows"].get()
            
            # This is just an example. Assuming the Ganguli VDI method requires these parameters. 
            # Adjust accordingly based on your function's signature.
            result = ht.air_cooler.h_Ganguli_VDI(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Pitch_Parallel, Pitch_Normal, Tube_Rows, Density, Cp, Viscosity, k, k_fin
            )
        else:
            result = "Method not recognized"

        self.htc_result_var.set(result)

if __name__ == "__main__":
    app = AirCoolerApp()
    app.mainloop()

```

# apps/Thermodynamics_Air_Properties.py

```py
import chemicals.air

class AirProperties:
    def __init__(self):
        pass

    def rho(self, T, P):
        return chemicals.air.lemmon2000_rho(T, P)

    def P(self, T, rho):
        return chemicals.air.lemmon2000_P(T, rho)

    def P_dew(self, T):
        return chemicals.air.lemmon2000_air_P_dew(T)

    def P_bubble(self, T):
        return chemicals.air.lemmon2000_air_P_bubble(T)

    def rho_dew(self, T):
        return chemicals.air.lemmon2000_air_rho_dew(T)

    def rho_bubble(self, T):
        return chemicals.air.lemmon2000_air_rho_bubble(T)

    def A0(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_A0(tau, delta)

    def dA0_dtau(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_dA0_dtau(tau, delta)

    def d2A0_dtau2(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d2A0_dtau2(tau, delta)

    # ... Continuing the methods:
    def d3A0_dtau3(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d3A0_dtau3(tau, delta)

    def d4A0_dtau4(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d4A0_dtau4(tau, delta)

    def Ar(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_Ar(tau, delta)

    def dAr_dtau(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_dAr_dtau(tau, delta)

    # ... Continuing in the same pattern for all the properties:

    # (You can continue adding functions in this manner. I'll provide a pattern for a couple more, but you can continue this for all functions):

    def d2Ar_dtau2(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d2Ar_dtau2(tau, delta)

    def d3Ar_dtau3(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d3Ar_dtau3(tau, delta)

    # ... and so on for the rest of the functions ...

    # As an example for how to handle functions that don't require rho, here's a method for the Henry's law constant:
    def iapws04_Henry_air(self, T):
        return chemicals.air.iapws04_Henry_air(T)

import tkinter as tk
from tkinter import ttk, messagebox

# The previous AirProperties class code should be here

class AirPropertiesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Air Properties Calculator")
        
        # Initialize AirProperties
        self.air = AirProperties()
        
        # User Input
        ttk.Label(root, text="Temperature (K)").grid(row=0, column=0, padx=10, pady=5)
        self.temperature = ttk.Entry(root)
        self.temperature.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(root, text="Pressure (Pa) / Density (mol/m^3)").grid(row=1, column=0, padx=10, pady=5)
        self.pressure_density = ttk.Entry(root)
        self.pressure_density.grid(row=1, column=1, padx=10, pady=5)

        # Buttons for calculations
        self.calculate_rho_button = ttk.Button(root, text="Calculate Density", command=self.calculate_rho)
        self.calculate_rho_button.grid(row=2, column=0, padx=10, pady=5)

        self.calculate_P_button = ttk.Button(root, text="Calculate Pressure", command=self.calculate_P)
        self.calculate_P_button.grid(row=2, column=1, padx=10, pady=5)

        # More Buttons for calculations
        self.calculate_P_dew_button = ttk.Button(root, text="Calculate Dew Point Pressure", command=self.calculate_P_dew)
        self.calculate_P_dew_button.grid(row=4, column=0, padx=10, pady=5)

        self.calculate_P_bubble_button = ttk.Button(root, text="Calculate Bubble Point Pressure", command=self.calculate_P_bubble)
        self.calculate_P_bubble_button.grid(row=4, column=1, padx=10, pady=5)

        self.calculate_rho_dew_button = ttk.Button(root, text="Calculate Dew Point Density", command=self.calculate_rho_dew)
        self.calculate_rho_dew_button.grid(row=5, column=0, padx=10, pady=5)

        self.calculate_rho_bubble_button = ttk.Button(root, text="Calculate Bubble Point Density", command=self.calculate_rho_bubble)
        self.calculate_rho_bubble_button.grid(row=5, column=1, padx=10, pady=5)

        # Output
        self.result_label = ttk.Label(root, text="")
        self.result_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def calculate_rho(self):
        try:
            T = float(self.temperature.get())
            P = float(self.pressure_density.get())
            rho = self.air.rho(T, P)
            self.result_label.config(text=f"Density: {rho} mol/m^3")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please provide valid input values for Temperature and Pressure.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def calculate_P(self):
        try:
            T = float(self.temperature.get())
            rho = float(self.pressure_density.get())
            P = self.air.P(T, rho)
            self.result_label.config(text=f"Pressure: {P} Pa")
        except ValueError:
            messagebox.showerror("Error", "Please provide valid input values for Temperature and Density.")
    def calculate_P_dew(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            P_dew = self.air.P_dew(T)
            self.result_label.config(text=f"Dew Point Pressure: {P_dew} Pa")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")


    # Add this method to the AirPropertiesApp class:
    def validate_float_input(self, input_str):
        """Validates if the input string is a valid float."""
        try:
            if input_str:  # Check if the string is not empty
                float(input_str)
            return True
        except ValueError:
            return False
        
    def calculate_P_bubble(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            P_bubble = self.air.P_bubble(T)
            self.result_label.config(text=f"Bubble Point Pressure: {P_bubble} Pa")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def calculate_rho_dew(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            rho_dew = self.air.rho_dew(T)
            self.result_label.config(text=f"Dew Point Density: {rho_dew} mol/m^3")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def calculate_rho_bubble(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            rho_bubble = self.air.rho_bubble(T)
            self.result_label.config(text=f"Bubble Point Density: {rho_bubble} mol/m^3")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AirPropertiesApp(root)
    root.mainloop()

```

# apps/Thermodynamics_Ammonia_Gas_Storage_Sizing.py

```py
import tkinter as tk
from tkinter import ttk
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CoolPropLiquid, CEOSGas, CoolPropGas, FlashPureVLS

def calculate_backup_volume(T1, P1, P2, volume_1):
    fluid = 'ammonia'
    constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
    zs = [1]
    backend = 'HEOS'
    gas = CoolPropGas(backend, fluid, T=T1, P=1e5, zs=zs)
    liquid = CoolPropLiquid(backend, fluid, T=T1, P=1e5, zs=zs)
    flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])

    state_1 = flasher.flash(T=T1, P=P1)
    moles = volume_1/state_1.V()
    state_2 = flasher.flash(P=P2, H=state_1.H())
    
    volume_2 = moles*state_2.V()
    
    return volume_2

def on_calculate():
    T1 = float(temp_entry.get()) + 273.15
    P1 = float(p1_entry.get()) * bar
    P2 = float(p2_entry.get()) * bar
    volume_1 = float(vol1_entry.get())
    volume_2 = calculate_backup_volume(T1, P1, P2, volume_1)
    
    volume_var.set(f"Required Backup Volume: {volume_2:.2f} m^3")

# GUI setup
app = tk.Tk()
app.title("Gas Storage Tank for Ammonia")

# Input widgets
ttk.Label(app, text="Initial Volume (m^3):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
vol1_entry = ttk.Entry(app)
vol1_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(app, text="Initial Temperature (°C):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
temp_entry = ttk.Entry(app)
temp_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(app, text="Initial Pressure (bar):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
p1_entry = ttk.Entry(app)
p1_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(app, text="Backup Vessel Max Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
p2_entry = ttk.Entry(app)
p2_entry.grid(row=3, column=1, padx=10, pady=5)

ttk.Button(app, text="Calculate", command=on_calculate).grid(row=4, columnspan=2, pady=10)

# Output label
volume_var = tk.StringVar()
ttk.Label(app, textvariable=volume_var).grid(row=5, columnspan=2, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Combustion_Calculations.py

```py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import chemicals.combustion

class CombustionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Combustion Calculator")
        self.geometry("600x400")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")
        
        # Tabs for each Functionality
        self.create_stoichiometry_tab()
        self.create_heat_tab()
        self.create_ratio_tab()
        self.create_ignition_tab()
    
    def create_stoichiometry_tab(self):
        # Stoichiometry Tab
        self.stoichiometry_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stoichiometry_tab, text="Stoichiometry")
        ttk.Label(self.stoichiometry_tab, text="Enter Compound (e.g. {'C': 1, 'H':4}): ").grid(row=0, column=0, sticky="w")
        self.compound_entry = ttk.Entry(self.stoichiometry_tab)
        self.compound_entry.grid(row=0, column=1, sticky="ew")
        self.stoichiometry_btn = ttk.Button(self.stoichiometry_tab, text="Calculate", command=self.calculate_stoichiometry)
        self.stoichiometry_btn.grid(row=1, column=0, columnspan=2)
        self.stoichiometry_result = ttk.Label(self.stoichiometry_tab, text="")
        self.stoichiometry_result.grid(row=2, column=0, columnspan=2, sticky="w")

    def calculate_stoichiometry(self):
        try:
            compound = eval(self.compound_entry.get())
            result = chemicals.combustion.combustion_stoichiometry(compound)
            self.stoichiometry_result["text"] = f"Result: {result}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_heat_tab(self):
        # Heat of Combustion Tab
        self.heat_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.heat_tab, text="Heat of Combustion")
        ttk.Label(self.heat_tab, text="Enter Compound (e.g. {'C': 1, 'H':4, 'O': 1}): ").grid(row=0, column=0, sticky="w")
        self.heat_compound_entry = ttk.Entry(self.heat_tab)
        self.heat_compound_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(self.heat_tab, text="Heat of Formation (Hf): ").grid(row=1, column=0, sticky="w")
        self.hf_entry = ttk.Entry(self.heat_tab)
        self.hf_entry.grid(row=1, column=1, sticky="ew")
        self.heat_btn = ttk.Button(self.heat_tab, text="Calculate", command=self.calculate_heat)
        self.heat_btn.grid(row=2, column=0, columnspan=2)
        self.heat_result = ttk.Label(self.heat_tab, text="")
        self.heat_result.grid(row=3, column=0, columnspan=2, sticky="w")

    def calculate_heat(self):
        try:
            compound = eval(self.heat_compound_entry.get())
            Hf = float(self.hf_entry.get())
            combustion_data = chemicals.combustion.combustion_data(compound, Hf=Hf)
            self.heat_result["text"] = f"HHV: {combustion_data.HHV}, LHV: {chemicals.combustion.LHV_from_HHV(combustion_data.HHV, 2)}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def create_ratio_tab(self):
        # Fuel-to-Air Ratio Tab
        self.ratio_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ratio_tab, text="Fuel-to-Air Ratio")
        ttk.Label(self.ratio_tab, text="Enter n_fuel: ").grid(row=0, column=0, sticky="w")
        self.n_fuel_entry = ttk.Entry(self.ratio_tab)
        self.n_fuel_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(self.ratio_tab, text="Enter n_air: ").grid(row=1, column=0, sticky="w")
        self.n_air_entry = ttk.Entry(self.ratio_tab)
        self.n_air_entry.grid(row=1, column=1, sticky="ew")
        self.ratio_btn = ttk.Button(self.ratio_tab, text="Calculate", command=self.calculate_ratio)
        self.ratio_btn.grid(row=2, column=0, columnspan=2)
        self.ratio_result = ttk.Label(self.ratio_tab, text="")
        self.ratio_result.grid(row=3, column=0, columnspan=2, sticky="w")

    def calculate_ratio(self):
        try:
            n_fuel = float(self.n_fuel_entry.get())
            n_air = float(self.n_air_entry.get())
            Vm_air = 0.024936627188566596  # These are example constants; you might need to adjust or add input fields.
            Vm_fuel = 0.024880983160354486
            MW_air = 28.850334
            MW_fuel = 17.86651
            ratio = chemicals.combustion.air_fuel_ratio_solver(ratio=5.0, Vm_air=Vm_air, Vm_fuel=Vm_fuel, MW_air=MW_air, MW_fuel=MW_fuel, n_air=n_air, n_fuel=n_fuel, basis='mole')
            self.ratio_result["text"] = f"Results: {ratio}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_ignition_tab(self):
        # Ignition & Octane Ratings Tab
        self.ignition_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ignition_tab, text="Ignition & Octane Ratings")
        ttk.Label(self.ignition_tab, text="Enter CASRN: ").grid(row=0, column=0, sticky="w")
        self.casrn_entry = ttk.Entry(self.ignition_tab)
        self.casrn_entry.grid(row=0, column=1, sticky="ew")
        self.ignition_btn = ttk.Button(self.ignition_tab, text="Get Details", command=self.get_ignition_details)
        self.ignition_btn.grid(row=1, column=0, columnspan=2)
        self.ignition_result = ttk.Label(self.ignition_tab, text="")
        self.ignition_result.grid(row=2, column=0, columnspan=2, sticky="w")

    def get_ignition_details(self):
        try:
            casrn = self.casrn_entry.get()
            RON_value = chemicals.combustion.RON(CASRN=casrn)
            MON_value = chemicals.combustion.MON(CASRN=casrn)
            ignition_delay_value = chemicals.combustion.ignition_delay(CASRN=casrn)
            self.ignition_result["text"] = f"RON: {RON_value}, MON: {MON_value}, Ignition Delay: {ignition_delay_value}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = CombustionApp()
    app.mainloop()

```

# apps/Thermodynamics_Compressor_Power_Sizing.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.constants import hour
from thermo import ChemicalConstantsPackage, PRMIX, IGMIX, FlashVL, CEOSLiquid, CEOSGas
from thermo.interaction_parameters import IPDB
from scipy.integrate import quad

class CompressionPowerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Compression Power Calculator")

        ttk.Label(self, text="Molecules (comma separated):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecules_entry = ttk.Entry(self, width=50)
        self.molecules_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecules_entry.insert(0, "CO2, O2")

        ttk.Label(self, text="Molar fractions (comma separated):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.molar_fractions_entry = ttk.Entry(self, width=50)
        self.molar_fractions_entry.grid(row=1, column=1, padx=10, pady=5)
        self.molar_fractions_entry.insert(0, "0.5, 0.5")

        ttk.Label(self, text="Initial Temperature (K):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=5)
        self.temperature_entry.insert(0, "290")

        ttk.Label(self, text="Initial Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.initial_pressure_entry = ttk.Entry(self)
        self.initial_pressure_entry.grid(row=3, column=1, padx=10, pady=5)
        self.initial_pressure_entry.insert(0, "1")

        ttk.Label(self, text="Final Pressure (bar):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.final_pressure_entry = ttk.Entry(self)
        self.final_pressure_entry.grid(row=4, column=1, padx=10, pady=5)
        self.final_pressure_entry.insert(0, "5")

        ttk.Label(self, text="Flow (mol/hour):").grid(row=5, column=0, sticky=tk.W, padx=10, pady=5)
        self.flow_entry = ttk.Entry(self)
        self.flow_entry.grid(row=5, column=1, padx=10, pady=5)
        self.flow_entry.insert(0, "2000")

        self.calculate_button = ttk.Button(self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=6, column=0, columnspan=2, pady=20)

        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row=7, column=0, columnspan=2, pady=5)

    def calculate(self):
        # Extract values from the GUI
        molecules = [m.strip() for m in self.molecules_entry.get().split(',')]
        zs = [float(fraction.strip()) for fraction in self.molar_fractions_entry.get().split(',')]

        if len(molecules) != len(zs):
            messagebox.showerror("Error", "Number of molecules and fractions don't match!")
            return

        T1 = float(self.temperature_entry.get())
        P1 = float(self.initial_pressure_entry.get()) * 1e5  # Convert to Pascals
        P2 = float(self.final_pressure_entry.get()) * 1e5    # Convert to Pascals
        flow = float(self.flow_entry.get()) / hour            # Convert to mol/s

        # Your provided calculations
        constants, correlations = ChemicalConstantsPackage.from_IDs(molecules)
        kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')

        eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas,
                 kijs=kijs)
        liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)

        state_1 = flasher.flash(T=T1, P=P1, zs=zs)
        state_2 = flasher.flash(S=state_1.S(), P=P2, zs=zs)
        shaft_duty_pr = (state_2.H() - state_1.H())*flow

        liquid = CEOSLiquid(IGMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        gas = CEOSGas(IGMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        flasher_ideal = FlashVL(constants, correlations, liquid=liquid, gas=gas)

        state_1 = flasher_ideal.flash(T=T1, P=P1, zs=zs)
        state_2 = flasher_ideal.flash(S=state_1.S(), P=P2, zs=zs)
        shaft_duty_ideal = (state_2.H() - state_1.H())*flow

        self.result_label.config(text=f"Shaft power with Peng-Robinson: {shaft_duty_pr:.4f} W\nShaft power with ideal-gas: {shaft_duty_ideal:.4f} W")

app = CompressionPowerApp()
app.mainloop()

```

# apps/Thermodynamics_Critical_Properties.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import chemicals

class ChemicalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chemical Properties GUI")
        self.geometry("900x700")
        
        # Entry for CASRN
        ttk.Label(self, text="Enter CASRN:").pack(pady=10)
        self.cas_entry = ttk.Entry(self)
        self.cas_entry.pack(pady=10)

        # Entry for Common Name
        ttk.Label(self, text="Or, Enter Common Name:").pack(pady=10)
        self.common_name_entry = ttk.Entry(self)
        self.common_name_entry.pack(pady=10)
        
        # Button to Fetch and Display Data
        ttk.Button(self, text="Fetch Data", command=self.fetch_data).pack(pady=20)

        # Treeview to Display Results
        self.data_tree = ttk.Treeview(self, columns=("Property", "Value"), show="headings")
        self.data_tree.heading("Property", text="Property")
        self.data_tree.heading("Value", text="Value")
        self.data_tree.pack(pady=20, fill=tk.BOTH, expand=True)

    def fetch_data(self):
        CASRN = self.cas_entry.get().strip()
        
        # If CAS is empty, try to fetch CAS from common name
        if not CASRN:
            common_name = self.common_name_entry.get().strip()
            if not common_name:
                messagebox.showerror("Error", "Please enter either a CAS number or a common name.")
                return

            try:
                CASRN = chemicals.identifiers.CAS_from_any(common_name)
            except Exception:
                messagebox.showerror("Error", "Unable to identify the CAS number for the provided name.")
                return
        
        # Clear previous entries
        for row in self.data_tree.get_children():
            self.data_tree.delete(row)

        # Tc data
        tc_methods = chemicals.critical.Tc_methods(CASRN)
        for method in tc_methods:
            tc_value = chemicals.critical.Tc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Tc ({method})", tc_value))

        # Pc data
        pc_methods = chemicals.critical.Pc_methods(CASRN)
        for method in pc_methods:
            pc_value = chemicals.critical.Pc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Pc ({method})", pc_value))

        # Vc data
        vc_methods = chemicals.critical.Vc_methods(CASRN)
        for method in vc_methods:
            vc_value = chemicals.critical.Vc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Vc ({method})", vc_value))

        # Zc data
        zc_methods = chemicals.critical.Zc_methods(CASRN)
        for method in zc_methods:
            zc_value = chemicals.critical.Zc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Zc ({method})", zc_value))

        
    def create_sidebar(self):
        ttk.Label(self.sidebar, text="Select Property:").pack(padx=10, pady=5)
        
        properties = ['Tc', 'Pc', 'Vc', 'Zc']
        for prop in properties:
            ttk.Button(self.sidebar, text=prop, command=lambda p=prop: self.update_content(p)).pack(fill=tk.X, padx=10, pady=5)

    def update_content(self, property_selected):
        # Clear current content
        for widget in self.content.winfo_children():
            widget.destroy()
        
        ttk.Label(self.content, text=f"Inputs for {property_selected}:").pack(pady=10)
        
        # Add input fields based on the selected property
        if property_selected == 'Tc':
            # CASRN Input
            ttk.Label(self.content, text="CASRN:").pack(pady=5)
            self.casrn_entry = ttk.Entry(self.content)
            self.casrn_entry.pack(pady=5)
            
            # Method Selection
            self.tc_methods = tk.StringVar(self)
            self.tc_methods.set("HEOS")  # Default value
            methods_dropdown = ttk.OptionMenu(self.content, self.tc_methods, *chemicals.critical.Tc_all_methods)
            ttk.Label(self.content, text="Method:").pack(pady=5)
            methods_dropdown.pack(pady=5)
        
        # TODO: Add more conditions for other properties Pc, Vc, Zc, and add their respective input fields and methods dropdown
        
        # Add Plot button and Results display
        ttk.Button(self.content, text="Plot", command=self.plot_results).pack(pady=20)
        self.results_label = ttk.Label(self.content, text="Results will be displayed here...")
        self.results_label.pack(pady=20)

    def plot_results(self):
        # Here, process the selected property, method, and input values to generate the plot
        
        # Placeholder for the process
        casrn_value = self.casrn_entry.get()
        method_selected = self.tc_methods.get()
        
        # Call the respective method from chemicals.critical based on inputs and get the results
        
        # TODO: This is a placeholder; replace it with actual function call
        results = f"Results for CASRN: {casrn_value} using {method_selected}"
        self.results_label.config(text=results)
        
        # TODO: Replace this example plotting with actual plotting based on inputs and methods
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(np.linspace(0, 10), np.sin(np.linspace(0, 10)))

        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

# Execute the GUI
app = ChemicalGUI()
app.mainloop()

```

# apps/Thermodynamics_Ethylene_Expansion.py

```py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CoolPropLiquid, CEOSGas, CoolPropGas, FlashPureVLS

class TemperatureCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Temperature Change Upon Ethylene Expansion")
        self.geometry("600x300")

        # Label and Entry for Initial Temperature
        ttk.Label(self, text="Enter Initial Temperature (K):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=0, column=1, padx=10, pady=5)

        # Label and Entry for Initial Pressure
        ttk.Label(self, text="Enter Initial Pressure (bar):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.pressure1_entry = ttk.Entry(self)
        self.pressure1_entry.grid(row=1, column=1, padx=10, pady=5)

        # Label and Entry for Pressure after first valve
        ttk.Label(self, text="Enter Pressure After First Valve (bar):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.pressure2_entry = ttk.Entry(self)
        self.pressure2_entry.grid(row=2, column=1, padx=10, pady=5)

        # Label and Entry for Pressure after second valve
        ttk.Label(self, text="Enter Pressure After Second Valve (bar):").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.pressure3_entry = ttk.Entry(self)
        self.pressure3_entry.grid(row=3, column=1, padx=10, pady=5)

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate Temperatures", command=self.calculate_temperatures)
        self.calculate_btn.grid(row=4, column=0, columnspan=2, pady=20)

        # Labels to Display the Results
        self.result_label2 = ttk.Label(self, text="")
        self.result_label2.grid(row=5, column=0, columnspan=2, pady=5)

        self.result_label3 = ttk.Label(self, text="")
        self.result_label3.grid(row=6, column=0, columnspan=2, pady=5)

    def calculate_temperatures(self):
        try:
            T1 = float(self.temperature_entry.get())
            P1 = float(self.pressure1_entry.get()) * bar
            P2 = float(self.pressure2_entry.get()) * bar
            P3 = float(self.pressure3_entry.get()) * bar

            fluid = 'ethylene'
            constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
            backend = 'HEOS'
            gas = CoolPropGas(backend, fluid, T=T1, P=P1, zs=[1])
            liquid = CoolPropLiquid(backend, fluid, T=T1, P=P1, zs=[1])

            flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])

            state_1 = flasher.flash(T=T1, P=P1)
            state_2 = flasher.flash(H=state_1.H(), P=P2)
            state_3 = flasher.flash(H=state_1.H(), P=P3)

            self.result_label2["text"] = f"The temperature after the first valve is {state_2.T: .2f} K"
            self.result_label3["text"] = f"The temperature after the second valve is {state_3.T: .2f} K"

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = TemperatureCalculator()
    app.mainloop()

```

# apps/Thermodynamics_Fetch_Chemical_Data.py

```py
chemicals.dipole.dipole_moment(CASRN, method=None)
chemicals.dipole.dipole_moment_methods(CASRN)
chemicals.dipole.dipole_moment_all_methods = ('CCCBDB', 'MULLER', 'POLING', 'PSI4_2022A')¶
chemicals.dippr.EQ100(T, A=0, B=0, C=0, D=0, E=0, F=0, G=0, order=0)
chemicals.dippr.EQ101(T, A, B, C=0.0, D=0.0, E=0.0, order=0)
chemicals.dippr.EQ102(T, A, B, C=0.0, D=0.0, order=0)
chemicals.dippr.EQ104(T, A, B, C=0.0, D=0.0, E=0.0, order=0)
chemicals.dippr.EQ105(T, A, B, C, D, order=0)
chemicals.dippr.EQ106(T, Tc, A, B, C=0.0, D=0.0, E=0.0, order=0)
chemicals.dippr.EQ107(T, A=0, B=0, C=0, D=0, E=0, order=0)
chemicals.dippr.EQ114(T, Tc, A, B, C, D, order=0)
chemicals.dippr.EQ115(T, A, B, C=0, D=0, E=0, order=0)
chemicals.dippr.EQ116(T, Tc, A, B, C, D, E, order=0)
chemicals.dippr.EQ127(T, A, B, C, D, E, F, G, order=0)
chemicals.dippr.EQ101_fitting_jacobian(Ts, A, B, C, D, E)
chemicals.dippr.EQ102_fitting_jacobian(Ts, A, B, C, D)
chemicals.dippr.EQ105_fitting_jacobian(Ts, A, B, C, D)
chemicals.dippr.EQ106_fitting_jacobian(Ts, Tc, A, B, C, D, E)
chemicals.dippr.EQ107_fitting_jacobian(Ts, A, B, C, D, E)
chemicals.heat_capacity.TRCCp(T, a0, a1, a2, a3, a4, a5, a6, a7)
chemicals.heat_capacity.TRCCp_integral(T, a0, a1, a2, a3, a4, a5, a6, a7, I=0)
chemicals.heat_capacity.TRCCp_integral_over_T(T, a0, a1, a2, a3, a4, a5, a6, a7, J=0)
chemicals.heat_capacity.Shomate(T, A, B, C, D, E)
chemicals.heat_capacity.Shomate_integral(T, A, B, C, D, E)
chemicals.heat_capacity.Shomate_integral_over_T(T, A, B, C, D, E)
class chemicals.heat_capacity.ShomateRange(coeffs, Tmin, Tmax)
calculate(T)

Return heat capacity as a function of temperature.

calculate_integral(Ta, Tb)

Return the enthalpy integral of heat capacity from Ta to Tb.

calculate_integral_over_T(Ta, Tb)

Return the entropy integral of heat capacity from Ta to Tb.

calculate(T)[source]
Return heat capacity as a function of temperature.

Parameters
Tfloat
Temperature, [K]

Returns
Cpfloat
Liquid heat capacity as T, [J/mol/K]

calculate_integral(Ta, Tb)[source]
Return the enthalpy integral of heat capacity from Ta to Tb.

Parameters
Tafloat
Initial temperature, [K]

Tbfloat
Final temperature, [K]

Returns
dHfloat
Enthalpy difference between Ta and Tb, [J/mol]

calculate_integral_over_T(Ta, Tb)[source]
Return the entropy integral of heat capacity from Ta to Tb.

Parameters
Tafloat
Initial temperature, [K]

Tbfloat
Final temperature, [K]

Returns
dSfloat
Entropy difference between Ta and Tb, [J/mol/K]

chemicals.heat_capacity.Poling(T, a, b, c, d, e)
chemicals.heat_capacity.Poling_integral(T, a, b, c, d, e)
chemicals.heat_capacity.Poling_integral_over_T(T, a, b, c, d, e)
chemicals.heat_capacity.PPDS2(T, Ts, C_low, C_inf, a1, a2, a3, a4, a5)
chemicals.heat_capacity.Lastovka_Shaw(T, similarity_variable, cyclic_aliphatic=False, MW=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_integral(T, similarity_variable, cyclic_aliphatic=False, MW=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_integral_over_T(T, similarity_variable, cyclic_aliphatic=False, MW=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_T_for_Hm(Hm, MW, similarity_variable, T_ref=298.15, factor=1.0, cyclic_aliphatic=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_T_for_Sm(Sm, MW, similarity_variable, T_ref=298.15, factor=1.0, cyclic_aliphatic=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_term_A(similarity_variable, cyclic_aliphatic)
chemicals.heat_capacity.Cpg_statistical_mechanics(T, thetas, linear=False)
chemicals.heat_capacity.Cpg_statistical_mechanics_integral(T, thetas, linear=False)
chemicals.heat_capacity.Cpg_statistical_mechanics_integral_over_T(T, thetas, linear=False)
chemicals.heat_capacity.vibration_frequency_cm_to_characteristic_temperature(frequency, scale=1)
chemicals.heat_capacity.Zabransky_quasi_polynomial(T, Tc, a1, a2, a3, a4, a5, a6)
chemicals.heat_capacity.Zabransky_quasi_polynomial_integral(T, Tc, a1, a2, a3, a4, a5, a6)
chemicals.heat_capacity.Zabransky_quasi_polynomial_integral_over_T(T, Tc, a1, a2, a3, a4, a5, a6)
chemicals.heat_capacity.Zabransky_cubic(T, a1, a2, a3, a4)
chemicals.heat_capacity.Zabransky_cubic_integral(T, a1, a2, a3, a4)
chemicals.heat_capacity.Zabransky_cubic_integral_over_T(T, a1, a2, a3, a4)
class chemicals.heat_capacity.ZabranskySpline(coeffs, Tmin, Tmax)
class chemicals.heat_capacity.ZabranskyQuasipolynomial(coeffs, Tc, Tmin, Tmax)
chemicals.heat_capacity.PPDS15(T, Tc, a0, a1, a2, a3, a4, a5)
chemicals.heat_capacity.TDE_CSExpansion(T, Tc, b, a1, a2=0.0, a3=0.0, a4=0.0)
chemicals.heat_capacity.Rowlinson_Poling(T, Tc, omega, Cpgm)
chemicals.heat_capacity.Rowlinson_Bondi(T, Tc, omega, Cpgm)
chemicals.heat_capacity.Dadgostar_Shaw(T, similarity_variable, MW=None, terms=None)
chemicals.heat_capacity.Dadgostar_Shaw_integral(T, similarity_variable, MW=None, terms=None)
chemicals.heat_capacity.Dadgostar_Shaw_integral_over_T(T, similarity_variable, MW=None, terms=None)
chemicals.heat_capacity.Dadgostar_Shaw_terms(similarity_variable)
chemicals.heat_capacity.Perry_151(T, a, b, c, d)
chemicals.heat_capacity.Lastovka_solid(T, similarity_variable, MW=None)
chemicals.heat_capacity.Lastovka_solid_integral(T, similarity_variable, MW=None)
chemicals.heat_capacity.Lastovka_solid_integral_over_T(T, similarity_variable, MW=None)
class chemicals.heat_capacity.PiecewiseHeatCapacity(models)
chemicals.heat_capacity.Cp_data_Poling¶
chemicals.heat_capacity.TRC_gas_data¶
chemicals.heat_capacity.CRC_standard_data¶
chemicals.heat_capacity.Cp_dict_PerryI¶
chemicals.heat_capacity.zabransky_dicts¶
chemicals.heat_capacity.Cp_dict_characteristic_temperatures_adjusted_psi4_2022a¶
chemicals.heat_capacity.Cp_dict_characteristic_temperatures_psi4_2022a¶
chemicals.interface.Brock_Bird(T, Tb, Tc, Pc)
chemicals.interface.Pitzer_sigma(T, Tc, Pc, omega)
chemicals.interface.Sastri_Rao(T, Tb, Tc, Pc, chemicaltype=None)
chemicals.interface.Zuo_Stenby(T, Tc, Pc, omega)
chemicals.interface.Hakim_Steinberg_Stiel(T, Tc, Pc, omega, StielPolar=0.0)
chemicals.interface.Miqueu(T, Tc, Vc, omega)
chemicals.interface.Aleem(T, MW, Tb, rhol, Hvap_Tb, Cpl)
chemicals.interface.Mersmann_Kind_sigma(T, Tm, Tb, Tc, Pc, n_associated=1)
chemicals.interface.sigma_Gharagheizi_1(T, Tc, MW, omega)
chemicals.interface.sigma_Gharagheizi_2(T, Tb, Tc, Pc, Vc)
chemicals.interface.Winterfeld_Scriven_Davis(xs, sigmas, rhoms)
chemicals.interface.Weinaug_Katz(parachors, Vml, Vmg, xs, ys)
chemicals.interface.Diguilio_Teja(T, xs, sigmas_Tb, Tbs, Tcs)
chemicals.interface.sigma_IAPWS(T)
chemicals.interface.API10A32(T, Tc, K_W)
chemicals.interface.Meybodi_Daryasafar_Karimi(rho_water, rho_oil, T, Tc)
chemicals.interface.REFPROP_sigma(T, Tc, sigma0, n0, sigma1=0.0, n1=0.0, sigma2=0.0, n2=0.0)
chemicals.interface.Somayajulu(T, Tc, A, B, C)
chemicals.interface.Jasper(T, a, b)
chemicals.interface.PPDS14(T, Tc, a0, a1, a2)
chemicals.interface.Watson_sigma(T, Tc, a1, a2, a3=0.0, a4=0.0, a5=0.0)
chemicals.interface.ISTExpansion(T, Tc, a1, a2, a3=0.0, a4=0.0, a5=0.0)
All of these coefficients are lazy-loaded, so they must be accessed as an attribute of this module.

chemicals.interface.sigma_data_Mulero_Cachadina
Data from [5] with REFPROP_sigma coefficients.

chemicals.interface.sigma_data_Jasper_Lange
Data as shown in [4] but originally in [3] with Jasper coefficients.

chemicals.interface.sigma_data_Somayajulu
Data from [1] with Somayajulu coefficients.

chemicals.interface.sigma_data_Somayajulu2
Data from [2] with Somayajulu coefficients. These should be preferred over the original coefficients.

chemicals.interface.sigma_data_VDI_PPDS_11
Data from [6] with chemicals.dippr.EQ106 coefficients.
```

# apps/Thermodynamics_Isentropic_Air_Compression.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, Mixture, SRKMIX, IdealGas, CEOSLiquid, CEOSGas, FlashPureVLS

class CompressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Isentropic Compression of Fluid")

        # Molecule input
        ttk.Label(self, text="Molecules (CAS, Formula, Common Name):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecule_entry = ttk.Entry(self, width=30)
        self.molecule_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecule_entry.insert(0, "O2,N2")

        # Mole fraction input
        ttk.Label(self, text="Mole fractions:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.mole_fraction_entry = ttk.Entry(self, width=30)
        self.mole_fraction_entry.grid(row=1, column=1, padx=10, pady=5)
        self.mole_fraction_entry.insert(0, "0.21,0.79")

        # Temperature input
        ttk.Label(self, text="Temperature (°C):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.temp_entry = ttk.Entry(self, width=30)
        self.temp_entry.grid(row=2, column=1, padx=10, pady=5)
        self.temp_entry.insert(0, "15")

        # Initial pressure input
        ttk.Label(self, text="Initial Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.p1_entry = ttk.Entry(self, width=30)
        self.p1_entry.grid(row=3, column=1, padx=10, pady=5)
        self.p1_entry.insert(0, "1")

        # Final pressure input
        ttk.Label(self, text="Final Pressure (bar):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.p2_entry = ttk.Entry(self, width=30)
        self.p2_entry.grid(row=4, column=1, padx=10, pady=5)
        self.p2_entry.insert(0, "8")

        # Calculate button
        self.calc_button = ttk.Button(self, text="Calculate", command=self.calculate_values_gui)
        self.calc_button.grid(row=5, columnspan=2, pady=10)

        # Result Text Area
        self.result_text = tk.Text(self, width=50, height=10)
        self.result_text.grid(row=6, columnspan=2, padx=10, pady=5)


    def calculate_values_gui(self):
        molecule_input = self.molecule_entry.get().split(',')
        molecules = [m.strip() for m in molecule_input]
        mole_fractions = list(map(float, self.mole_fraction_entry.get().split(',')))
        
        T1 = float(self.temp_entry.get()) + 273.15
        P1 = float(self.p1_entry.get()) * bar
        P2 = float(self.p2_entry.get()) * bar

        try:
            actual_power_ideal, T2_ideal, actual_power_srk, T2_srk = calculate_values(T1, P1, P2, molecules, mole_fractions)
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ideal Gas Power: {actual_power_ideal:.4f} J/mol\n")
            self.result_text.insert(tk.END, f"Ideal Gas Outlet Temp: {T2_ideal:.2f} K\n")
            self.result_text.insert(tk.END, f"SRK Power: {actual_power_srk:.4f} J/mol\n")
            self.result_text.insert(tk.END, f"SRK Outlet Temp: {T2_srk:.2f} K\n")
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")


def calculate_values(T1, P1, P2, molecules, mole_fractions):
    fluid = Mixture(molecules, zs=mole_fractions)

    # For Ideal-Gas EOS
    Cp = fluid.Cp(T=T1)  # Molar heat capacity at constant pressure
    gamma = fluid.isentropic_exponent(T=T1)  # Heat capacity ratio
    T2_ideal = T1 * (P2 / P1)**((gamma - 1) / gamma)
    actual_power_ideal = Cp * (T1 - T2_ideal)

    # SRK
    eos_kwargs = dict(Tcs=fluid.Tcs, Pcs=fluid.Pcs, omegas=fluid.omegas)
    liquid = CEOSLiquid(SRKMIX, T=T1, P=P1, zs=mole_fractions, eos_kwargs=eos_kwargs)
    gas_srk = CEOSGas(SRKMIX, T=T1, P=P1, zs=mole_fractions, eos_kwargs=eos_kwargs)
    
    flasher_srk = FlashPureVLS(fluid, gas=gas_srk, liquids=[liquid])
    state_1_srk = flasher_srk.flash(T=T1, P=P1)
    state_2_srk = flasher_srk.flash(S=state_1_srk.S, P=P2)
    actual_power_srk = state_2_srk.H - state_1_srk.H


    return actual_power_ideal, T2_ideal, actual_power_srk, state_2_srk.T
if __name__ == "__main__":
    app = CompressionApp()
    app.mainloop()





```

# apps/Thermodynamics_Isentropic_Oxygen_Compression.py

```py
import tkinter as tk
from tkinter import ttk
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, SRKMIX, IdealGas, CEOSLiquid, CEOSGas, FlashPureVLS
from fluids import isentropic_work_compression, isentropic_T_rise_compression

def calculate_values(T1, P1, P2):
    fluid = 'oxygen'
    constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
    
    # Ideal-Gas EOS
    gas = IdealGas(HeatCapacityGases=correlations.HeatCapacityGases)
    flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[], solids=[])
    state_1_ideal = flasher.flash(T=T1, P=P1)
    state_2_ideal = flasher.flash(S=state_1_ideal.S(), P=P2)
    actual_power_ideal = (state_2_ideal.H() - state_1_ideal.H())
    
    # SRK
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    gas = CEOSGas(SRKMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
    state_1 = flasher.flash(T=T1, P=P1)
    state_2 = flasher.flash(S=state_1.S(), P=P2)
    actual_power_srk = (state_2.H() - state_1.H())
    
    return actual_power_ideal, state_2_ideal.T, actual_power_srk, state_2.T

def on_calculate():
    T1 = float(temp_entry.get()) + 273.15
    P1 = float(p1_entry.get()) * bar
    P2 = float(p2_entry.get()) * bar
    actual_power_ideal, T2_ideal, actual_power_srk, T2_srk = calculate_values(T1, P1, P2)
    
    ideal_power_var.set(f"Ideal Gas Power: {actual_power_ideal:.4f} J/mol")
    ideal_temp_var.set(f"Ideal Gas Outlet Temp: {T2_ideal:.2f} K")
    srk_power_var.set(f"SRK Power: {actual_power_srk:.4f} J/mol")
    srk_temp_var.set(f"SRK Outlet Temp: {T2_srk:.2f} K")

# GUI setup
app = tk.Tk()
app.title("Isentropic Compression of Oxygen")

# Input widgets
ttk.Label(app, text="Temperature (°C):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
temp_entry = ttk.Entry(app)
temp_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(app, text="Initial Pressure (bar):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
p1_entry = ttk.Entry(app)
p1_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(app, text="Final Pressure (bar):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
p2_entry = ttk.Entry(app)
p2_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Button(app, text="Calculate", command=on_calculate).grid(row=3, columnspan=2, pady=10)

# Output labels
ideal_power_var = tk.StringVar()
ttk.Label(app, textvariable=ideal_power_var).grid(row=4, columnspan=2, pady=5)

ideal_temp_var = tk.StringVar()
ttk.Label(app, textvariable=ideal_temp_var).grid(row=5, columnspan=2, pady=5)

srk_power_var = tk.StringVar()
ttk.Label(app, textvariable=srk_power_var).grid(row=6, columnspan=2, pady=5)

srk_temp_var = tk.StringVar()
ttk.Label(app, textvariable=srk_temp_var).grid(row=7, columnspan=2, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Joule_Thomson_Methane.py

```py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashPureVLS

class JTCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Joule-Thomson Coefficient Calculator for Methane")
        self.geometry("500x200")

        # Label and Entry for Temperature
        ttk.Label(self, text="Enter Temperature (K):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=0, column=1, padx=10, pady=5)

        # Label and Entry for Pressure
        ttk.Label(self, text="Enter Pressure (bar):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.pressure_entry = ttk.Entry(self)
        self.pressure_entry.grid(row=1, column=1, padx=10, pady=5)

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate JT Coefficient", command=self.calculate_JT)
        self.calculate_btn.grid(row=2, column=0, columnspan=2, pady=20)

        # Label to Display the Result
        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

    def calculate_JT(self):
        try:
            T = float(self.temperature_entry.get())
            P = float(self.pressure_entry.get()) * bar

            fluid = 'methane'
            constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])

            zs = [1]
            eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            flasher = FlashPureVLS(constants, correlations, liquids=[liquid], gas=gas, solids=[])

            res = flasher.flash(T=T, P=P, zs=zs)
            self.result_label["text"] = f"The JT coefficient at the specified conditions is {res.Joule_Thomson():.4g} K/Pa"

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = JTCalculator()
    app.mainloop()

```

# apps/Thermodynamics_Joule_Thomson_Nitrogen.py

```py
import tkinter as tk
from tkinter import ttk
from scipy.constants import atm
from thermo import ChemicalConstantsPackage, SRKMIX, CEOSGas, VirialCSP, VirialGas

def calculate_joule_thomson(T, P):
    fluid = 'nitrogen'
    constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
    model = VirialCSP(Tcs=constants.Tcs, Pcs=constants.Pcs, Vcs=constants.Vcs,
                      omegas=constants.omegas, B_model='VIRIAL_B_TSONOPOULOS',
                      C_model='VIRIAL_C_ZERO')
    virial_gas = VirialGas(model=model, T=T, P=P, zs=[1], HeatCapacityGases=correlations.HeatCapacityGases)
    virial_result = virial_gas.Joule_Thomson()
    
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(SRKMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    srk_result = gas.Joule_Thomson()
    
    return virial_result, srk_result

def on_calculate():
    T = float(temperature_entry.get())
    P = float(pressure_entry.get()) * atm
    virial_result, srk_result = calculate_joule_thomson(T, P)
    result_var.set(f"Virial: {virial_result:.2e} K/Pa\nSRK: {srk_result:.2e} K/Pa")

app = tk.Tk()
app.title("Joule-Thomson Coefficient of Nitrogen")

# Input widgets
ttk.Label(app, text="Temperature (K):").grid(row=0, column=0, padx=10, pady=5)
temperature_entry = ttk.Entry(app)
temperature_entry.grid(row=0, column=1, padx=10, pady=5)
temperature_entry.insert(0, '150')

ttk.Label(app, text="Pressure (atm):").grid(row=1, column=0, padx=10, pady=5)
pressure_entry = ttk.Entry(app)
pressure_entry.grid(row=1, column=1, padx=10, pady=5)
pressure_entry.insert(0, '10')

ttk.Button(app, text="Calculate", command=on_calculate).grid(row=2, columnspan=2, pady=10)

result_var = tk.StringVar()
result_label = ttk.Label(app, textvariable=result_var)
result_label.grid(row=3, columnspan=2, padx=10, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Joule_Thomson.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CoolPropLiquid, CEOSGas, CoolPropGas, FlashPureVLS
import CoolProp
class JTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Joule-Thomson Effect Calculator")

        # Molecule input
        ttk.Label(self, text="Molecule:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecule_entry = ttk.Entry(self, width=30)
        self.molecule_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecule_entry.insert(0, "nitrogen")

        # Mole fraction input (though mostly this will be 1 for pure species)
        ttk.Label(self, text="Mole fraction:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.molar_fraction_entry = ttk.Entry(self, width=30)
        self.molar_fraction_entry.grid(row=1, column=1, padx=10, pady=5)
        self.molar_fraction_entry.insert(0, "1")

        # Initial conditions input
        ttk.Label(self, text="Initial Temperature (K):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.T1_entry = ttk.Entry(self, width=30)
        self.T1_entry.grid(row=2, column=1, padx=10, pady=5)
        self.T1_entry.insert(0, "300")

        ttk.Label(self, text="Initial Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.P1_entry = ttk.Entry(self, width=30)
        self.P1_entry.grid(row=3, column=1, padx=10, pady=5)
        self.P1_entry.insert(0, "200")

        ttk.Label(self, text="Final Pressure (bar):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.P2_entry = ttk.Entry(self, width=30)
        self.P2_entry.grid(row=4, column=1, padx=10, pady=5)
        self.P2_entry.insert(0, "1")

        # Calculate button
        self.btn_calculate = ttk.Button(self, text="Calculate Outlet Temperatures", command=self.calculate)
        self.btn_calculate.grid(row=5, column=0, columnspan=2, pady=20)

        # Results display
        self.result_text = tk.Text(self, height=5, width=40)
        self.result_text.grid(row=6, column=0, columnspan=2, padx=10, pady=5)
        self.result_text.insert(tk.END, "Results will be displayed here...")

    def calculate(self):
        fluid_input = self.molecule_entry.get().replace(" ", "").split(",")
        zs_input = self.molar_fraction_entry.get().replace(" ", "").split(",")

        if len(fluid_input) != len(zs_input):
            messagebox.showerror("Error", "Number of molecules and mole fractions do not match.")
            return

        try:
            zs = [float(z) for z in zs_input]
            constants, correlations = ChemicalConstantsPackage.from_IDs(fluid_input)  # Use fluid_input here

            T1 = float(self.T1_entry.get())
            P1 = float(self.P1_entry.get()) * bar
            P2 = float(self.P2_entry.get()) * bar

            # Calculate using high precision
            backend = 'HEOS'
            # Determine if it's a mixture or pure fluid
            if len(fluid_input) > 1:  # Mixture
                fluid_string = '&'.join([f"{f}[%] = {100*z}" for f, z in zip(fluid_input, zs)])
                H1 = CoolProp.PropsSI('H', 'T', T1, 'P', P1, fluid_string)
                T2_precise = CoolProp.PropsSI('T', 'H', H1, 'P', P2, fluid_string)
            else:  # Pure fluid
                gas = CoolPropGas(backend, fluid_input[0], T=T1, P=P1, zs=zs)
                liquid = CoolPropLiquid(backend, fluid_input[0], T=T1, P=P1, zs=zs)
                flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
                state_1 = flasher.flash(T=T1, P=P1)
                state_2 = flasher.flash(H=state_1.H(), P=P2)
                T2_precise = state_2.T
            flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
            state_1 = flasher.flash(T=T1, P=P1)
            state_2 = flasher.flash(H=state_1.H(), P=P2)
            T2_precise = state_2.T

            # Calculate using Peng-Robinson
            eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
            state_1 = flasher.flash(T=T1, P=P1)
            state_2 = flasher.flash(H=state_1.H(), P=P2)
            T2_PR = state_2.T

            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Outlet Temp (High Precision): {T2_precise:.2f} K\n")
            self.result_text.insert(tk.END, f"Outlet Temp (Peng-Robinson): {T2_PR:.2f} K\n")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")


app = JTApp()
app.mainloop()

```

# apps/Thermodynamics_Liq_Water_Thermal_Expansion.py

```py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class WaterVolumeChangeCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Volume Change of Water Calculator")
        self.geometry("600x250")

        # Label and Entry for Initial Temperature
        ttk.Label(self, text="Enter Initial Temperature (°C):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.init_temp_entry = ttk.Entry(self)
        self.init_temp_entry.grid(row=0, column=1, padx=10, pady=5)

        # Label and Entry for Final Temperature
        ttk.Label(self, text="Enter Final Temperature (°C):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.final_temp_entry = ttk.Entry(self)
        self.final_temp_entry.grid(row=1, column=1, padx=10, pady=5)

        # Label and Entry for Initial Volume
        ttk.Label(self, text="Enter Initial Volume (L):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.init_vol_entry = ttk.Entry(self)
        self.init_vol_entry.grid(row=2, column=1, padx=10, pady=5)

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate Volume Change", command=self.calculate_volume_change)
        self.calculate_btn.grid(row=3, column=0, columnspan=2, pady=20)

        # Label to Display the Result
        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def calculate_volume_change(self):
        try:
            T_initial = float(self.init_temp_entry.get())
            T_final = float(self.final_temp_entry.get())
            V_initial = float(self.init_vol_entry.get())

            beta = 0.000207  # coefficient of volume expansion for water in 1/°C
            delta_T = T_final - T_initial

            delta_V = V_initial * beta * delta_T

            self.result_label["text"] = f"The change in volume due to the temperature change is: {delta_V:.4g} L.\n The final volume is: {V_initial + delta_V:.4g} L."

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values!")

if __name__ == "__main__":
    app = WaterVolumeChangeCalculator()
    app.mainloop()

```

# apps/Thermodynamics_Liquid_Nitrogen_Production.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from thermo import *
from thermo.interaction_parameters import SPDB

class LiquidNitrogenCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Liquid Nitrogen Production Via Volume Expansion of the Compressed Gas")
        self.geometry("800x400")

        # Initial Temperature
        ttk.Label(self, text="Enter Initial Temperature (°C):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=0, column=1, padx=10, pady=5)

        # Initial Pressure
        ttk.Label(self, text="Enter Initial Pressure (bar):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.pressure1_entry = ttk.Entry(self)
        self.pressure1_entry.grid(row=1, column=1, padx=10, pady=5)

        # Pressure after Valve
        ttk.Label(self, text="Enter Pressure After Valve (bar):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.pressure2_entry = ttk.Entry(self)
        self.pressure2_entry.grid(row=2, column=1, padx=10, pady=5)

        # EOS selection
        ttk.Label(self, text="Select EOS:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.eos_combobox = ttk.Combobox(self, values=[
            "High-Precision", "PR-Pina-Martinez", "SRK-Pina-Martinez", "PR", "SRK", 
            "VDW", "PRSV", "PRSV2", "TWUPR", "TWUSRK", 
            "PRTranslatedConsistent", "SRKTranslatedConsistent"])
        self.eos_combobox.grid(row=3, column=1, padx=10, pady=5)
        self.eos_combobox.set("High-Precision")

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate Liquid Fraction", command=self.calculate_fraction)
        self.calculate_btn.grid(row=4, column=0, columnspan=2, pady=20)

        # Result Display
        self.result_label = ttk.Label(self, text="", font=("Arial", 14))
        self.result_label.grid(row=5, column=0, columnspan=2, pady=5)

    def calculate_fraction(self):
        T1 = float(self.temperature_entry.get()) + 273.15
        P1 = float(self.pressure1_entry.get()) * 1e5
        P2 = float(self.pressure2_entry.get()) * 1e5
        eos_selection = self.eos_combobox.get()

        # Perform calculations using the Thermo library
        try:
            fluid = 'nitrogen'
            constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
            zs = [1]

            flasher = None
            if eos_selection == "High-Precision":
                gas = CoolPropGas('HEOS', fluid, T=T1, P=P1, zs=zs)
                liquid = CoolPropLiquid('HEOS', fluid, T=T1, P=P1, zs=zs)
                flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])

            # ... (You can add code for other EOSs in a similar way)

            if flasher:
                state_1 = flasher.flash(T=T1, P=P1, zs=zs)
                state_2 = flasher.flash(P=P2, H=state_1.H(), zs=zs)
                self.result_label["text"] = f"The {eos_selection} EOS predicted liquid molar fraction is {state_2.LF:.8f}."
            else:
                self.result_label["text"] = f"The EOS {eos_selection} is not supported yet."

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = LiquidNitrogenCalculator()
    app.mainloop()

```

# apps/Thermodynamics_Material_Properties.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from ht import *

def fetch_material_properties():
    try:
        material_name = material_entry.get()
        temperature = float(temp_entry.get())
        
        material = nearest_material(material_name)

        if not material:
            result_var.set("No matching material found.")
            return

        k = k_material(material, T=temperature)
        rho = rho_material(material)
        Cp = Cp_material(material, T=temperature)

        result_var.set(f"Material: {material}\n"
                       f"Thermal Conductivity: {k} W/mK\n"
                       f"Density: {rho} kg/m^3\n"
                       f"Heat Capacity: {Cp} J/kgK")

    except Exception as e:
        result_var.set("Error!")
        messagebox.showerror("Error", str(e))

app = tk.Tk()
app.title("Material Property Fetcher")

# Material Input field
ttk.Label(app, text="Enter Material:").grid(row=0, column=0, padx=10, pady=5)
material_entry = ttk.Entry(app)
material_entry.grid(row=0, column=1, padx=10, pady=5)

# Temperature Input field
ttk.Label(app, text="Enter Temperature (K):").grid(row=1, column=0, padx=10, pady=5)
temp_entry = ttk.Entry(app)
temp_entry.insert(0, "298.15")  # Default temperature
temp_entry.grid(row=1, column=1, padx=10, pady=5)

# Fetch button
ttk.Button(app, text="Fetch Properties", command=fetch_material_properties).grid(row=2, columnspan=2, pady=10)

# Output label
result_var = tk.StringVar()
result_label = ttk.Label(app, textvariable=result_var, font=("Arial", 10), anchor="w", justify=tk.LEFT)
result_label.grid(row=3, columnspan=2, padx=10, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Multi_Component_Flash.py

```py
import tkinter as tk
from tkinter import ttk
from thermo import *
from scipy.constants import atm
import numpy as np

class ThermoApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Thermo-Scipy GUI")
        self.geometry("600x450")

        # Input Fields
        self.T_label = ttk.Label(self, text="Temperature (K):")
        self.T_entry = ttk.Entry(self)
        
        self.P_label = ttk.Label(self, text="Pressure (Pa):")
        self.P_entry = ttk.Entry(self)
        
        self.zs_label = ttk.Label(self, text="Molar Fractions (comma separated):")
        self.zs_entry = ttk.Entry(self, width=50)
        
        # Calculate Button
        self.calc_btn = ttk.Button(self, text="Calculate", command=self.calculate)
        
        # Output Labels
        self.output_label = ttk.Label(self, text="Results:")
        self.phases_label = ttk.Label(self)
        self.densities_label = ttk.Label(self)
        self.fugacity_label = ttk.Label(self)
        
        # Layout
        self.T_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.T_entry.grid(row=0, column=1, padx=10, pady=5)

        self.P_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.P_entry.grid(row=1, column=1, padx=10, pady=5)

        self.zs_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.zs_entry.grid(row=2, column=1, padx=10, pady=5)
        
        self.calc_btn.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.output_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.phases_label.grid(row=5, column=0, columnspan=2, padx=10)
        self.densities_label.grid(row=6, column=0, columnspan=2, padx=10)
        self.fugacity_label.grid(row=7, column=0, columnspan=2, padx=10)
        
    def calculate(self):
        # Get inputs
        T = float(self.T_entry.get())
        P = float(self.P_entry.get())
        zs = [float(z) for z in self.zs_entry.get().split(",")]

        # The calculation based on the provided code
        pure_constants = ChemicalConstantsPackage.constants_from_IDs(
            ['methane', 'ethane', 'propane', 'n-butane', 'n-pentane', 'n-hexane'])

        pseudos = ChemicalConstantsPackage(Tcs=[606.28,825.67], Pcs=[25.42*atm, 14.39*atm],
                                           omegas=[0.4019, 0.7987], MWs=[140.0, 325.0])
        constants = pure_constants + pseudos

        properties = PropertyCorrelationsPackage(constants=constants)
        
        kijs = [[0.0, 0.002, 0.017, 0.015, 0.02, 0.039, 0.05, 0.09],
                [0.002, 0.0, 0.0, 0.025, 0.01, 0.056, 0.04, 0.055],
                [0.017, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01],
                [0.015, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.039, 0.056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.05, 0.04, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.09, 0.055, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]]

        eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

        gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
        liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
        liq2 = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

        flashN = FlashVLN(constants, properties, liquids=[liq, liq2], gas=gas)
        res = flashN.flash(T=T, P=P, zs=zs)

        # Set the output to the labels
        self.phases_label.config(text=f"There are {res.phase_count} phases present")
        self.densities_label.config(text=f"Mass densities of each liquid are {res.liquid0.rho_mass()} and {res.liquid0.rho_mass()} kg/m^3")
        max_fugacity_err = np.max(np.abs(1-np.array(res.liquid0.fugacities())/res.liquid1.fugacities()))
        self.fugacity_label.config(text=f"The maximum relative difference in fugacity is {max_fugacity_err:.8f}.")

if __name__ == "__main__":
    app = ThermoApp()
    app.mainloop()

```

# apps/Thermodynamics_Petroleum_Psudocomponent_Physical_Properties.py

```py
import tkinter as tk
from tkinter import ttk, StringVar
from math import log
from thermo import ChemicalConstantsPackage
from scipy.constants import psi
from math import log, exp
import numpy as np
from scipy.constants import psi
from thermo import *
from chemicals import *

# ... (Include all the functions you provided above: Tc_Kesler_Lee_SG_Tb, Pc_Kesler_Lee_SG_Tb, etc.)
def Tc_Kesler_Lee_SG_Tb(SG, Tb):
    r'''Estimates critical temperature of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_.

    .. math::
        T_c = 341.7 + 811.1SG + [0.4244 + 0.1174SG]T_b
        + \frac{[0.4669 - 3.26238SG]10^5}{T_b}

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]

    Returns
    -------
    Tc : float
        Estimated critical temperature [K]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R.

    >>> Tc_Kesler_Lee_SG_Tb(0.7365, 365.555)
    545.0124354151242

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    Tb = 9/5.*Tb # K to R
    Tc = 341.7 + 811.1*SG + (0.4244 + 0.1174*SG)*Tb + ((0.4669 - 3.26238*SG)*1E5)/Tb
    Tc = 5/9.*Tc # R to K
    return Tc

def Pc_Kesler_Lee_SG_Tb(SG, Tb):
    r'''Estimates critical pressure of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_.

    .. math::
        \ln(P_c) = 8.3634 - \frac{0.0566}{SG} - \left[0.24244 + \frac{2.2898}
        {SG} + \frac{0.11857}{SG^2}\right]10^{-3}T_b
        + \left[1.4685 + \frac{3.648}{SG} + \frac{0.47227}{SG^2}\right]
        10^{-7}T_b^2-\left[0.42019 + \frac{1.6977}{SG^2}\right]10^{-10}T_b^3

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]

    Returns
    -------
    Pc : float
        Estimated critical pressure [Pa]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine and psi.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R and Pa instead of psi.

    >>> Pc_Kesler_Lee_SG_Tb(0.7365, 365.555)
    3238323.346840464

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    Tb = 9/5.*Tb # K to R
    Pc = exp(8.3634 - 0.0566/SG - (0.24244 + 2.2898/SG + 0.11857/SG**2)*1E-3*Tb
    + (1.4685 + 3.648/SG + 0.47227/SG**2)*1E-7*Tb**2
    -(0.42019 + 1.6977/SG**2)*1E-10*Tb**3)
    Pc = Pc*psi
    return Pc

def MW_Kesler_Lee_SG_Tb(SG, Tb):
    r'''Estimates molecular weight of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_.

    .. math::
        MW = -12272.6 + 9486.4SG + [4.6523 - 3.3287SG]T_b + [1-0.77084SG
        - 0.02058SG^2]\left[1.3437 - \frac{720.79}{T_b}\right]\frac{10^7}{T_b}
        + [1-0.80882SG + 0.02226SG^2][1.8828 - \frac{181.98}{T_b}]
        \frac{10^{12}}{T_b^3}

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]

    Returns
    -------
    MW : float
        Estimated molecular weight [g/mol]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R and Pa instead of psi.

    >>> MW_Kesler_Lee_SG_Tb(0.7365, 365.555)
    98.70887589833501

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    Tb = 9/5.*Tb # K to R
    MW = (-12272.6 + 9486.4*SG + (4.6523 - 3.3287*SG)*Tb + (1.-0.77084*SG - 0.02058*SG**2)*
    (1.3437 - 720.79/Tb)*1E7/Tb + (1.-0.80882*SG + 0.02226*SG**2)*
    (1.8828 - 181.98/Tb)*1E12/Tb**3)
    return MW

def omega_Kesler_Lee_SG_Tb_Tc_Pc(SG, Tb, Tc=None, Pc=None):
    r'''Estimates accentric factor of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_. If Tc and Pc are provided, the Kesler-Lee
    routines for estimating them are not used.

    For Tbr > 0.8:
    .. math::
        \omega = -7.904 + 0.1352K - 0.007465K^2 + 8.359T_{br}
        + ([1.408-0.01063K]/T_{br})

    Otherwise:
    .. math::
        \omega = \frac{-\ln\frac{P_c}{14.7} - 5.92714 + \frac{6.09648}{T_{br}}
        + 1.28862\ln T_{br} - 0.169347T_{br}^6}{15.2518 - \frac{15.6875}{T_{br}}
         - 13.4721\ln T_{br} + 0.43577T_{br}^6}

        K = \frac{T_b^{1/3}}{SG}

        T_{br} = \frac{T_b}{T_c}

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]
    Tc : float, optional
        Estimated critical temperature [K]
    Pc : float, optional
        Estimated critical pressure [Pa]

    Returns
    -------
    omega : float
        Acentric factor [-]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine and psi.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R and Pa instead of psi.

    >>> omega_Kesler_Lee_SG_Tb_Tc_Pc(0.7365, 365.555, 545.012, 3238323.)
    0.306392118159797

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    if Tc is None:
        Tc = Tc_Kesler_Lee_SG_Tb(SG, Tb)
    if Pc is None:
        Pc = Pc_Kesler_Lee_SG_Tb(SG, Tb)
    Tb = 9/5.*Tb # K to R
    Tc = 9/5.*Tc # K to R
    K = Tb**(1/3.)/SG
    Tbr = Tb/Tc
    if Tbr > 0.8:
        omega = -7.904 + 0.1352*K - 0.007465*K**2 + 8.359*Tbr + ((1.408-0.01063*K)/Tbr)
    else:
        omega = ((-log(Pc/101325.) - 5.92714 + 6.09648/Tbr + 1.28862*log(Tbr)
        - 0.169347*Tbr**6) / (15.2518 - 15.6875/Tbr - 13.4721*log(Tbr) +0.43577*Tbr**6))
    return omega

def calculate_properties():
    # Retrieve values from the entry widgets
    SG = float(sg_var.get())
    Tb = float(tb_var.get())
    
    # Calculate properties
    Tc = Tc_Kesler_Lee_SG_Tb(SG, Tb)
    Pc = Pc_Kesler_Lee_SG_Tb(SG, Tb)
    MW = MW_Kesler_Lee_SG_Tb(SG, Tb)
    omega = omega_Kesler_Lee_SG_Tb_Tc_Pc(SG, Tb, Tc, Pc)
    
    # Update the results in the output labels
    tc_result_var.set(f"Tc: {Tc:.2f} K")
    pc_result_var.set(f"Pc: {Pc:.2f} Pa")
    mw_result_var.set(f"MW: {MW:.2f} g/mol")
    omega_result_var.set(f"ω: {omega:.4f}")

# Create the main application window
app = tk.Tk()
app.title("Petroleum Pseudocomponents Properties Calculator")

# Create and place the input labels and fields
sg_label = ttk.Label(app, text="Specific Gravity:")
sg_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
sg_var = StringVar()
sg_entry = ttk.Entry(app, textvariable=sg_var)
sg_entry.grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)

tb_label = ttk.Label(app, text="Boiling Point (K):")
tb_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
tb_var = StringVar()
tb_entry = ttk.Entry(app, textvariable=tb_var)
tb_entry.grid(column=1, row=1, sticky=tk.EW, padx=5, pady=5)

# Create and place the button to perform the calculations
calculate_button = ttk.Button(app, text="Calculate", command=calculate_properties)
calculate_button.grid(column=0, row=2, columnspan=2, pady=10)

# Create and place labels to display the results
tc_result_var = StringVar()
tc_result_label = ttk.Label(app, textvariable=tc_result_var)
tc_result_label.grid(column=0, row=3, columnspan=2, sticky=tk.W, padx=5, pady=5)

pc_result_var = StringVar()
pc_result_label = ttk.Label(app, textvariable=pc_result_var)
pc_result_label.grid(column=0, row=4, columnspan=2, sticky=tk.W, padx=5, pady=5)

mw_result_var = StringVar()
mw_result_label = ttk.Label(app, textvariable=mw_result_var)
mw_result_label.grid(column=0, row=5, columnspan=2, sticky=tk.W, padx=5, pady=5)

omega_result_var = StringVar()
omega_result_label = ttk.Label(app, textvariable=omega_result_var)
omega_result_label.grid(column=0, row=6, columnspan=2, sticky=tk.W, padx=5, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Plot_Txy_Pxy.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from thermo import *
from thermo.unifac import DOUFSG, DOUFIP2016
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DiagramApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Txy, Pxy, and xy Diagrams")

        # Molecules input
        ttk.Label(self, text="Molecules (comma separated):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecules_entry = ttk.Entry(self, width=50)
        self.molecules_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecules_entry.insert(0, "ethanol, water")

        # Mole Fractions input
        ttk.Label(self, text="Molar fractions (comma separated):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.molar_fractions_entry = ttk.Entry(self, width=50)
        self.molar_fractions_entry.grid(row=1, column=1, padx=10, pady=5)
        self.molar_fractions_entry.insert(0, "0.5, 0.5")

        # Temperature or Pressure input
        ttk.Label(self, text="Temperature (K):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self, width=30)
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=5)
        self.temperature_entry.insert(0, "373")

        ttk.Label(self, text="Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.pressure_entry = ttk.Entry(self, width=30)
        self.pressure_entry.grid(row=3, column=1, padx=10, pady=5)
        self.pressure_entry.insert(0, "1")

        # Plot buttons
        self.btn_Txy = ttk.Button(self, text="Plot Txy", command=self.plot_Txy)
        self.btn_Txy.grid(row=4, column=0, pady=20)
        
        self.btn_Pxy = ttk.Button(self, text="Plot Pxy", command=self.plot_Pxy)
        self.btn_Pxy.grid(row=4, column=1, pady=20)

        # Add Plot buttons for the two xy diagrams
        self.btn_xy_vary_P = ttk.Button(self, text="Plot xy (vary P)", command=self.plot_xy_vary_P)
        self.btn_xy_vary_P.grid(row=4, column=3, pady=20)

        self.btn_xy_vary_T = ttk.Button(self, text="Plot xy (vary T)", command=self.plot_xy_vary_T)
        self.btn_xy_vary_T.grid(row=4, column=4, pady=20)
        
    def get_flasher(self):
        # Extract values from the GUI
        molecules = [m.strip() for m in self.molecules_entry.get().split(',')]
        zs = [float(fraction.strip()) for fraction in self.molar_fractions_entry.get().split(',')]

        if len(molecules) != len(zs):
            messagebox.showerror("Error", "Number of molecules and fractions don't match!")
            return None

        # Load constants and properties
        constants, properties = ChemicalConstantsPackage.from_IDs(molecules)

        # Configure the activity model
        GE = UNIFAC.from_subgroups(chemgroups=constants.UNIFAC_Dortmund_groups, version=1, T=300, xs=zs,
                                interaction_data=DOUFIP2016, subgroups=DOUFSG)

        # Configure the liquid model with activity coefficients
        liquid = GibbsExcessLiquid(
            VaporPressures=properties.VaporPressures,
            HeatCapacityGases=properties.HeatCapacityGases,
            VolumeLiquids=properties.VolumeLiquids,
            GibbsExcessModel=GE,
            equilibrium_basis='Psat', caloric_basis='Psat',
            T=300, P=1e5, zs=zs)

        # Use Peng-Robinson for the vapor phase
        eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
        gas = CEOSGas(PRMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=eos_kwargs)

        # Create a flasher instance
        flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
        return flasher

    def plot_Txy(self):
        flasher = self.get_flasher()
        if not flasher:
            return

        P = float(self.pressure_entry.get()) * 1e5  # Convert to Pascals
        T, x, y = flasher.plot_Txy(P=P, pts=100)

        # Use matplotlib for external plotting
        plt.figure()
        plt.plot(x, T, label="Liquid Phase")
        plt.plot(y, T, label="Vapor Phase")
        plt.title("Txy Diagram")
        plt.xlabel("Composition")
        plt.ylabel("Temperature (K)")
        plt.legend()
        plt.show()

    def plot_Pxy(self):
        flasher = self.get_flasher()
        if not flasher:
            return

        T = float(self.temperature_entry.get())
        P, x, y = flasher.plot_Pxy(T=T, pts=100)

        plt.figure()
        plt.plot(x, P, label="Liquid Phase")
        plt.plot(y, P, label="Vapor Phase")
        plt.title("Pxy Diagram")
        plt.xlabel("Composition")
        plt.ylabel("Pressure (Pa)")
        plt.legend()
        plt.show()

    def plot_xy_vary_P(self):
        flasher = self.get_flasher()
        if not flasher:
            return

        T = float(self.temperature_entry.get())
        P, x, y = flasher.plot_xy(T=T, pts=100)  # Vary P at the specified T

        plt.figure()
        plt.plot(x, y, label="xy Curve (vary P)")
        plt.title("xy Diagram (varying P)")
        plt.xlabel("Liquid Composition")
        plt.ylabel("Vapor Composition")
        plt.legend()
        plt.show()

    def plot_xy_vary_T(self):
        flasher = self.get_flasher()
        if not flasher:
            return

        P = float(self.pressure_entry.get()) * 1e5  # Convert to Pascals
        P, x, y = flasher.plot_xy(P=P, pts=100)  # Vary T at the specified P

        plt.figure()
        plt.plot(x, y, label="xy Curve (vary T)")
        plt.title("xy Diagram (varying T)")
        plt.xlabel("Liquid Composition")
        plt.ylabel("Vapor Composition")
        plt.legend()
        plt.show()


app = DiagramApp()
app.mainloop()
```

# apps/Thermodynamics_R134a_Compressor_Sizing.py

```py
import tkinter as tk
from tkinter import ttk
from scipy.constants import bar, hour
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CoolPropLiquid, CEOSGas, CoolPropGas, FlashPureVLS

def calculate_values(T1, VF1, P2, eta_isentropic, eta_mechanical, flow_rate):
    fluid = 'R134a'
    constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
    zs = [1]
    backend = 'HEOS'
    gas = CoolPropGas(backend, fluid, T=T1, P=1e5, zs=zs)
    liquid = CoolPropLiquid(backend, fluid, T=T1, P=1e5, zs=zs)
    flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])

    state_1 = flasher.flash(T=T1, VF=VF1)
    state_2_ideal = flasher.flash(S=state_1.S(), P=P2)
    delta_H_ideal = (state_2_ideal.H()-state_1.H())
    H_added_to_fluid_actual = delta_H_ideal/eta_isentropic
    state_2 = flasher.flash(H=state_1.H() + H_added_to_fluid_actual, P=P2)

    actual_power_per_kg = (state_2.H_mass() - state_1.H_mass())/(eta_mechanical)
    actual_power = actual_power_per_kg * flow_rate/hour
    
    return actual_power, state_2.T

def on_calculate():
    T1 = float(temp_entry.get()) + 273.15
    P2 = float(p2_entry.get()) * bar
    eta_isentropic = float(eta_isen_entry.get())
    eta_mechanical = float(eta_mech_entry.get())
    flow_rate = float(flow_rate_entry.get())
    actual_power, T2 = calculate_values(T1, 1, P2, eta_isentropic, eta_mechanical, flow_rate)
    
    power_var.set(f"Actual Power: {actual_power:.0f} W")
    temp_var.set(f"Outlet Temperature: {T2:.2f} K")

# GUI setup
app = tk.Tk()
app.title("R134a Compression using High Precision EOS")

# Input widgets
ttk.Label(app, text="Initial Temperature (°C):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
temp_entry = ttk.Entry(app)
temp_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(app, text="Outlet Pressure (bar):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
p2_entry = ttk.Entry(app)
p2_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(app, text="Isentropic Efficiency:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
eta_isen_entry = ttk.Entry(app)
eta_isen_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(app, text="Mechanical Efficiency:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
eta_mech_entry = ttk.Entry(app)
eta_mech_entry.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(app, text="Mass Flow Rate (kg/hr):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
flow_rate_entry = ttk.Entry(app)
flow_rate_entry.grid(row=4, column=1, padx=10, pady=5)

ttk.Button(app, text="Calculate", command=on_calculate).grid(row=5, columnspan=2, pady=10)

# Output labels
power_var = tk.StringVar()
ttk.Label(app, textvariable=power_var).grid(row=6, columnspan=2, pady=5)

temp_var = tk.StringVar()
ttk.Label(app, textvariable=temp_var).grid(row=7, columnspan=2, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Radiation_Designer.py

```py
import tkinter as tk
from tkinter import ttk, messagebox
from ht.radiation import *
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

def plot_blackbody_spectral_radiance(T):
    wavelengths = np.linspace(1e-9, 3e-6, 1000)  # Wavelength range: 1nm to 3um
    intensities = [blackbody_spectral_radiance(T, wl) for wl in wavelengths]
    fig, ax = plt.subplots()
    ax.plot(wavelengths, intensities, label=f"T = {T} K")
    ax.set_xlabel("Wavelength (m)")
    ax.set_ylabel("Spectral Radiance (W/m^2)")
    ax.legend()
    ax.set_title("Blackbody Spectral Radiance")

    # Embedding the plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=6, columnspan=2, padx=10, pady=10)
    canvas.draw()

def plot_blackbody_spectrum(T):
    wavelengths = np.linspace(0, 2e-6, 1000)  # Wavelength range: 0 to 2000nm
    intensities = []
    for wl in wavelengths:
        if wl < 1e-9:
            intensities.append(0)
        else:
            intensities.append(blackbody_spectral_radiance(T, wl))

    # Check if a figure exists, if not create one
    if not plt.get_fignums():
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
        ax = fig.gca()

    ax.plot(wavelengths, intensities, label=f"T = {T} K")
    ax.set_xlabel("Wavelength (m)")
    ax.set_ylabel("Spectral Radiance (W/m^2)")
    ax.legend()
    ax.set_title("Blackbody Spectral Radiance")

    # Embedding the plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=6, columnspan=2, padx=10, pady=10)
    canvas.draw()


def fetch_blackbody_characteristics():
    try:
        T_values = [float(T) for T in entry_1.get().split(',')]
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant

        results = []  # List to hold the results for all temperatures

        for T in T_values:
            # Calculating intensity
            E = sigma * T**4

            # Calculating luminosity if radius is provided
            L_str = ""
            R_value = entry_2.get().strip()
            if R_value:  # if radius is not empty
                R = float(R_value)
                if R <= 0:
                    raise ValueError("Radius must be a positive value.")
                L = 4 * math.pi * R**2 * sigma * T**4
                L_str = f"\nLuminosity (L) [Assuming sphere]: {L} W"

            # Calculating peak wavelength using Wien's Law
            lambda_max = 0.29e-2 / T  # cm to meter conversion
            
            result = (f"For T = {T}K:\nIntensity (E): {E} W/m^2{L_str}"
                      f"\nPeak Wavelength (λ_max): {lambda_max:.2e} m")
            results.append(result)

            plot_blackbody_spectrum(T)

        # Set the result_var to the joined results for all temperatures
        result_var.set("\n\n".join(results))

    except ValueError as ve:  # Catch ValueError for issues with conversions
        result_var.set("Error!")
        messagebox.showerror("Error", str(ve))  # use ve instead of e to provide a specific error message.
    except Exception as e:
        result_var.set("Error!")
        messagebox.showerror("Error", str(e))
def fetch_radiation_data():
    result = ""  # Add this line
    try:
        option = selected_option.get()
        if option == "Blackbody Spectral Radiance":
            T = float(entry_1.get())
            result = blackbody_spectral_radiance(T, float(entry_2.get()))
            plot_blackbody_spectral_radiance(T)
        elif option == "Grey Transmittance":
            extinction_coefficient = float(entry_1.get())
            molar_density = float(entry_2.get())
            length = float(entry_3.get())
            result = grey_transmittance(extinction_coefficient, molar_density, length)
        elif option == "Radiant Heat Flux":
            emissivity = float(entry_1.get())
            T = float(entry_2.get())
            T2 = float(entry_3.get() or 0)
            result = q_rad(emissivity, T, T2)
        elif option == "Solar Spectrum":
            wavelengths, SSI, uncertainties = solar_spectrum()
            result = f"Min Wavelength: {min(wavelengths)}\nMax Wavelength: {max(wavelengths)}\nMin SSI: {min(SSI)}\nMax SSI: {max(SSI)}"
        # In the fetch_radiation_data function, add:
        elif option == "Blackbody Characteristics":
            fetch_blackbody_characteristics()
        else:
            result = "Invalid selection."

        result_var.set(result)

    except Exception as e:
        result_var.set("Error!")
        messagebox.showerror("Error", str(e))
def update_blackbody_input_fields(*args):
    label_1.config(text="Temperature [K] (comma-separated for multiple)")
    label_2.config(text="Radius [m] (Optional for sphere assumption)")
    label_3.config(text="")
    
def update_input_fields(*args):
    option = selected_option.get()
    label_1.config(text="")
    label_2.config(text="")
    label_3.config(text="")
    entry_1.delete(0, tk.END)
    entry_2.delete(0, tk.END)
    entry_3.delete(0, tk.END)
    
    description_var.set("")  # Initially set the description to empty
    
    # Only show the labels and entries when needed
    label_1.grid_remove()
    label_2.grid_remove()
    label_3.grid_remove()
    entry_1.grid_remove()
    entry_2.grid_remove()
    entry_3.grid_remove()
    
    if option == "Blackbody Spectral Radiance":
        label_1.config(text="Temperature [K]")
        label_2.config(text="Wavelength [m]")
        description_var.set("Blackbody spectral radiance provides the emitted energy by a black body at a specific temperature and wavelength.")
        
        label_1.grid(row=1, column=0, padx=10, pady=5)
        label_2.grid(row=2, column=0, padx=10, pady=5)
        entry_1.grid(row=1, column=1, padx=10, pady=5)
        entry_2.grid(row=2, column=1, padx=10, pady=5)
        
    elif option == "Grey Transmittance":
        label_1.config(text="Extinction Coefficient [m^2/mol]")
        label_2.config(text="Molar Density [mol/m^3]")
        label_3.config(text="Length [m]")
        description_var.set("Grey transmittance estimates the fraction of radiation transmitted through a medium based on its properties.")
        
        label_1.grid(row=1, column=0, padx=10, pady=5)
        label_2.grid(row=2, column=0, padx=10, pady=5)
        label_3.grid(row=3, column=0, padx=10, pady=5)
        entry_1.grid(row=1, column=1, padx=10, pady=5)
        entry_2.grid(row=2, column=1, padx=10, pady=5)
        entry_3.grid(row=3, column=1, padx=10, pady=5)
        
    elif option == "Radiant Heat Flux":
        label_1.config(text="Emissivity [-]")
        label_2.config(text="Temperature T [K]")
        label_3.config(text="Temperature T2 [K]")
        description_var.set("Radiant heat flux calculates the heat energy radiated between two surfaces based on their temperatures and emissivity.")
        
        label_1.grid(row=1, column=0, padx=10, pady=5)
        label_2.grid(row=2, column=0, padx=10, pady=5)
        label_3.grid(row=3, column=0, padx=10, pady=5)
        entry_1.grid(row=1, column=1, padx=10, pady=5)
        entry_2.grid(row=2, column=1, padx=10, pady=5)
        entry_3.grid(row=3, column=1, padx=10, pady=5)
        
    elif option == "Solar Spectrum":
        description_var.set("The solar spectrum provides key wavelength range and intensity details about the sun's emitted radiation. No user input required.")
        pass

app = tk.Tk()
app.title("Radiation Property Fetcher")

# Dropdown for selecting the computation type
selected_option = tk.StringVar()
option_menu = ttk.OptionMenu(app, selected_option, "Blackbody Spectral Radiance", "Blackbody Spectral Radiance", "Grey Transmittance", "Radiant Heat Flux", "Solar Spectrum", command=update_input_fields)
option_menu.grid(row=0, columnspan=2, pady=10)
# Dropdown addition for blackbody characteristics
option_menu['menu'].add_command(label="Blackbody Characteristics", command=tk._setit(selected_option, "Blackbody Characteristics", update_blackbody_input_fields))

# Label to display a description for the selected option
description_var = tk.StringVar()
description_label = ttk.Label(app, textvariable=description_var, font=("Arial", 9), anchor="w", justify=tk.LEFT, wraplength=400)
description_label.grid(row=6, columnspan=2, padx=10, pady=10, sticky='w')

# Input fields
label_1 = ttk.Label(app, text="Temperature [K]")
label_1.grid(row=1, column=0, padx=10, pady=5)
entry_1 = ttk.Entry(app)
entry_1.grid(row=1, column=1, padx=10, pady=5)

label_2 = ttk.Label(app, text="Wavelength [m]")
label_2.grid(row=2, column=0, padx=10, pady=5)
entry_2 = ttk.Entry(app)
entry_2.grid(row=2, column=1, padx=10, pady=5)

label_3 = ttk.Label(app, text="")
label_3.grid(row=3, column=0, padx=10, pady=5)
entry_3 = ttk.Entry(app)
entry_3.grid(row=3, column=1, padx=10, pady=5)

# Fetch button
ttk.Button(app, text="Fetch Properties", command=fetch_radiation_data).grid(row=4, columnspan=2, pady=10)

# Output label
result_var = tk.StringVar()
result_label = ttk.Label(app, textvariable=result_var, font=("Arial", 10), anchor="w", justify=tk.LEFT)
result_label.grid(row=5, columnspan=2, padx=10, pady=5)

app.mainloop()

```

# apps/Thermodynamics_Rayleigh_Bernard_Convection.py

```py
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 1.0, 1.0  # domain size
Nx, Ny = 128, 128  # grid size
dx, dy = Lx/Nx, Ly/Ny

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

X, Y = np.meshgrid(x, y)
# Temperature parameters
T_top = 0.0
T_bottom = 1.0
T = T_top + (T_bottom - T_top) * (Ly - Y) / Ly

# Visualize initial condition
plt.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Initial Temperature Field')
plt.show()
# Simulation parameters
dt = 0.001
Ra = 1e6  # Rayleigh number
Pr = 1.0  # Prandtl number
g = 9.81  # Gravity
alpha = 1e-4  # Thermal expansion coefficient
nu = 1.0/Pr  # Kinematic viscosity

# Abstracted simulation loop
for t in range(1000):
    # This is a gross oversimplification
    buoyancy_force = -g * alpha * (T - T_top)
    T += buoyancy_force * dt

    # Add some artificial diffusion to mimic the effects of viscosity
    T += nu * (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
               np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4*T) * dt
    if t % 100 == 0:
        plt.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Temperature Field at t={t*dt:.2f}')
        plt.pause(0.1)


```

# apps/Thermodynamics_Ternary_Nitrogen_Oxygen_Argon.py

```py
import tkinter as tk
from tkinter import ttk
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, FlashVL
from thermo.interaction_parameters import IPDB
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_phase_envelope(nitrogen_frac, oxygen_frac, argon_frac, plot_type, TP_value):
    constants, properties = ChemicalConstantsPackage.from_IDs(['oxygen', 'nitrogen', 'argon'])
    kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
    zs = [nitrogen_frac, oxygen_frac, argon_frac]

    fig = plt.Figure(figsize=(6, 4))

    if plot_type == "TP":
        flasher.plot_TP(zs, Tmin=40, Tmax=130, fig=fig)
    elif plot_type == "PT":
        flasher.plot_PT(zs, Pmin=1e4, Pmax=3.5e6, fig=fig)
    elif plot_type == "TernaryT":
        flasher.plot_ternary(T=TP_value, fig=fig)
    elif plot_type == "TernaryP":
        flasher.plot_ternary(P=TP_value, fig=fig)

    return fig

def on_plot():
    nitrogen_frac = float(nitrogen_entry.get())
    oxygen_frac = float(oxygen_entry.get())
    argon_frac = float(argon_entry.get())
    plot_type = plot_type_combobox.get()
    TP_value = float(TP_entry.get())

    fig = plot_phase_envelope(nitrogen_frac, oxygen_frac, argon_frac, plot_type, TP_value)

    for widget in frame_plot.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

app = tk.Tk()
app.title("Nitrogen, Oxygen, Argon Ternary Air System Phase Envelope")

# Input widgets
ttk.Label(app, text="Mole Fraction Nitrogen:").grid(row=0, column=0, padx=10, pady=5)
nitrogen_entry = ttk.Entry(app)
nitrogen_entry.grid(row=0, column=1, padx=10, pady=5)
nitrogen_entry.insert(0, '0.78')

ttk.Label(app, text="Mole Fraction Oxygen:").grid(row=1, column=0, padx=10, pady=5)
oxygen_entry = ttk.Entry(app)
oxygen_entry.grid(row=1, column=1, padx=10, pady=5)
oxygen_entry.insert(0, '0.21')

ttk.Label(app, text="Mole Fraction Argon:").grid(row=2, column=0, padx=10, pady=5)
argon_entry = ttk.Entry(app)
argon_entry.grid(row=2, column=1, padx=10, pady=5)
argon_entry.insert(0, '0.01')

ttk.Label(app, text="Plot Type:").grid(row=3, column=0, padx=10, pady=5)
plot_types = ["TP", "PT", "TernaryT", "TernaryP"]
plot_type_combobox = ttk.Combobox(app, values=plot_types)
plot_type_combobox.grid(row=3, column=1, padx=10, pady=5)
plot_type_combobox.set("TP")

ttk.Label(app, text="T (for Ternary) or P (in Pa):").grid(row=4, column=0, padx=10, pady=5)
TP_entry = ttk.Entry(app)
TP_entry.grid(row=4, column=1, padx=10, pady=5)
TP_entry.insert(0, '100000')

ttk.Button(app, text="Plot", command=on_plot).grid(row=5, columnspan=2, pady=10)

frame_plot = ttk.Frame(app)
frame_plot.grid(row=6, columnspan=2, padx=10, pady=5, sticky='nsew')
frame_plot.columnconfigure(0, weight=1)
frame_plot.rowconfigure(0, weight=1)

app.columnconfigure(0, weight=1)
app.columnconfigure(1, weight=1)
app.rowconfigure(6, weight=1)

app.mainloop()

```

# apps/Thermodynamics_Water_Compression_Sizing.py

```py
import tkinter as tk
from tkinter import ttk
from scipy.constants import bar, hour
import numpy as np
from thermo import FlashPureVLS, IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
from scipy.integrate import quad
from chemicals import property_molar_to_mass

# Define the calculation functions

def calculate_shaft_and_cooling(T1, P1, P2, mass_flow):
    T1 += 273.15
    T2 = T1

    liquid = IAPWS95Liquid(T=T1, P=P1, zs=[1])
    gas = IAPWS95Gas(T=T1, P=P1, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    mole_flow = property_molar_to_mass(mass_flow, MW=iapws_constants.MWs[0])

    entry = flasher.flash(T=T1, P=P1)
    leaving = flasher.flash(T=T2, P=P2)

    def to_int(P, flasher):
        state = flasher.flash(T=T1, P=P)
        return state.V()
    integral_result = quad(to_int, P1, P2, args=(flasher,))[0]
    shaft_duty = integral_result*mole_flow
    cooling_duty = shaft_duty - (leaving.H() - entry.H())*mole_flow
    return shaft_duty, cooling_duty

# Define GUI layout and behavior

def on_calculate():
    T1 = float(temperature_entry.get())
    P1 = float(initial_pressure_entry.get()) * bar
    P2 = float(final_pressure_entry.get()) * bar
    mass_flow = float(flow_rate_entry.get()) / hour

    shaft_duty, cooling_duty = calculate_shaft_and_cooling(T1, P1, P2, mass_flow)

    shaft_power_var.set(f"Shaft Power: {shaft_duty:.4f} W")
    cooling_duty_var.set(f"Cooling Duty: {cooling_duty:.4f} W")

# Create the main window
app = tk.Tk()
app.title("Reversible & Isothermal Compression of Liquid Water")

# Labels and entry fields
ttk.Label(app, text="Temperature (°C):").grid(column=0, row=0, sticky=tk.W)
temperature_entry = ttk.Entry(app)
temperature_entry.grid(column=1, row=0)

ttk.Label(app, text="Initial Pressure (bar):").grid(column=0, row=1, sticky=tk.W)
initial_pressure_entry = ttk.Entry(app)
initial_pressure_entry.grid(column=1, row=1)

ttk.Label(app, text="Final Pressure (bar):").grid(column=0, row=2, sticky=tk.W)
final_pressure_entry = ttk.Entry(app)
final_pressure_entry.grid(column=1, row=2)

ttk.Label(app, text="Flow Rate (kg/h):").grid(column=0, row=3, sticky=tk.W)
flow_rate_entry = ttk.Entry(app)
flow_rate_entry.grid(column=1, row=3)

# Calculate button
calculate_button = ttk.Button(app, text="Calculate", command=on_calculate)
calculate_button.grid(columnspan=2, row=4, pady=10)

# Result display labels
shaft_power_var = tk.StringVar()
cooling_duty_var = tk.StringVar()
ttk.Label(app, textvariable=shaft_power_var).grid(columnspan=2, row=5, pady=5)
ttk.Label(app, textvariable=cooling_duty_var).grid(columnspan=2, row=6, pady=5)

# Explanations
shaft_power_explanation = ("Shaft Power (or Shaft Duty) represents the mechanical power required by the "
                          "shaft of the compressor to compress the liquid. It relates to the work done by the "
                          "shaft to change the state of the fluid.")

cooling_duty_explanation = ("Cooling Duty describes the amount of heat that needs to be removed from a system "
                            "or component to maintain its temperature at the desired level. In this context, "
                            "it's the difference between the shaft power and the enthalpy change of the system "
                            "during compression. If this heat isn't removed, the temperature of the system would "
                            "increase, which isn't desired in an isothermal compression process.")

ttk.Label(app, text=shaft_power_explanation, wraplength=400).grid(columnspan=2, row=7, pady=5)
ttk.Label(app, text=cooling_duty_explanation, wraplength=400).grid(columnspan=2, row=8, pady=5)

app.mainloop()

```

# apps/Units_Density.py

```py
import tkinter as tk
from tkinter import ttk

class DensityConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Density Converter")
        self.geometry("600x400")

        # Units and their respective conversion values relative to 1 kg/m^3
        self.units = {
            "kg/m^3": 1,
            "g/cm^3": 1e-3,
            "g/ml": 1e-3,
            "lb/ft^3": 0.062427961,
            "lb/in^3": 0.0361273,
            "slugs/ft^3": 0.00194032,
            "tonnes/m^3": 1e-3
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    value_in_kg_per_m3 = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_kg_per_m3)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_kg_per_m3):
        """Update all entries based on the value in kg/m^3, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_kg_per_m3 * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = DensityConverter()
    app.mainloop()

```

# apps/Units_Energy.py

```py
import tkinter as tk
from tkinter import ttk

class EnergyConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Energy Converter")
        self.geometry("600x500")

        # Units and their respective conversion values relative to 1 Joule
        self.units = {
            "Joule (J)": 1,
            "Kilojoule (kJ)": 1e-3,
            "Megajoule (MJ)": 1e-6,
            "Gigajoule (GJ)": 1e-9,
            "Calorie (cal)": 0.239006,
            "Kilocalorie (kcal)": 0.000239006,
            "British Thermal Unit (BTU)": 0.000947817,
            "Therm (US)": 9.4804e-9,
            "Kilowatt-hour (kWh)": 2.7778e-7,
            "Electronvolt (eV)": 6.242e+18,
            "Erg": 1e7,
            "Foot-pound (ft·lbf)": 0.737562
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    value_in_j = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_j)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_j):
        """Update all entries based on the value in Joules, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_j * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = EnergyConverter()
    app.mainloop()

```

# apps/Units_Length_m_to_all.py

```py
import tkinter as tk
from tkinter import ttk

class LengthConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Length Converter")
        self.geometry("600x400")

        # Units and their respective conversion values relative to 1 meter
        self.units = {
            "meters": 1,
            "km": 0.001,
            "cm": 100,
            "mm": 1000,
            "inches": 39.3701,
            "feet": 3.28084,
            "miles": 0.000621371
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    value_in_m = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_m)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_m):
        """Update all entries based on the value in meters, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_m * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = LengthConverter()
    app.mainloop()

```

# apps/Units_Mass.py

```py
import tkinter as tk
from tkinter import ttk

class MassConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mass Converter")
        self.geometry("600x400")

        # Units and their respective conversion values relative to 1 kilogram
        self.units = {
            "kg": 1,
            "g": 1e3,
            "mg": 1e6,
            "lb": 2.20462,
            "oz": 35.274,
            "tonne": 1e-3,
            "stone": 0.157473,
            "grain": 15432.4,
            "slugs": 0.0685218
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    value_in_kg = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_kg)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_kg):
        """Update all entries based on the value in kilograms, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_kg * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = MassConverter()
    app.mainloop()

```

# apps/Units_Pressure_Pa_to_all.py

```py
import tkinter as tk
from tkinter import ttk

class PressureConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pressure Converter")
        self.geometry("600x400")

        # Units and their respective conversion values relative to 1 Pascal
        self.units = {
            "Pa": 1,
            "kPa": 1e-3,
            "MPa": 1e-6,
            "bar": 1e-5,
            "bar(g)": 1e-5,
            "bar(a)": 1.01325e-5,
            "torr": 7.50062,
            "atm": 9.86923e-6,
            "psi": 0.000145038,
            "m water": 0.101972,
            "ft water": 0.0298907,
            "in water": 0.0360912,
            "mmHg": 0.00750062
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    if unit == "bar(g)":
                        value_in_pa = (float(entry.get()) + 1.01325) / self.units[unit]
                    else:
                        value_in_pa = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_pa)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_pa):
        """Update all entries based on the value in Pascal, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                if unit == "bar(g)":
                    entry_value = (value_in_pa * self.units[unit]) - 1.01325
                else:
                    entry_value = value_in_pa * self.units[unit]

                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = PressureConverter()
    app.mainloop()

```

# apps/Units_TempC_to_all.py

```py
import tkinter as tk
from tkinter import ttk

class TemperatureConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Temperature Converter")
        self.geometry("600x250")

        # Units and their respective conversion functions relative to Celsius
        self.units = {
            "Celsius": lambda x: x,
            "Kelvin": lambda x: x + 273.15,
            "Fahrenheit": lambda x: (x * 9/5) + 32,
            "Rankine": lambda x: (x + 273.15) * 9/5
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    if unit == "Celsius":
                        celsius_value = float(entry.get())
                    elif unit == "Kelvin":
                        celsius_value = float(entry.get()) - 273.15
                    elif unit == "Fahrenheit":
                        celsius_value = (float(entry.get()) - 32) * 5/9
                    else:  # Rankine
                        celsius_value = (float(entry.get()) - 491.67) * 5/9

                    self.update_all_except(unit, celsius_value)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, celsius_value):
        """Update all entries based on the value in Celsius, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = self.units[unit](celsius_value)
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = TemperatureConverter()
    app.mainloop()

```

# apps/Units_Viscosity.py

```py
import tkinter as tk
from tkinter import ttk

class ViscosityConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Viscosity Converter")
        self.geometry("600x300")

        # Labels and Entries
        ttk.Label(self, text="Density:").grid(row=0, column=0, padx=10, pady=5)
        self.density_value = ttk.Entry(self)
        self.density_value.grid(row=0, column=1, padx=10, pady=5)
        self.density_units = ttk.Combobox(self, values=["kg/m³", "lb/ft³"], state="readonly")
        self.density_units.set("kg/m³")
        self.density_units.grid(row=0, column=2, padx=10, pady=5)
        self.density_value.bind('<Return>', self.on_entry)
        self.density_value.bind('<FocusOut>', self.on_entry)

        ttk.Label(self, text="Dynamic Viscosity:").grid(row=1, column=0, padx=10, pady=5)
        self.dynamic_value = ttk.Entry(self)
        self.dynamic_value.grid(row=1, column=1, padx=10, pady=5)
        self.dynamic_units = ttk.Combobox(self, values=["Pa·s", "P", "cP"], state="readonly")
        self.dynamic_units.set("Pa·s")
        self.dynamic_units.grid(row=1, column=2, padx=10, pady=5)
        self.dynamic_value.bind('<Return>', self.on_entry)
        self.dynamic_value.bind('<FocusOut>', self.on_entry)

        ttk.Label(self, text="Kinematic Viscosity:").grid(row=2, column=0, padx=10, pady=5)
        self.kinematic_value = ttk.Entry(self)
        self.kinematic_value.grid(row=2, column=1, padx=10, pady=5)
        self.kinematic_units = ttk.Combobox(self, values=["m²/s", "St", "cSt"], state="readonly")
        self.kinematic_units.set("m²/s")
        self.kinematic_units.grid(row=2, column=2, padx=10, pady=5)
        self.kinematic_value.bind('<Return>', self.on_entry)
        self.kinematic_value.bind('<FocusOut>', self.on_entry)

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        try:
            if event.widget == self.density_value or event.widget == self.dynamic_value:
                self.update_kinematic()

            elif event.widget == self.kinematic_value:
                self.update_dynamic()

        except ValueError:
            pass

    def update_kinematic(self):
        density = float(self.density_value.get()) * self.convert_density(self.density_units.get(), "kg/m³")
        dynamic_viscosity = float(self.dynamic_value.get()) * self.convert_dynamic_viscosity(self.dynamic_units.get(), "Pa·s")
        
        kinematic_viscosity = dynamic_viscosity / density

        self.kinematic_value.delete(0, tk.END)
        self.kinematic_value.insert(0, kinematic_viscosity / self.convert_kinematic_viscosity("m²/s", self.kinematic_units.get()))

    def update_dynamic(self):
        density = float(self.density_value.get()) * self.convert_density(self.density_units.get(), "kg/m³")
        kinematic_viscosity = float(self.kinematic_value.get()) * self.convert_kinematic_viscosity(self.kinematic_units.get(), "m²/s")
        
        dynamic_viscosity = kinematic_viscosity * density

        self.dynamic_value.delete(0, tk.END)
        self.dynamic_value.insert(0, dynamic_viscosity / self.convert_dynamic_viscosity("Pa·s", self.dynamic_units.get()))

    def convert_density(self, from_unit, to_unit):
        conversions = {
            "kg/m³": 1,
            "lb/ft³": 16.0185  # kg/m³ to lb/ft³ conversion factor
        }
        return conversions[to_unit] / conversions[from_unit]

    def convert_dynamic_viscosity(self, from_unit, to_unit):
        conversions = {
            "Pa·s": 1,
            "P": 0.1,
            "cP": 0.001
        }
        return conversions[to_unit] / conversions[from_unit]

    def convert_kinematic_viscosity(self, from_unit, to_unit):
        conversions = {
            "m²/s": 1,
            "St": 0.0001,
            "cSt": 0.000001
        }
        return conversions[to_unit] / conversions[from_unit]

if __name__ == "__main__":
    app = ViscosityConverter()
    app.mainloop()

```

# apps/Units_Volume.py

```py
import tkinter as tk
from tkinter import ttk

class VolumeConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Volume Converter")
        self.geometry("900x700")

        # Units and their respective conversion values relative to 1 cubic meter
        self.units = {
            "cubic meter (m^3)": 1,
            "litre": 1000,
            "millilitre (ml)": 1e6,
            "gallon (US)": 264.172,
            "gallon (UK)": 219.969,
            "quart (US)": 1056.69,
            "quart (UK)": 879.877,
            "20-foot container": 33.2,
            "40-foot container": 67.7,
            "53-foot truck trailer": 130,  # Approximate volume
            "Fuel Truck (small, 20,000l)": 20,
            "Fuel Truck (large, 36,000l)": 36,
            "Tanker Ship (small)": 1e3,   # Placeholder value
            "Tanker Ship (VLCC)": 3e5,    # Placeholder value
            "Air Freight ULD LD3": 4.5,   # Typical air cargo container
            "Rail Car (average)": 120,    # Typical volume, can vary widely
            "Standard Pallet": 1.5,       # Approximation, varies based on height/stack
            "Average Car": 3,             # Rough average, can vary widely
            "Semi-trailer": 85            # Approximation
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    value_in_m3 = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_m3)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_m3):
        """Update all entries based on the value in cubic meters, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_m3 * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = VolumeConverter()
    app.mainloop()

```

# chemical_databases/AFS2018merged_data_modified_english.csv

```csv
ChemicalName,CAS­nr,Year,TWA(ppm),TWA(mg/m^3),STEL(ppm),STEL(mg/m3)
Acetaldehyde,75­0-7­-0,1990,25,45,50,90
Acetamide,60­3-5­-5,1993,10,25,25,60
Acetone,67­6-4­-1,1993,250,600,500,1200
Acetonitrile,75­0-5­-8,1993,30,50,60,100
Acrolein,107­0-2­-8,2018,"0,02","0,05","0,05","0,12"
Acrylamide,79­0-6­-1,2018,−,"0,03",−,"0,1"
Acrylonitrile,107­1-3­-1,1993,2,"4,5",6,13
Acrylic acid,79­1-0­-7,2018,10,29,20,59
Allyl alcohol,107­1-8­-6,2015,2,5,5,12
Allylamine,107­1-1­-9,1984,2,5,6,14
Allyl chloride,107­0-5­-1,1993,1,3,3,9
"Aluminium*,  metall  och  oxid  (som  Al)",7429-90-5,1996,,,,
– totaldamm,7429-90-5,,−,5,−,−
– respirabel  fraktion,7429-90-5,,−,2,−,−
"Aluminium, lösliga föreningar (som Al)",7429-90-5,1996,,,,
Aluminum Dust,7429-90-5,,−,1,−,−
pAminoazobenzene,60­0-9­-3,,,,,
4 Aminodiphenyl,92­6-7­-1,,,,,
4-Aminotoluene,106­4-9­-0,2020,1,"4,46",2,"8,92"
Amitrol,61­8-2­-5,2018,−,"0,2",−,−
Ammonia,7664­-41­-7,2011,20,14,50,36
Amyl acetate,-na-n,,,,,
Aniline,62­5-3­-3,2020,1,4,2,8
"Antimony*, and compounds (such as Sb),",7440­-36-­0,2011,,,,
Antimony trihydride,7440­-36-­0,,,,,
– inhalable fraction,7440­-36-­0,,−,"0,25",−,−
Antimony trioxide,1309­6-4­-4,,,,,
Antimony trihydride,7803­5-2­-3,1974,"0,05","0,3",−,−
Arsenic,7440-38-2,2021,,"0,01",,
"Gasoline, industrial, octane type",-na-n,1989,200,900,300,1400
pBenzoquinone,106­5-1­-4,1978,"0,1","0,4","0,3","1,3"
Benzo(a)pyrene,50­3-2­-8,1993,−,2,−,"0,02"
Benzotrichloride,98­0-7­-7,,,,,
Benzyl butyl phthalate,85­6-8­-7,1987,−,3,−,5
Benzyl chloride,100­4-4­-7,1978,1,5,2,11
Benzidine,92­8-7­-5,,,,,
"Beryllium*, and compounds (such as Be)",7440-­41-­7,2020,−,"0,0006",−,−
Beryllium,7440-­41-­7,,,"0,00021",,
Biphenyl,7440-­41-­7,,,,,
Bisphenol A,80­0-5­-7,2018,,,,
Bisphenol A,80­0-5­-7,,−,2,−,−
Lead,7439­-92-­1,2011,,,,
Lead inhalable fraction,7439­-92-­1,,−,"0,1",−,−
Lead  respirable  fraction,7439­-92-­1,,−,"0,05",−,−
Lead monoxide,1317­-36­-8,,,,,
Lead monoxide Cotton dust,1317­-36­-8,,,,,
Lead monoxide Dust,1317­-36­-8,,,,,
Borax,1303-­96­-4,1978,,,,
Borax Dust,1303­-96­-4,,−,2,−,5
Bromine,7726­-95­-6,1974,"0,1","0,7","0,3",2
"1,3 Butadiene",106­9-9­-0,2018,"0,5",1,5,10
nButanol,71­3-6­-3,1989,15,45,30,90
isoButanol,78­8-3­-1,1987,50,150,75,250
sec Butanol,78­9-2­-2,1987,50,150,75,250
tertButanol,75­6-5­-0,1987,50,150,75,250
2Butoxyethanol,-na-n,,,,,
See: Ethylene glycol monobutyl ether,-na-n,,,,,
2Butoxyethyl acetate,-na-n,,,,,
See: Ethylene glycol monobutyl ether acetate,-na-n,,,,,
Butyl acetate,-na-n,2020,50,241,150,723
Butyl acetate,-na-n,2020,50,241,150,723
nButyl acrylate,141­3-2­-2,2015,2,11,10,53
Butylamine,-na-n,1984,−,−,5,15
nButylamine,109­7-3­-9,,,,,
isoButylamine,78­8-1­-9,,,,,
secButylamine,13952­8-4­-6,,,,,
tertButylamine,75­6-4­-9,,,,,
Butyl glycidyl ether,2426­0-8­-6,1981,10,50,15,80
Butyl methacrylate,97­8-8­-1,1987,50,300,75,450
"But-2-yne-1,4 diol",110­6-5­-6,2018,−,"0,5",−,−
Beta-Butyrolactone,3068­8-8­-0,,,,,
CFC 11,75­6-9­-4,1984,500,3000,750,4500
CFC 12,75­7-1­-8,1984,500,2500,750,4000
CFC 113,76­1-3­-1,1981,500,4000,750,6000
Cyanamid,420­0-4­-2,2015,"0,58",1,−,−
Cyanides (as CN),-na-n,2018,,,,
inhalable fraction,-na-n,,−,1,−,4
Potassium cyanide,151­5-0­-8,,,,,
Sodium cyanide,143­3-3­-9,,,,,
See also Hydrogen Cyanide,-na-n,,,,,
Cyan chloride,506­7-7­-4,1978,"0,1","0,3","0,3","0,8"
Cyclohexane,110­8-2­-7,2015,200,700,−,−
Cyclohexanol,108­9-3­-0,1978,50,200,75,300
Cyclohexanone,108­9-4­-1,2015,10,41,20,81
Cyclohexylamine,108­9-1­-8,1993,5,20,10,40
"Dust,  inorganic",-na-n,,,5,,
Dust inhalable  fraction,-na-n,2018,−,5,−,−
Dust – respirable  fraction,-na-n,2018,−,"2,5",−,−
"Dust  and  mist,  organic",-na-n,2018,,5,,
Dust inhalable  fraction,-na-n,,−,5,−,−
"Dust, cotton (raw cotton)",-na-n,2018,,"0,5",,
–inhalable fraction,-na-n,,−,"0,5",−,−
"Dust, thermoplastics",-na-n,2018,,3,,
–inhalable fraction,-na-n,,−,3,−,−
"Dust, coal incl. carbon black",-na-n,2018,,3,,
–inhalable fraction,-na-n,,−,3,−,−
"Dust, flour,",-na-n,2000,,,,
–inhalable fraction,-na-n,,−,3,−,−
"Dust, paper",-na-n,2018,,,,
–inhalable fraction,-na-n,,−,2,−,−
"Dust,  PVC",9002­8-6­-2,2018,,,,
– inhalable  fraction,-na-n,,−,1,−,−
– respirable  fraction,-na-n,,−,"0,5",−,−
"Dust, textiles",-na-n,2018,,,,
–inhalable fraction,-na-n,,−,1,−,−
"Dust, wood",-na-n,2018,,,,
–inhalable fraction,-na-n,,−,2,−,−
Dust from hardwoods (hardwood),-na-n,,,,,
Dust from softwoods (softwoods),-na-n,,,,,
Decanes and other higher aliphatic hydrocarbons,-na-n,1989,−,350,−,500
Desflurane,57041­6-7­-5,2000,10,70,20,140
Diacetone alcohol,-na-n,,,,,
See: 4Hydroxy4methyl2pentanone,-na-n,,,,,
Diacetyl,431­0-3­-8,2018,"0,02","0,07","0,1","0,36"
"4,4´Diamino3,3´ ",101­1-4­-4,2020,−,"0,01",−,−
Dibenzyl phthalate,523­3-1­-9,1987,−,3,−,5
"1,2Dibromo3chloropropane",96­1-2­-8,,,,,
"1,2Dibromoethane (ethyl dibromide)",106­9-3­-4,2018,"0,1","0,8",−,−
Dibutyl phthalate,84­7-4­-2,1987,−,3,−,5
"1,2:3,4Diepoxybutane",1464­5-3­-5,,,,,
Diesel exhaust,-na-n,2020,−,"0,05",−,−
Diethanolamine,111­-42­-2,1993,3,15,6,30
Diethylamine,109­-89­-7,2015,5,15,10,30
2Diethylaminoethanol,100­-37­-8,1996,2,10,10,50
Diethylene glycol,111­-46­-6,1993,10,45,20,90
Diethylene glycol monobutyl ether,112­-3-4­5,2015,10,68,15,101
Diethylene glycol monobutyl ether acetate,124­1-7­-4,1996,15,130,30,250
Diethylene glycol monoethyl ether,111­9-0­-0,2000,15,80,30,170
Diethylene glycol monoethyl ether acetate,112­1-5­-2,2000,15,110,30,220
Diethylene glycol monomethyl ether,111­7-7­-3,2015,10,50,−,−
Diethylenetriamine,111­4-0­-0,1996,1,"4,5",2,10
Diethyl ether,60­2-9­-7,2015,100,308,200,616
Diethyl phthalate,84­6-6­-2,1987,−,3,−,5
Diethyl sulfate,64­6-7­-5,,,,,
Diphenyl,92­5-2­-4,1974,"0,2","1,3","0,4","2,5"
Diphenylamine,122­3-9­-4,1996,−,4,−,12
Diphenyl ether,101­8-4­-8,2018,1,7,2,14
Diphosphorus pentasulfide,1314­8-0­-3,2015,−,1,−,−
Diglycidyl ethers,2238­0-7­-5,1981,−,−,"0,2","1,1"
Diisocyanates,-na-n,2005,2,−,5,−
Hexamethylene diisocyanate,822­0-6­-0,,2,"0,02",5,"0,03"
Isophorone diisocyanate,4098­7-1­-9,,2,18,5,46
"4,4'Methylene diphenyl diisocyanate",101­6-8­-8,,2,"0,03",5,"0,05"
"1,5Naphthalene diisocyanate",3173­7-2­-6,,2,17,5,44
Toluene diisocyanate,26471­6-2­-5,,2,14,5,"0,04"
"2,4Toluene diisocyanate",584­8-4­-9,,,,,
"2,6Toluene diisocyanate",91­0-8­-7,,,,,
Trimethylhexamethylene diisocyanate,28679­1-6­-5,,2,17,5,44
"2,2,4Trimethylhexamethylene",16938­2-2­-0,,,,,
diisocyanate,-na-n,,,,,
"2,4,4Trimethylhexamethylene",15646­9-6­-5,,,,,
diisocyanate,-na-n,,,,,
Diisodecyl phthalate,26761­4-0­-0,1987,−,3,−,5
Diisopropylamine,108­1-8­-9,1993,5,20,10,40
"2,6Diisopropylphenyl isocyanate",28178­4-2­-9,2005,5,"0,04","0,01","0,08"
oDichlorobenzene,95­5-0­-1,2018,20,122,50,306
pDichlorobenzene,106­4-6­-7,2018,2,12,10,60
"3,3' Dichlorobenzidine",91­9-4­-1,,,,,
"2,2Dichlorodiethyl ether",111­4-4­-4,,,,,
"2,2'Dichlorodiethyl sulfide (mustard gas)",505­6-0­-2,,,,,
Dichlorodifluoromethane,-na-n,,,,,
See: CFC 12,-na-n,,,,,
"1,1'Dichlorodimethyl ether",542­8-8­-1,,,,,
"1,1Dichloroethane",75­3-4­-3,2015,100,412,−,−
"1,2Dichloroethane",107­0-6­-2,2018,1,4,5,20
"1,1Dichloroethylene",75­3-5­-4,2018,2,8,5,20
Dichloromethane,-na-n,,,,,
See: Methylene chloride,-na-n,,,,,
Nitrous oxide,-na-n,,,,,
See:  Nitrous oxide,-na-n,,,,,
"N,N-methylacetamide",127­1-9­-5,2015,10,35,20,70
Dimethyl adipate,627­9-3­-0,2005,5,36,−,−
Dimethylamine,124­4-0­-3,2015,2,"3,5",5,9
"N,NDimethy4aminoazobenzene",60­1-1­-7,,,,,
S(2(Dimethylamino)ethyl)pseudothiourea dihy,16111­2-7­-6,,,,,
hydrochloride (PBA 1),-na-n,,,,,
"N,NDmethylaniline",121­6-9­-7,1993,1,5,2,10
"3,3'Dimethylbenzidine",-na-n,,,,,
See: oTolidine,-na-n,,,,,
Dimethyldisulfide,624­9-2­-0,1993,1,−,−,−
Dimethyl ether,115­1-0­-6,1996,500,950,800,1500
Dimethylethylamine,598­5-6­-1,1993,2,6,5,15
"N,NDimethyformamide",68­1-2­-2,2015,5,15,10,30
Dimethyl phthalate,131­1-1­-3,1987,−,3,−,5
Dimethyl glutarate,1119­4-0­-0,2005,5,33,−,−
"1,1Dimethylhydrazine",57­1-4­-7,,,,,
"1,2Dimethylhydrazine",540­7-3­-8,,,,,
Dimethyl succinate,106­6-5­-0,2005,5,30,−,−
Dimethyl sulfate,77­7-8­-1,,,,,
Dimethyl sulfide,75­1-8­-3,1993,1,−,−,−
Dimethyl sulfoxide,67­6-8­-5,1993,50,150,150,500
Dinitrobenzene,25154­5-4­-5,1978,"0,15",1,"0,3",2
"1,2­Dinitrobensen",528­2-9­-0,,,,,
Dimetylsulfat,77­7-8­-1,,,,,
Dimetylsulfid,75­1-8­-3,1993,1,−,−,−
Dimetylsulfoxid,67­6-8­-5,1993,50,150,150,500
Dinitrobensen,25154­5-4­-5,1978,"0,15",1,"0,3",2
"1,2­Dinitrobensen",528­2-9­-0,,,,,
"1,3­Dinitrobensen",99­6-5­-0,,,,,
"1,4­Dinitrobensen",100­2-5­-4,,,,,
Dinitrotoluen,25321­1-4­-6,1993,−,"0,15",−,"0,3"
"2,4­Dinitrotoluen",121­1-4­-2,,,,,
"2,6­Dinitrotoluen",606­2-0­-2,,,,,
Dioktylftalat,-na-n,1987,−,3,−,5
Di­n­oktylftalat,117­8-4­-0,,,,,
Di(2­etylhexyl)ftalat,117­8-1­-7,,,,,
Dioxan,123­9-1­-1,1996,10,35,25,90
Dipropylenglykolmonometyleter,34590­9-4­-8,1993,50,300,75,450
Disulfiram,97­7-7­-8,1993,−,1,−,2
Enfluran,13838­1-6­-9,1981,10,80,20,150
"Enzymer, subtilisiner",1395­2-1­-7,1996,,1 glycinenhet/m3,,3 glycinenheter/m3
"Enzymer, subtilisiner",9014­0-1­-1,,,,,
 Epiklorhydrin,106­8-9­-8,2018,"0,5","1,9",1,4
Erionit,12510­4-2­-8,,,,,
Etanol,64­1-7­-5,1993,500,1000,1000,1900
Etanolamin,141­4-3­-5,2015,1,"2,5",3,"7,5"
Eten,74­8-5­-1,2000,250,330,1000,1200
 Etylacetat,141­7-8­-6,2018,150,550,300,1100
Etylakrylat,140­8-8­-5,2015,5,20,10,40
Etylamin,75­0-4­-7,2015,5,"9,4",10,"18,8"
Etylbensen,100­4-1­-4,2015,50,220,200,884
Etyl­2­cyanoakrylat,7085­8-5­-0,2000,2,10,4,20
Etylendiamin,107­1-5­-3,1978,10,25,15,35
Etylenglykol,107­2-1­-1,2015,10,25,40,104
 Etylenglykoldinitrat,628­9-6­-6,2018,16,"0,1","0,03","0,2"
Etylenglykolmonobutyleter,111­7-6­-2,2015,10,50,50,246
Etylenglykolmonobutyleteracetat,112­0-7­-2,2015,10,70,50,333
Etylenglykolmonoetyleter,110­8-0­-5,2015,2,8,−,−
Etylenglykolmonoetyleteracetat,111­1-5­-9,2015,2,11,−,−
Etylenglykolmonoisopropyleter,109­5-9­-1,1996,10,45,20,90
Etylenglykolmonoisopropyleteracetat,19234­2-0­-9,1996,10,60,20,120
 Etylenglykolmonometyleter,109­8-6­-4,2018,1,−,−,−
 Etylenglykolmonometyleteracetat,110­4-9­-6,2018,1,−,−,−
Etylenglykolmonopropyleter,2807­3-0­-9,1996,10,45,20,90
Etylenimin (Aziridin),151­5-6­-4,,,,,
 Etylenoxid,75­2-1­-8,2018,1,"1,8",5,9
Etylentiourea,96­4-5­-7,,,,,
 2­Etylhexanol,104­7-6­-7,2018,1,"5,4",−,−
2­Etylhexyllaktat,6283­8-6­-9,1996,5,40,10,80
2­Etylkarbamat,-na-n,,,,,
Se: Uretan,-na-n,,,,,
Etyllaktat,97­6-4­-3,1996,5,25,10,50
Etylmetakrylat,97­6-3­-2,1987,50,250,75,350
Etylmetansulfonat,62­5-0­-0,,,,,
N­Etylmorfolin,100­7-4­-3,1984,5,25,10,50
Fenol,108­9-5­-2,2015,1,4,4,16
Fenylglycidyleter,122­6-0­-1,1981,10,60,15,90
Fenylisocyanat,103­7-1­-9,1984,5,"0,02","0,01","0,05"
Asbest,-na-n,2005,"0,1 fibrer/cm³",,−,−
Asbest,-na-n,2005,"0,1 fibrer/cm³",−,−,"C,M"
,-na-n,,"0,5 fibrer/cm³",−,−,
Övriga fibrer,-na-n,1990,,,,
 Eldfasta keramiska fibrer,-na-n,2018,"0,2 fibrer/cm³","−
−","C,M","28,30"
Specialfibrer,-na-n,2005,"0,2 fibrer/cm³","−
−","C,M",28
Övriga fibrer,-na-n,2005,1 fibrer/cm³,"−
−",,28
"Fibrer, syntetiska oorg.",-na-n,2005,"0,2 fibrer/cm³","−
−",M,28
Fosgen,75­4-4­-5,2015,"0,02","0,08","0,05","0,2"
Ftalater,-na-n,1987,−,3,−,5
Ftalsyraanhydrid,85­4-4­-9,2011,"0,03","0,2","0,06","0,4"
Furfural,98­0-1­-1,1990,2,8,5,20
Furfurylalkohol,98­0-0­-0,1990,5,20,10,40
Glutaraldehyd,111­3-0­-8,2011,−,−,"0,1","0,4"
Halotan,151­6-7­-7,1990,5,40,10,80
HCFC 22,75­4-5­-6,1984,500,1800,750,2500
n­Heptan* och andra heptaner,142­82-­5-*,1989,200,800,300,1200
2­Heptanon,110­4-3­-0,2015,25,120,100,475
 3­Heptanon,106­3-5­-4,2018,20,95,50,250
Hexahydroftalsyraanhydrid,85­4-2­-7,2011,,,,
Hexametylentetramin,100­9-7­-0,1984,−,3,−,5
 n­Hexan,110­5-4­-3,2018,20,72,50,180
"Hexaner, utom n­hexan",-na-n,1989,200,700,300,1100
2­Hexanon,591­7-8­-6,1993,1,4,2,8
HFC 134 a,811­9-7­-2,1996,500,2000,750,3000
 Hydrazin,302­0-1­-2,2018,"0,01",13,−,−
Hydrokinon,123­3-1­-9,1993,−,"0,5",−,"1,5"
2­Hydroxietylakrylat,818­6-1­-1,1981,1,5,2,10
4­Hydroxi­4­metyl­2­pentanon,123­4-2­-2,1993,25,120,50,240
Indium* och oorg föreningar (som In),7440­74-­6-*,1996,,,,
–totaldamm,-na-n,,−,"0,1",−,−
 Isoamylalkohol,123­5-1­-3,2020,5,18,10,37
Isobutylmetakrylat,97­8-6­-9,1987,50,300,75,450
Isocyanater,-na-n,,,,,
Se respektive isocyanat,-na-n,,,,,
Isocyansyra,75­1-3­-8,2004,"0,01",18,,"0,02"
Isofluran,26675­4-6­-7,1990,10,75,,20
Isoforon,78­5-9­-1,1993,−,−,,5
Isopropanol,67­6-3­-0,1989,150,350,,250
2­Isopropoxietanol,-na-n,,,,,
Se: Etylenglykolmonoisoproyleter,-na-n,,,,,
2­Isopropoxietylacetat,-na-n,,,,,
Se: Etylenglykolmonoisopropyleteracetat,-na-n,,,,,
Isopropylamin,75­3-1­-0,1993,5,12,,10
 Isopropylbensen,98­8-2­-8,2020,10,50,,50
Isopropylnitrat,1712­6-4­-7,1978,10,45,,15
Jod,7553­5-6­-2,1974,−,−,,"0,1"
Järnoxid (som Fe),1309­3-7­-1,1978,,,,
–respirabel fraktion,-na-n,,−,"3,5",,−
,-na-n,,−,4,−,
" Kadmium*, och oorg föreningar",7440­43-­9-*,2020,,,,
(som Cd),-na-n,,,"0,0012",,
–inhalerbar fraktion,-na-n,,,,,
Kadmiumdifluorid,7790­7-9­-6,,,,,
Kadmiumdiklorid,10108­6-4­-2,,,,,
 Kalciumhydroxid,1305­6-2­-0,2018,,1,,
–respirabel fraktion,-na-n,,−,1,,−
 Kalciumoxid,1305­7-8­-8,2018,,,,3
–respirabel fraktion,-na-n,,"−
1","−
4",,
Kaliumaluminiumtetrafluorid,60304­3-6­-1,2005,,,,3
–inhalerbar fraktion,-na-n,,"−

0,4","−
−",,
Kaliumhydroxid,1310­5-8­-3,2005,1,2,,3
–inhalerbar fraktion,-na-n,,"−

1","−

2",,
Kaprolaktam (damm + ånga),105­6-0­-2,2015,"−
5","−
40",,
Karbonyldiklorid,-na-n,,,,,
Se: Fosgen,-na-n,,,,,
3­Karen (jfr Terpener),13466­7-8­-9,1990,"25 
150","50 
300","S,V",
Katekol,120­8-0­-9,1993,"5 
20","10 
40","H,V",
Klor,7782­5-0­-5,2015,"−
−","0,5 
1,5",,
Klorbensen,108­9-0­-7,2015,"5 
23","15 
70",,
"2­Klor­1,3­butadien",126­9-9­-8,1990,"1 
3,5","5 
18","H,V",
Klordifluormetan,-na-n,,,,,
Se: HCFC22,-na-n,,,,,
Klordioxid,10049­0-4­-4,1996,"0,1 
0,3","0,3 
0,8",,
Kloretan,75­0-0­-3,2015,"100 
268","−
−",,
2­Kloretanol,107­0-7­-3,1981,"−
−","1 
3,5",H,24
Klorfenoler och salter(som klorfenol),-na-n,1990,−,"0,5",−,"1,5"
Klorkresol,59­5-0­-7,1993,−,3,−,6
Kloroform,67­6-6­-3,1978,2,10,5,25
"Kobolt*, och oorg. föreningar (som Co)",7440­48-­4-*,2011,,,,
–inhalerbar fraktion,-na-n,,−,"0,02",−,−
Koldioxid,124­3-8­-9,1974,5000,9000,10000,18000
Koldisulfid,75­1-5­-0,1978,5,16,8,25
 Kolmonoxid,630­0-8­-0,2018,20,23,100,117
 Koltetraklorid,56­2-3­-5,2018,1,"6,4",3,19
" Koppar*, och oorg. föreningar",7440­50-­8-*,2018,,,,
–respirabel fraktion,-na-n,,−,"0,01",−,−
Kresol,1319­7-7­-3,2000,1,"4,5",2,9
"Kristobalit,",14464­4-6­-1,1996,,,,
–respirabel fraktion,-na-n,,−,"0,05",−,−
"Krom*, och oorg. (II, III)­föreningar (som Cr)",7440­47-­3-*,2005,,,,
–totaldamm,-na-n,,−,"0,5",−,−
–inhalerbar fraktion,-na-n,,−,5,−,15
–respirabel fraktion,-na-n,−,"0,05",−,−,
"Krom*, och oorg. (II, III)­föreningar (som Cr)",7440­47-­3-*,2005,,,,3
–totaldamm,-na-n,−,"0,5",−,−,
–inhalerbar fraktion,-na-n,−,5,−,15,
 Litium* och föreningar (som Li),7439­93-­2-*,2018,,,,
–inhalerbar fraktion,-na-n,,−,−,−,"0,02"
Litiumhydrid,7580­6-7­-8,,,,,
Lustgas,10024­9-7­-2,1981,100,180,500,900
Maleinsyraanhydrid,108­3-1­-6,2011,"0,05","0,2","0,1","0,4"
" Mangan*, och oorg. föreningar (som Mn),",7439­96-­5-*,2018,,,,
–inhalerbar fraktion,-na-n,,−,"0,2",−,−
–respirabel fraktion,-na-n,,−,"0,05",−,−
Metakrylsyra,79­4-1­-4,1987,20,70,30,100
Metanol,67­5-6­-1,1990,200,250,250,350
Metantiol,74­9-3­-1,1993,1,−,−,−
1­Metoxi­2­propanol,107­9-8­-2,2015,50,190,150,568
1­Metoxi­2­propylacetat,108­6-5­-6,2015,50,275,100,550
Metylacetat,79­2-0­-9,1993,150,450,300,900
Metylakrylat,96­3-3­-3,2015,5,18,10,36
Metylamin,74­8-9­-5,1984,10,13,20,25
Metylbromid,74­8-3­-9,1990,"5 
19","10 
40","H,V",
Metyl­n­butylketon,-na-n,,,,,
Se: 2­Hexanon,-na-n,,,,,
Metyl­2­cyanoakrylat,137­0-5­-3,2000,"2 
9","4 
18","M,S,V",
" 4,4´­Metylendianilin",101­7-7­-9,2018,"0,01 
0,08","−
−","C,H,S",13
 Metylenklorid,75­0-9­-2,2018,"35 
120","70 
250","C,H",37
Metyletylketon,78­9-3­-3,2015,"50 
150","300 
900",,
Metyletylketonperoxid,1338­2-3­-4,1981,"−
−","0,2 
1,5",,
 Metylformiat,107­3-1­-3,2018,"50 
125","100 
250",H,
5­Metyl­3­heptanon,541­8-5­-5,2015,"10 
53","20 
107",,
Metylhexahydroftalsyraanhydrid,25550­5-1­-0,,,,"M,S","4,32"
 5­Metyl­2­hexanon,110­1-2­-3,2018,"20 
95","50 
250",,
 Metylisobutylketon,108­1-0­-1,2018,"20 
83","50 
200",,
Metylisocyanat,624­8-3­-9,2005,"0,01 
0,024","0,02 
0,047",M,"2,21"
4­Metylisopropylbensen,99­8-7­-6,1984,"25 
140","35 
190",V,
Metyljodid,74­8-8­-4,1981,"1 
6","5 
30","C,H,V",
 Metylklorid,74­8-7­-3,2020,"10 
20","20 
40","C,H,V",
Metylklormetyleter,107­3-0­-2,,,,C,1
Metylmetakrylat,80­6-2­-6,2015,50,200,100,400
Metylmetansulfonat,66­2-7­-3,,,,,
N­Metylmorfolin,109­0-2­-4,1984,5,20,10,40
N­Metyl­N­nitrosourea,684­9-3­-5,,,,,
"2­Metyl­2,4­pentandiol",107­4-1­-5,1996,−,−,25,120
4­Metyl­2­pentanol,108­1-1­-2,1996,25,110,40,170
 N­Metyl­2­pyrrolidon (NMP),872­5-0­-4,2020,"3,6","14,4",20,80
α­Metylstyren,98­8-3­-9,2015,20,98,100,492
Metyltertiärbutyleter,1634­0-4­-4,2015,30,110,100,367
Metyltetrahydroftalsyraanhydrid,26590­2-0­-5,,,,,
–totaldamm,-na-n,,−,10,−,−
–respirabel fraktion,-na-n,,−,5,−,−
"Molybden, lättlösliga föreningar",-na-n,1984,,,,
(som Mo),-na-n,,,,,
–totaldamm,-na-n,,−,5,−,−
Monoklorättikssyra,79­1-1­-8,1993,1,4,2,8
Monometylhydrazin,60­3-4­-4,,,,,
Morfolin,110­9-1­-8,2015,10,35,20,72
Myrsyra,64­1-8­-6,1990,3,5,5,9
Naftalen,91­2-0­-3,2000,10,50,15,80
"Naftalener, klorerade",1321­6-5­-9,1978,−,"0,2",−,"0,6"
α­Naftylamin,134­3-2­-7,,,,,
β­Naftylamin,91­5-9­-8,,,,,
Natriumazid,26628­2-2­-8,2015,−,"0,1",−,"0,3"
Natriumhydroxid,1310­7-3­-2,2005,,,,
–inhalerbar fraktion,-na-n,,−,1,−,2
"Nickel, metall",7440­0-2­-0,1978,,,,
–totaldamm,-na-n,,−,"0,5",−,−
"Nickelföreningar (som Ni), utom",-na-n,1978,,,,
Nickelkarbonyl och Trinickeldisulfid,-na-n,,,,,
–totaldamm,-na-n,,−,"0,1",−,−
Nickelkarbonyl,13463­3-9­-3,1974,1,7,−,−
Nikotin,54­1-1­-5,2011,−,"0,1",−,−
 Nitrobensen,98­9-5­-3,2018,"0,2",1,−,−
 Nitroetan,79­2-4­-3,2018,20,62,50,150
 Nitroglycerin,55­6-3­-0,2018,"0,01",95,"0,02","0,19"
Nitrometan,75­5-2­-5,1993,20,50,50,130
1­Nitropropan,108­0-3­-2,1990,5,18,10,35
 2­Nitropropan,79­4-6­-9,2018,2,7,6,20
N­Nitrosodimetylamin,62­7-5­-9,,,,,
Nitrotoluen,1321­1-2­-6,1993,1,6,2,11
Penicillin,-na-n,2011,,,,
–inhalerbar fraktion,-na-n,,−,"0,1",−,−
Pentaerytritol,115­7-7­-5,1996,,,,
–totaldamm,-na-n,,−,5,−,−
"1,1,1,2,2­Pentafluoretan",354­3-3­-6,2005,500,2500,750,3750
Pentaklorfenol* och salter,87­86-­5-*,1974,−,"0,5",−,"1,5"
(som pentaklorfenol),-na-n,,,,,
Pentaner,-na-n,1978,600,1800,750,2000
n­Pentan,109­6-6­-0,,,,,
Pentaner,-na-n,1978,600,1800,750,2000
Pentylacetater,-na-n,2015,50,270,100,540
Piperazin* och salter,110­85-­0-*,2015,"0,03","0,1","0,08","0,3"
"Platina*, metall och svårlösliga föreningar",7440­06-­4-*,2000,,,,
–totaldamm,-na-n,,−,1,−,−
Platina lösliga föreningar (som Pt),-na-n,2000,,,,
–totaldamm,-na-n,,−,2,−,−
n­Propanol,71­2-3­-8,1989,150,350,250,600
Propen,115­0-7­-1,1996,500,900,−,−
β­Propiolakton,57­5-7­-8,,,,,
Propionsyra,79­0-9­-4,2015,10,30,20,62
Propylacetat,109­6-0­-4,1996,100,400,200,800
"1,2­Propylenglykoldinitrat",6423­4-3­-4,1987,"0,1","0,7","0,3",2
"Propylenglykolmonometyleter,",1320­6-7­-8,1990,50,190,75,300
 Propylenoxid,75­5-6­-9,2018,1,"2,4",5,"12,5"
Pyretrum,8003­3-4­-7,2015,−,1,−,−
Pyridin,110­8-6­-1,1996,2,7,3,10
Radon,10043­9-2­-2,2011,,,,
Resorcinol,108­4-6­-3,1993,10,45,−,−
Salpetersyra,7697­3-7­-2,2015,"0,5","1,3",1,"2,6"
Saltsyra,7647­0-1­-0,2015,2,3,4,6
"Selen*, och oorg. föreningar (som Se) utom",7782­49-­2-*,1990,,,,
väteselenid,-na-n,,,,,
–totaldamm,-na-n,,−,"0,1",−,−
Sevofluran,28523­8-6­-6,2000,10,80,20,170
"Silver*, metall och svårlösliga",7440­22-­4-*,1990,,,,
föreningar (som Ag),-na-n,,,,,
–totaldamm,-na-n,,−,"0,1",−,−
"Silver, lösliga föreningar (som Ag)",-na-n,1990,,,,
–totaldamm,-na-n,,−,"0,01",−,−
 Skärvätska,-na-n,2018,,,,
Stearater,-na-n,1996,,,,
–totaldamm,-na-n,,−,5,−,−
Styren,100­4-2­-5,2011,10,43,20,86
"Sulfider, (summan av dimetyldisulfid, dime­",-na-n,1993,1,−,−,−
tylsulfid och metantiol),-na-n,,,,,
Sulfotep,3689­2-4­-5,2015,−,"0,1",−,−
 Svaveldioxid,7446­0-9­-5,2018,"0,5","1,3",1,"2,7"
Svavelhexafluorid,2551­6-2­-4,1993,1000,6000,−,−
Svavelsyra,7664­9-3­-9,2011,,,,
–inhalerbar fraktion,-na-n,,−,"0,1",−,"0,2"
Svaveltetrafluorid,7783­6-0­-0,1993,−,−,"0,1","0,4"
Talk,14807­9-6­-6,1996,,,,
– totaldamm,-na-n,,−,2,−,−
– respirabel  fraktion,-na-n,,−,1,−,−
Tellur* metall och föreningar (som Te),13494­80-­9-*,1981,,,,
–totaldamm,-na-n,,−,"0,1",−,−
Tenn* metall och oorg. föreningar,7440­31-­5-*,2011,,,,
(som Sn),-na-n,,,,,
–inhalerbar fraktion,-na-n,,−,2,−,−
Tennorganiska föreningar (som Sn),-na-n,1978,,,"H,V",3
–totaldamm,-na-n,,"−

0,1","−

0,2",,
" Terfenyl, hydrerad",61788­3-2­-7,2018,"2 
19","5 
48",,
Terpener,-na-n,1990,"25 
150","50 
300","S,V",
Terpentin,8006­6-4­-2,1990,"25 
150","50 
300","H,S,V",
"1,1,2,2­Tetrabrometan",79­2-7­-6,1993,"1 
14","2 
30",V,
Tetraetylbly (som Pb),78­0-0­-2,1981,"−
0,05","−
0,2","H,R,V",
 Tetraetylortosilikat,78­1-0­-4,2018,"5 
44","10 
86",,
"1,1,1,2­Tetrafluoretan",-na-n,,,,,
Se: HFC 134a,-na-n,,,,,
"1,2,2,2­Tetrafluoroetyldifluormetyleter",-na-n,,,,,
Se: Desfluran,-na-n,,,,,
Tetrahydroftalsyraanhydrid,85­4-3­-8,,,,"M,S","4,32"
,935­7-9­-5,,,,,
Tetrahydrofuran,109­9-9­-9,2015,"50 
150","100 
300",,
 Tetrakloretylen,127­1-8­-4,2018,"10 
70","25 
170","C,H",
Tetraklorfenol* och salter,25167­83-­3-*,1990,"−

0,5","−

1,5","H,V",
Tioglykolsyra,68­1-1­-1,1996,1,4,2,8
Tiram,137­2-6­-8,1993,−,1,−,2
Titandioxid,13463­6-7­-7,1990,,,,
–totaldamm,-na-n,,−,5,−,−
"o­Tolidin (3,3’­dimetylbensidin)",119­9-3­-7,,,,,
Toluen,108­8-8­-3,2015,50,192,100,384
 o­Toluidin,95­5-3­-4,2018,"0,1","0,5",−,−
Tridymit,15468­3-2­-3,1996,,,,
–respirabel fraktion,-na-n,,−,"0,05",−,−
Trietanolamin,102­7-1­-6,2011,"0,8",5,"1,6",10
Trietylamin,121­4-4­-8,2015,1,"4,2",3,"12,6"
Trietylentetramin,112­2-4­-3,1984,1,6,2,12
"1,1,1­Trifluoretan",420­4-6­-2,2005,500,1750,750,2625
"1,2,4­Triklorbensen",120­8-2­-1,2015,2,15,5,38
"1,1,1­Trikloretan",71­5-5­-6,2015,50,300,200,1110
" 1,1,2­Trikloretylen",79­0-1­-6,2018,10,54,25,140
Triklorfenol* och salter,25167­82-­2-*,"−

1990 
0,5",−,"1,5","C,H,V",
Vanadinpentoxid (som V),1314­6-2­-1,1987,,,,
– totaldamm,-na-n,,−,"0,2",−,−
– respirabel  fraktion,-na-n,,−,−,−,"0,05"
Vinylacetat,108­0-5­-4,1993,5,18,10,35
 Vinylbromid,593­6-0­-2,2018,1,"4,4",−,−
Vinylidenklorid,-na-n,,,,,
"Se: 1,1­Dikloreten",-na-n,,,,,
 Vinylklorid,75­0-1­-4,2018,1,"2,5",5,13
Vinyltoluen,25013­1-5­-4,1993,10,50,30,150
2­Vinyltoluen,611­1-5­-4,,,,,
Väteklorid,-na-n,,,,,
Se: Saltsyra,-na-n,,,,,
Väteperoxid,7722­8-4­-1,1990,"1 
1,4","2 
3",,
Väteselenid,7783­0-7­-5,2015,"0,01 
0,03","0,05 
0,2",,
Vätesulfid,7783­0-6­-4,2015,"5 
7","10 
14",,
Wollastonit,-na-n,,,,,
"Se: Fibrer, naturliga kristallina­Övriga",-na-n,,,,,
Xylen,1330­2-0­-7,2015,"50 
221","100 
442",H,
o­Xylen,95­4-7­-6,,,,,

```

# README.md

```md
# Franks-Chemical-Simulator
Chemical Engineering Design Software 
This is a simple GUI on many topics in chemical engineering design.
-Process Safety
-Process Economics
-Heat Exchanger Design
-Chemical Data
-Flowsheet Simulations

```

# requirements.txt

```txt
pygame==2.5.0
csv==1.0
pickle==4.0
scipy==1.11.1
thermo==0.2.26
fluids==1.0.24
tkinter==8.6
fpdf==1.7.2
pandas==2.0.3
matplotlib==3.7.2
chemicals==1.1.4
openpyxl
scikit-learn
pyswarms
pysindy

```

# results/Pipe_Dimension_Results.xlsx

This is a binary file of the type: Excel Spreadsheet

# setup_db.py

```py
try:
    # At the top of your start.py script, after importing necessary modules
    import psycopg2
    import getpass
    import os
    import subprocess
    import sys
    import importlib

    print("Import Success")
except:
    print("Import Error")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

current_directory = os.path.dirname(os.path.abspath(__file__))

required_libraries = [
    "subprocess",
    "psycopg2",
    "getpass",
    "os",
    "importlib",
    "sys",
    # any other libraries you want to ensure are installed
]

missing_libraries = []

for library in required_libraries:
    try:
        importlib.import_module(library)
        print(f"{library} is installed.")
    except ImportError:
        print(f"{library} is not installed.")
        missing_libraries.append(library)

if missing_libraries:
    print("Attempting to install missing libraries...")
    subprocess.run(["pip", "install", *missing_libraries])
    print("Installation of missing libraries completed.")

def run_psql_command(command, env):
    try:
        result = subprocess.run(['psql', '-c', command], env=env, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")
        return None

def create_role_and_database(superuser, superuser_password, role_name, role_password, db_name):
    try:
        env = {'PGUSER': superuser, 'PGPASSWORD': superuser_password}
        
        # Check if role already exists
        role_check = run_psql_command(f"SELECT 1 FROM pg_roles WHERE rolname='{role_name}';", env)
        if role_check and '1 row' in role_check:
            print(f"Role {role_name} already exists.")
        else:
            # Create role
            run_psql_command(f"CREATE ROLE {role_name} WITH LOGIN PASSWORD '{role_password}';", env)
            print(f"Role {role_name} created.")

        # Check if database already exists
        db_check = run_psql_command(f"SELECT 1 FROM pg_database WHERE datname='{db_name}';", env)
        if db_check and '1 row' in db_check:
            print(f"Database {db_name} already exists.")
        else:
            # Create database
            run_psql_command(f"CREATE DATABASE {db_name};", env)
            run_psql_command(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {role_name};", env)
            print(f"Database {db_name} created and privileges granted to {role_name}.")

    except Exception as e:
        print(f"An error occurred during database setup: {e}")

def write_db_config(db_name, role_name, role_password):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    with open(f'{current_directory}/chemical_databases/db_config.txt', 'w') as file:
        file.write(f"dbname={db_name}\n")
        file.write(f"user={role_name}\n")
        file.write(f"password={role_password}\n")
        file.write("host=localhost\n")
    print("Database configuration file written.")


def detect_os():
    if sys.platform.startswith('darwin'):
        return 'macOS'
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        return 'Windows'
    elif sys.platform.startswith('linux'):
        return 'Linux'
    else:
        return 'Unknown'

def is_postgresql_installed():
    try:
        subprocess.run(['psql', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def main():
    print("Welcome to the Database Setup for Frank's Chemical Process Simulator")

    # Detect OS and check PostgreSQL installation
    os_type = detect_os()
    print(f"Detected operating system: {os_type}")

    if not is_postgresql_installed():
        print("PostgreSQL is not installed or not added to PATH. Please install PostgreSQL and ensure 'psql' is accessible in your PATH.")
        if os_type == 'macOS':
            print("Attempting to add PostgreSQL to PATH for this session.")
            os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
            if is_postgresql_installed():
                print("PostgreSQL found after updating PATH.")
            else:
                print("PostgreSQL still not found. Please ensure it's installed and '/opt/homebrew/bin' is in your PATH.")
                return
            print("On macOS, you can install PostgreSQL using Homebrew:")
            print("  brew install postgresql")
            print("  brew services start postgresql")
            print("After installation, ensure that 'psql' is in your PATH.")
        elif os_type == 'Linux':
            print("On Linux, you can install PostgreSQL using your distribution's package manager.")
            print("For example, on Ubuntu:")
            print("  sudo apt-get update")
            print("  sudo apt-get install postgresql postgresql-contrib")
        elif os_type == 'Windows':
            print("On Windows, download and install PostgreSQL from the official website.")
            print("Ensure that the installation path is added to your system's PATH.")
        return

    print("Welcome to the Database Setup for Frank's Chemical Process Simulator")
    
    # Prompt for PostgreSQL superuser username and password
    superuser = input("Enter your PostgreSQL superuser username (typically 'postgres'): ")
    superuser_password = getpass.getpass("Enter your PostgreSQL superuser password: ")

    # Prompt for new role and database details
    role_name = input("Enter a new role name for the application (e.g., 'User'): ")
    role_password = getpass.getpass("Enter a password for the new role: ")
    db_name = input("Enter the name for the new database (e.g., 'DATA'): ")

    create_role_and_database(superuser, superuser_password, role_name, role_password, db_name)
    write_db_config(db_name, role_name, role_password)

if __name__ == "__main__":
    main()
```

# start.py

```py
# This part is for checking whats missing
try:
    # At the top of your start.py script, after importing necessary modules
    import psycopg2
    import pygame
    import csv
    import pickle
    import math
    from enum import Enum
    import os
    import subprocess
    import importlib
    import chemicals
    import fluids
    import thermo
    import ht
    print("Import Success")
except:
    print("Import Error")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

current_directory = os.path.dirname(os.path.abspath(__file__))

required_libraries = [
    "subprocess",
    "importlib",
    "pygame",
    "scipy",
    "thermo",
    "fluids",
    "ht",
    "numpy",
    "csv",
    "pickle",
    "math",
    "enum",
    "os",
    "sys",
    "time",
    "random",
    "datetime",
    "tkinter",
    "fpdf",
    "pandas",
    "matplotlib",
    "chemicals",
    "openpyxl",
    #"scikitlearn",
    "pyswarms",
    "pysindy",
    "psycopg2"
    # any other libraries you want to ensure are installed
]

missing_libraries = []

for library in required_libraries:
    try:
        importlib.import_module(library)
        print(f"{library} is installed.")
    except ImportError:
        print(f"{library} is not installed.")
        missing_libraries.append(library)

if missing_libraries:
    print("Attempting to install missing libraries...")
    subprocess.run(["pip", "install", *missing_libraries])
    print("Installation of missing libraries completed.")


# Function to get database connection
def get_db_connection(config_file='chemical_databases/db_config.txt'):
    try:
        # Initialize a dictionary to hold the configuration
        config = {}
        
        # Read configuration from file
        with open(config_file, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                config[key] = value

        # Establish the database connection
        return psycopg2.connect(
            dbname=config['dbname'],
            user=config['user'],
            password=config['password'],
            host=config['host']
        )

    except Exception as e:
        print("Error while connecting to the database:", e)
        return None

# Function to check and create the database if it doesn't exist
def setup_database():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        # Create tables if they don't exist (example for a chemical_data table)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chemical_data (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                density FLOAT,
                viscosity FLOAT,
                ...
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("Database setup completed.")
    else:
        print("Database setup failed.")
        print("Please try running setup_db.py")

# Run the database setup function
setup_database()

flowsheet_version = "Flowsheet Simulator v.1.0.0"
# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128, 128)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# Set up the Pygame window
scale_factor = 0.8
window_width = int(screen_width * scale_factor)
window_height = int(screen_height * scale_factor)
window_size = (window_width, window_height)
WINDOW_SIZE = (screen_width, screen_height)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Frank's Chemical Process Simulator")

# Set up fonts
font = pygame.font.Font(None, 40)

# Define button size as a percentage of the window size
button_width_percent = 0.2
button_height_percent = 0.1

# Calculate button size based on the window size
button_width = int(window_width * button_width_percent)
button_height = int(window_height * button_height_percent)


# Define buttons
button_padding = 50
button_start_x = (WINDOW_SIZE[0] - button_width) / 4
button_start_y = (WINDOW_SIZE[1] - (button_height + button_padding) * 5) / 2

# Define the font size as a percentage of the window size
font_size_percent = 0.05

# Calculate the font size based on the window size
font_size = int(window_height * font_size_percent)

# Create a font object
font = pygame.font.SysFont(None, font_size)

# Defines the abstract class called page
class Page:
    """an abstract class representing a single screen"""

    def render(self, pygame_screen):
        pass
# Defines the class called PageManager
class PageManager:
    def __init__(self, initial_page):
        self.current_page = initial_page
        self.current_page.manager = self  # set the manager attribute of the initial page
        self.running = True

    def go_to(self, page):
        self.current_page = page
        self.current_page.manager = self  # set the manager attribute of the new page

    def handle_event(self, event):
        self.current_page.handle_event(event)

    def render(self, screen):
        self.current_page.render(screen)
# Defines the Main Menu pages
class MainMenuPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.in_data_processing = True
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.in_statistics = True
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.in_thermodynamics = True
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        elif self.in_thermodynamics:
            self.manager.go_to(ThermodynamicsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Data Processing Page
class DataProcessingPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(5)]
        
        # New Buttons
        #self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Matrix Interpolation Filler", "Remove Duplicates", "Temporal Resampling", "Subplotter", "Back"]
        
        # New Button Texts
        #self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_back = False
        self.in_matrix_interpolation_filler = False
        self.in_remove_duplicates = False
        self.in_temporal_resampling = False
        self.in_subplotter = False
        self.in_back = False



        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_matrix_interpolation_filler = True
                    elif i == 1:
                        self.in_remove_duplicates = True
                    elif i == 2:
                        self.in_temporal_resampling = True
                    elif i == 3:
                        self.in_subplotter = True
                    elif i == 4:
                        self.in_back = True
                        self.manager.go_to(MainMenuPage())
        if self.in_matrix_interpolation_filler:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Matrix_Filler_Interpolation.py")])
            self.in_matrix_interpolation_filler = False
        elif self.in_remove_duplicates:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Remove_Duplicates.py")])
            self.in_remove_duplicates = False
        elif self.in_temporal_resampling:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Temporal_Resampling.py")])
            self.in_temporal_resampling = False
        elif self.in_subplotter:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Subplotter.py")])
            self.in_subplotter = False
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
  

    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Data Pre Processing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Statistics Page
class StatisticsPage(Page):
    def __init__(self, page_manager=None):
        self.menu_texts = ["Spearman Pearson Testing",
                            "ANOVA", "Check Consistency",
                            "FFT", "Gaussian Filter",
                            "Matrix Filler Interpolate",
                            "Moving Average Filter",
                            "Random Forest -> Particle Swarm Optimization",
                            "Reduction PCA","PIML","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]
        # Existing Button Texts
        self.in_spearman_pearson = False
        self.in_anova = False
        self.in_check_consistency = False
        self.in_fft = False
        self.in_gaussian_filter = False
        self.in_matrix_filler_interpolate = False
        self.in_moving_average_filter = False
        self.in_random_forest = False
        self.in_reduction_pca = False
        self.in_PIML = False
        self.in_back = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_spearman_pearson = True
                    elif i == 1:
                        self.in_anova = True
                    elif i == 2:
                        self.in_check_consistency = True
                    elif i == 3:
                        self.in_fft = True
                    elif i == 4:
                        self.in_gaussian_filter = True
                    elif i == 5:
                        self.in_matrix_filler_interpolate = True
                    elif i == 6:
                        self.in_moving_average_filter = True
                    elif i == 7:
                        self.in_random_forest = True
                    elif i == 8:
                        self.in_reduction_pca = True
                    elif i == 9:
                        self.in_PIML = True
                    elif i == 10:
                        self.in_back = True

        if self.in_spearman_pearson:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Spearman_Pearson_v2.py")])
            self.in_spearman_pearson = False
        elif self.in_anova:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_ANOVA.py")])
            self.in_anova = False
        elif self.in_check_consistency:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Check_Consistency.py")])
            self.in_check_consistency = False
        elif self.in_fft:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_FFT_Technique.py")])
            self.in_fft = False
        elif self.in_gaussian_filter:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Gaussian_Filter.py")])
            self.in_gaussian_filter = False
        elif self.in_matrix_filler_interpolate:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Matrix_Filler_Interpolation.py")])
            self.in_matrix_filler_interpolate = False
        elif self.in_moving_average_filter:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Moving_Average_Smoothing_Filter.py")])
            self.in_moving_average_filter = False
        elif self.in_random_forest:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Optimizer_RandomForest_ParticleSwarm.py")])
            self.in_random_forest = False
        elif self.in_reduction_pca:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Reduction_PCA_SVD.py")])
            self.in_reduction_pca = False
        elif self.in_PIML:
            self.manager.go_to(PIMLPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Industrial Statistics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Physics Informed Machine Learning Page
class PIMLPage(Page):
    def __init__(self, page_manager=None):
        self.menu_texts = ["Genetic Programming","SINDY","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]
        # Existing Button Texts
        self.in_genetic_programming = False
        self.in_SINDY = False
        self.in_back = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_genetic_programming = True
                    elif i == 1:
                        self.in_SINDY = True
                    elif i == 2:
                        self.in_check_consistency = True
                    elif i == 3:
                        self.in_back = True

        if self.in_genetic_programming:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_PIML_Genetic_Programming.py")])
            self.in_genetic_programming = False
        elif self.in_SINDY:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_PIML_SINDy.py")])
            self.in_SINDY = False
        elif self.in_back:
            self.manager.go_to(StatisticsPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()

        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Physics Informed Machine Learning Statistics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Thermodynamics Page
class ThermodynamicsPage(Page):
    def __init__(self, page_manager=None):

        self.menu_texts = ["Air Cooler Design","Ammonia Gas Storage",
                            "Combustion","Compressor Power",
                            "Ethylene Expansion","Isentropic Air Compression",
                            "Isentropic Oxygen Compression","Joule-Thomson Methane",
                            "Liquid Nitrogen","Material Properties",
                            "Multi Component Flash","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]

        self.in_air_cooler_design = False
        self.in_ammonia_gas_storage = False
        self.in_combustion = False
        self.in_compressor_power = False
        self.in_ethylene_expansion = False
        self.in_isentropic_air_compression = False
        self.in_isentropic_oxygen_compression = False
        self.in_joule_thomson_methane = False
        self.in_liquid_nitrogen = False
        self.in_material_properties = False
        self.in_multi_component_flash = False
        self.in_back = False
        self.in_thermodynamics_page = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_air_cooler_design = True
                    elif i == 1:
                        self.in_ammonia_gas_storage = True
                    elif i == 2:
                        self.in_combustion = True
                    elif i == 3:
                        self.in_compressor_power = True
                    elif i == 4:
                        self.in_ethylene_expansion = True
                    elif i == 5:
                        self.in_isentropic_air_compression = True
                    elif i == 6:
                        self.in_isentropic_oxygen_compression = True
                    elif i == 7:
                        self.in_joule_thomson_methane = True
                    elif i == 8:
                        self.in_liquid_nitrogen = True
                    elif i == 9:
                        self.in_material_properties = True
                    elif i == 10:
                        self.in_multi_component_flash = True
                    elif i == 11:
                        self.in_back = True
                    
        if self.in_air_cooler_design:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Air_Cooler_Designer.py")])
            self.in_air_cooler_design = False
        elif self.in_ammonia_gas_storage:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Ammonia_Gas_Storage_Sizing.py")])
            self.in_ammonia_gas_storage = False
        elif self.in_combustion:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Combustion_Calculations.py")])
            self.in_combustion = False
        elif self.in_compressor_power:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Compressor_Power_Sizing.py")])
            self.in_compressor_power = False
        elif self.in_ethylene_expansion:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Ethylene_Expansion.py")])
            self.in_ethylene_expansion = False
        elif self.in_isentropic_air_compression:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Isentropic_Air_Compression.py")])
            self.in_isentropic_air_compression = False
        elif self.in_isentropic_oxygen_compression:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Isentropic_Oxygen_Compression.py")])
            self.in_isentropic_oxygen_compression = False
        elif self.in_joule_thomson_methane:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Joule_Thomson_Methane.py")])
            self.in_joule_thomson_methane = False
        elif self.in_liquid_nitrogen:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Liquid_Nitrogen_Production.py")])
            self.in_liquid_nitrogen = False
        elif self.in_material_properties:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Material_Properties.py")])
            self.in_material_properties = False
        elif self.in_multi_component_flash:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Multi_Component_Flash.py")])
            self.in_multi_component_flash = False
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
            self.in_back = False
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Thermodynamics Page", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Flow Sheet Simulation Page Buttons
class FlowsheetSimulationPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
        self.menu_texts = [
        "New Flowsheet",
        "Load Flowsheet",
        "Back"
    ]
        self.in_new_flowsheet = False
        self.in_load_flowsheet = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_new_flowsheet = True
                    elif i == 1:
                        self.in_load_flowsheet = True
                    elif i == 2:
                        self.in_back = True
        if self.in_new_flowsheet:
            self.manager.go_to(NewFlowsheetPage(page_manager=self.manager))
        elif self.in_load_flowsheet:
            self.manager.go_to(LoadFlowsheetPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Flowsheet Simulator v1.0.0", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Run Flow sheet Page
class RunFlowsheetSimulation(Page):
    def __init__(self, filename, screen, page_manager):
        super().__init__()
        print("Initializing RunFlowsheetSimulation...")
        self.page_manager = page_manager
        self.filename = filename
        self.flowsheet = Flowsheet("My Flowsheet")
        self.grid = None
        self.recently_saved = False
        self.paused = False
        self.save_as = False
        self.go_to_main_menu = False
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.flowsheet_renderer = FlowsheetRenderer(self.flowsheet)
        self.placing_pump = False
        self.placing_tank = False
        self.return_to_main_menu = False  # Add this line
        tab_width = 200
        tab_height = self.screen.get_height()
        self.tab_rect = pygame.Rect(self.screen.get_width() - tab_width, 0, tab_width, tab_height)

        # Load the Flowsheet object from the file
        with open(self.filename, 'rb') as f:
            self.flowsheet = pickle.load(f)

        # Load blocks from the loaded flowsheet
        for block_instance in self.flowsheet.blocks:
            block_type = type(block_instance)
            block = {
                "type": block_type.value,
                "rect": pygame.Rect(block_instance.x, block_instance.y, 150, 50),
                "instance": block_instance,
            }
            self.block_list.append(block)

        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128, 128)

        # Define fonts
        self.font = pygame.font.SysFont(None, 48)

        # Initialize dragged_block
        self.dragged_block = None

        # Define pause menu buttons
        self.pause_buttons = [
            {"label": "Save", "rect": pygame.Rect(100, 100, 300, 50)},
            {"label": "Save As", "rect": pygame.Rect(100, 200, 300, 50)},
            {"label": "Iterator", "rect": pygame.Rect(100, 300, 300, 50)},
            {"label": "Quit to Main", "rect": pygame.Rect(100, 400, 300, 50)},
            {"label": "Quit to Desktop", "rect": pygame.Rect(100, 500, 300, 50)}
        ]
        # Block types
        self.block_types = list(BlockType)
        self.block_list = []
        # Menu
        self.menu_open = False
        self.menu_rect = pygame.Rect(100, 100, 200, 200)
        # Mouse
        self.left_click = False
        self.clicked_on_block = False
        # Initialize dragged_block
        self.dragged_block = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.paused = not self.paused
            elif event.key == pygame.K_TAB:
                self.menu_open = not self.menu_open
                if self.menu_open:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.menu_rect.x = mouse_x
                    self.menu_rect.y = mouse_y
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if self.tab_rect.collidepoint(event.pos):
                # Handle tab menu clicks
                for index, button in enumerate(self.tab_buttons):
                    if button["rect"].collidepoint(event.pos):
                        if button["type"] == "Centrifugal Pump":
                            self.placing_pump = True
                        elif button["type"] == "Tank":
                            self.placing_tank = True
                        else:
                            self.placing_pump = False
                            self.placing_tank = False
                        
            elif self.placing_pump:
                # Place centrifugal pump on the flowsheet
                pump_rect = pygame.Rect(event.pos[0] - 20, event.pos[1] - 20, 40, 40)
                pump_instance = {"type": BlockType.CentrifugalPump.value, "rect": pump_rect}
                self.block_list.append(pump_instance)
                self.placing_pump = False
        if self.paused:
            # Handle pause menu events
            self.draw_pause_menu()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.pause_buttons[0]['rect'].collidepoint(event.pos):
                    # Save Has been clicked
                    
                    with open(self.filename, 'wb') as f:
                        pickle.dump(self.flowsheet, f)
                    print("Game Sucessfully Saved")
                    self.recently_saved = True
                elif self.pause_buttons[1]['rect'].collidepoint(event.pos):
                    # Save as has been clicked
                    print("Save As")
                    self.save_as = True
                elif self.pause_buttons[2]['rect'].collidepoint(event.pos):
                    # Iterator has been pressed
                    print("Iterator has been pressed")
                # Check if the user clicked the "Quit to Main" button
                elif self.pause_buttons[3]["label"] == "Quit to Main" and self.pause_buttons[3]["rect"].collidepoint(mouse_x, mouse_y):
                    print("Back to Main Menu")
                    self.return_to_main_menu = True  # Update this line

                elif self.pause_buttons[4]['rect'].collidepoint(event.pos):
                    # Quit to desktop has been pressed
                    print("Quit to desktop")
                    pygame.quit()

            # Handle save as input box events
            if self.save_as:
                # Create an input box for the filename
                input_rect = pygame.Rect(100, 300, 400, 50)
                input_text = ""
                # Draw the input box and text
                overlay = pygame.Surface((self.screen_width, self.screen_height))
                overlay.fill(self.GRAY)
                overlay.set_alpha(128)
                screen.blit(overlay, (0, 0))
                pygame.draw.rect(screen, self.WHITE, input_rect)
                pygame.display.update(input_rect)

                while self.save_as:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.save_as = False
                            elif event.key == pygame.K_RETURN:
                                # Save the file and return to the game
                                filename = input_text.strip() + ".pkl"
                                with open(filename, "wb") as f:
                                    pickle.dump(self.flowsheet, f)
                                print("Game saved as", filename)
                                self.save_as = False
                            elif event.key == pygame.K_BACKSPACE:
                                input_text = input_text[:-1]
                                input_surface = self.font.render(input_text, True, self.BLACK)
                                pygame.draw.rect(screen, self.WHITE, input_rect)
                                screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                                pygame.draw.rect(screen, self.WHITE, input_rect, 2)
                                pygame.display.update(input_rect)

                            elif event.key not in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                                input_text += event.unicode

                            input_surface = self.font.render(input_text, True, self.BLACK)
                            screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                            pygame.draw.rect(screen, self.WHITE, input_rect, 2)
                            pygame.display.update(input_rect)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.menu_open = not self.menu_open
                    if self.menu_open:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        self.menu_rect.x = mouse_x
                        self.menu_rect.y = mouse_y      
        if self.menu_open:
            self.draw_menu()
            if self.left_click:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                for i, block_type in enumerate(self.block_types):
                    if self.menu_rect.collidepoint(mouse_x, mouse_y) and (
                        self.menu_rect.y + 10 + i * 40 <= mouse_y < self.menu_rect.y + 10 + (i + 1) * 40
                    ):
                        self.add_block(block_type, mouse_x, mouse_y)
                        self.menu_open = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.left_click = True
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    if not self.menu_open:
                        for block in self.block_list:
                            if block["rect"].collidepoint(mouse_x, mouse_y):
                                self.dragged_block = block
                                break
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.left_click = False
                    self.dragged_block = None
        else:
            if self.left_click and self.dragged_block is not None:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.dragged_block["rect"].x = mouse_x - self.dragged_block["rect"].width // 2
                self.dragged_block["rect"].y = mouse_y - self.dragged_block["rect"].height // 2
    def draw_pause_menu(self):
        # Get the screen dimensions
        screen_width, screen_height = self.screen.get_size()
        # Create a semi-transparent overlay
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((128, 128, 128, 128))
        # Draw the pause menu buttons on the overlay
        for button in self.pause_buttons:
            pygame.draw.rect(overlay, self.WHITE, button["rect"])
            text = self.font.render(button["label"], True, self.BLACK)
            text_rect = text.get_rect(center=button["rect"].center)
            overlay.blit(text, text_rect)

        # Draw the overlay on the screen
        self.screen.blit(overlay, (0, 0))
    def draw_menu(self):
        pygame.draw.rect(self.screen, self.WHITE, self.menu_rect)
        for i, block_type in enumerate(self.block_types):
            text = self.font.render(block_type.value, True, self.BLACK)
            self.screen.blit(text, (self.menu_rect.x + 10, self.menu_rect.y + 10 + i * 40))
    def add_block(self, block_type, x, y):
        # Create block instance based on block_type
        if block_type == BlockType.Tank:
            block = Tank(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.CentrifugalPump:
            block = CentrifugalPump(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.HandValve:
            block = HandValve(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.ControlValve:
            block = ControlValve(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.ReliefValve:
            block = ReliefValve(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.FlashTank:
            block = FlashTank(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.DistillationColumn:
            block = DistillationColumn(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.GeneralHeatExchanger:
            block = GeneralHeatExchanger(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.STHeatExchanger:
            block = STHeatExchanger(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.PlateHeatExchanger:
            block = PlateHeatExchanger(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        self.flowsheet.blocks.append(block)
        print(f"Added {block.name} at position {block.position}")  # Add this line for debugging

        block.size = (150, 50)

        block_info = {"type": block_type.value, "rect": pygame.Rect(x, y, 150, 50), "instance": block}
        self.block_list.append(block_info)
    def draw_blocks(self):
        for block in self.block_list:
            if block["type"] == BlockType.CentrifugalPump.value:
                self.flowsheet_renderer.draw_centrifugal_pump(self.screen, block["rect"].center)
                print("Drawing Pump")
            elif block["type"] == BlockType.Tank.value:
                self.flowsheet_renderer.draw_storage_tank(self.screen, block["rect"].center)
                print("Drawing Tank")
            
            else:
                pygame.draw.rect(self.screen, self.BLUE, block["rect"])
                text = self.font.render(block["instance"].name, True, self.BLACK)  # Update this line to use block's name instead of its type
                self.screen.blit(text, (block["rect"].x + 10, block["rect"].y + 10))
    def run(self):
        # Initialize pygame elements
        print("Running RunFlowsheetSimulation...")  # Debugging line
        self.menu_open = False
        self.menu_rect = pygame.Rect(100, 100, 200, 200)
        self.block_types = list(BlockType)
        self.block_list = []
        self.dragged_block = None
        self.flowsheet = Flowsheet("My Flowsheet")
        self.block_list = []

        # Load blocks from the flowsheet
        for block_instance in self.flowsheet.blocks:
            block_type = type(block_instance)
            block = {
                "type": block_type.value,
                "rect": pygame.Rect(block_instance.x, block_instance.y, 150, 50),
                "instance": block_instance,
            }
            self.block_list.append(block)
    def render(self, screen):
        self.screen.fill(self.WHITE)
        self.flowsheet_renderer.render(self.screen)

        if self.menu_open:
            self.draw_menu()

        else:
            if self.left_click and self.dragged_block is not None:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.dragged_block["rect"].x = mouse_x - self.dragged_block["rect"].width // 2
                self.dragged_block["rect"].y = mouse_y - self.dragged_block["rect"].height // 2


        if self.paused:
            self.draw_pause_menu()
        # Draw centrifugal pumps
        for block in self.flowsheet.blocks:
            if isinstance(block, CentrifugalPump):
                self.flowsheet_renderer.draw_centrifugal_pump(screen, block.position)
            elif isinstance(block, Tank):
                self.flowsheet_renderer.draw_storage_tank(screen, block.position[0], block.position[1], width=30, height=100, border_width=2)
        pygame.display.update()
# Define the New flosheet Page
class NewFlowsheetPage(Page):
    def __init__(self, page_manager=None):
        self.input_rect = pygame.Rect(100, 300, 400, 50)
        self.input_text = ""
        self.back_rect = pygame.Rect(100, 400, 100, 50)  # Back button rectangle
        self.back_text = "Back"
        self.in_simulation = False
        self.in_back = False
        self.new_flowsheet_created = False

    def handle_event(self, event):
        # Handle keyboard events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Create a new Flowsheet object with the entered name and save it
                filename = self.input_text.strip() + ".pkl"
                print("Creating new flowsheet:", filename)
                flowsheet = Flowsheet(self.input_text.strip())
                with open(filename, "wb") as f:
                    pickle.dump(flowsheet, f)
                self.manager.go_to(RunFlowsheetSimulation(filename, screen, page_manager=self.manager))
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
                self.update_input_field()
            elif event.unicode.isalnum():
                self.input_text += event.unicode
                self.update_input_field()

        # Handle mouse events
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_rect.collidepoint(event.pos):
                self.in_back = True

        pygame.display.update()


    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_text, True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rect)
        screen.blit(input_surface, (self.input_rect.x + 5, self.input_rect.y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

    def render(self, pygame_screen):
        screen.fill(WHITE)

        title = font.render("New Flowsheet", True, BLACK)
        screen.blit(title, (WINDOW_SIZE[1], 100))

        input_title = font.render("Type the name of the new flowsheet and hit Enter:", True, BLACK)
        screen.blit(input_title, (100, 200))

        pygame.draw.rect(screen, WHITE, self.input_rect)
        self.update_input_field()  # Call the update_input_field() method here
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

        # Draw the back button
        pygame.draw.rect(screen, GRAY, self.back_rect)
        back_button = font.render(self.back_text, True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect.center
        screen.blit(back_button, back_button_rect)

        pygame.display.update(self.input_rect)

        # Check if the "Back" button has been pressed
        if self.in_back:
            self.in_back = False
            self.manager.go_to(FlowsheetSimulationPage())
# Define Load Flowsheet Page
class LoadFlowsheetPage(Page):
    def __init__(self, page_manager=None):
        self.input_rect = pygame.Rect(100, 300, 400, 50)
        self.input_text = ""
        self.back_rect = pygame.Rect(100, 400, 100, 50)  # Back button rectangle
        self.back_text = "Back"
        self.in_simulation = False
        self.in_back = False
        self.file_loaded = False

    def handle_event(self, event):
        # Handle keyboard events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Check if the entered file exists
                filename = self.input_text.strip() + ".pkl"
                if os.path.exists(filename):
                    print("Loading flowsheet:", filename)
                    self.manager.go_to(RunFlowsheetSimulation(filename, screen, page_manager=self.manager))
                else:
                    print("File does not exist")
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
                self.update_input_field()
            elif event.unicode.isalnum():
                self.input_text += event.unicode
                self.update_input_field()

        # Handle mouse events
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_rect.collidepoint(event.pos):
                self.in_back = True

        pygame.display.update()

    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_text, True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rect)
        screen.blit(input_surface, (self.input_rect.x + 5, self.input_rect.y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

    def render(self, pygame_screen):
        screen.fill(WHITE)

        title = font.render("Load Flowsheet", True, BLACK)
        screen.blit(title, (WINDOW_SIZE[1], 100))

        input_title = font.render('Type the name of the flowsheet (.pkl file) and hit Enter:', True, BLACK)
        screen.blit(input_title, (100, 200))
        input_title2 = font.render('(A flowsheet must be created first in "New Flowsheet" page)', True, BLACK)
        screen.blit(input_title2, (100, 250))

        pygame.draw.rect(screen, WHITE, self.input_rect)
        self.update_input_field()  # Call the update_input_field() method here
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

        # Draw the back button
        pygame.draw.rect(screen, GRAY, self.back_rect)
        back_button = font.render(self.back_text, True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect.center
        screen.blit(back_button, back_button_rect)

        pygame.display.update(self.input_rect)

        # Check if the "Back" button has been pressed
        if self.in_back:
            self.in_back = False
            self.manager.go_to(FlowsheetSimulationPage())
# Define Equiptment Sizing Page
class EquipmentSizingPage(Page):
    def __init__(self,page_manager = None):
        start_x = button_start_x
        self.menu_rects = [
        pygame.Rect(start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 5, button_width, button_height),
        pygame.Rect(start_x, button_start_y, button_width, button_height)
    ]
        self.menu_texts = [
        "Heat Exchangers",
        "Separations",
        "Pumps",
        "Reactors",
        "Controls",
        "Back",
        "Vessels"
    ]
        self.in_heat_exchangers= False
        self.in_separations = False
        self.in_pumps = False
        self.in_reactos = False
        self.in_controls = False
        self.in_back = False
        self.in_vessels = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_heat_exchangers = True
                    elif i == 1:
                        self.in_separations = True
                    elif i == 2:
                        self.in_pumps = True
                    elif i == 3:
                        self.in_reactos = True
                    elif i == 4:
                        self.in_controls = True
                    elif i == 5:
                        self.in_back = True
                    elif i == 6:
                        self.in_vessels = True
        if self.in_heat_exchangers:
            self.manager.go_to(HeatExchangerPage())
        elif self.in_separations:
            self.manager.go_to(SeparationsPage())
        elif self.in_pumps:
            self.manager.go_to(PumpsPage())
        elif self.in_reactos:
            self.manager.go_to(ReactorsPage())
        elif self.in_controls:
            self.manager.go_to(ControlsPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
        elif self.in_vessels:
            self.manager.go_to(VesselsPage())
    def render(self, pygame_screen):
# Draw the main menu
        screen.fill(WHITE)
        text = font.render("Equiptment Sizing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Process Vessels Page
class VesselsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Pressure Vessels",
        "Open Tanks",
        "Ventilated Tanks",
        "Back"
    ]
        self.in_pressure_vessels= False
        self.in_open_tanks = False
        self.in_ventilated_tanks = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_pressure_vessels = True
                    elif i == 1:
                        self.in_open_tanks = True
                    elif i == 2:
                        self.in_ventilated_tanks = True
                    elif i == 3:
                        self.in_back = True
        if self.in_pressure_vessels:
            self.manager.go_to(PressureVesselsPage())
        elif self.in_open_tanks:
            self.manager.go_to(OpenTankSizingPage1())
        elif self.in_ventilated_tanks:
            self.manager.go_to(VentilatedTanksPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Process Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page
class PressureVesselsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
        "Boiler Vessels",
        "Expansion Vessels",
        "Generic Horizontal",
        "Generic Vertical",
        "Back"
    ]
        self.in_boiler_vessels= False
        self.in_expansion_vessels = False
        self.in_generic_horizontal = False
        self.in_generic_vertical = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_boiler_vessels = True
                    elif i == 1:
                        self.in_expansion_vessels = True
                    elif i == 2:
                        self.in_generic_horizontal = True
                    elif i == 3:
                        self.in_generic_vertical = True
                    elif i == 4:
                        self.in_back = True
        if self.in_boiler_vessels:
            self.manager.go_to(BoilerVesselPage())
        elif self.in_expansion_vessels:
            self.manager.go_to(ExpansionVesselPage())
        elif self.in_generic_horizontal:
            self.manager.go_to(GenericHorizontalVesselPage())
        elif self.in_generic_vertical:
            self.manager.go_to(GenericVerticalVesselPage())
        elif self.in_back:
            self.manager.go_to(VesselsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Process Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page for ASME Standards
class GenericVerticalPressureVesselPageASME(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Pressure Vessels ASME", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page for SIS Standards
class GenericVerticalPressureVesselsPageSIS(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Pressure Vessel SIS", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page for EN Standards
class GenericVerticalPressureVesselsPageEN(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Pressure Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Generic Horizontal Vessels Page
class GenericHorizontalVesselPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "ASME Code",
        "SIS Code",
        "EN Code",
        "Back"
    ]
        self.in_ASME= False
        self.in_SIS = False
        self.in_EN = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_ASME = True
                    elif i == 1:
                        self.in_SIS = True
                    elif i == 2:
                        self.in_EN = True
                    elif i == 3:
                        self.in_back = True
        if self.in_pressure_vessels:
            self.manager.go_to(PressureVesselsPageASME())
        elif self.in_open_tanks:
            self.manager.go_to(PressureVesselsPageSIS())
        elif self.in_ventilated_tanks:
            self.manager.go_to(PressureVesselsPageEN())
        elif self.in_back:
            self.manager.go_to(PressureVesselsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Horizontal Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Generic Vertical Vessels Page
class GenericVerticalVesselPage(Page):
    def __init__(self) -> None:
        super().__init__()
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
            "ASME Code",
            "SIS Code",
            "EN Code",
            "Back"
        ]
        self.in_ASME= False
        self.in_SIS = False
        self.in_EN = False
        self.in_back = False
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_ASME = True
                    elif i == 1:
                        self.in_SIS = True
                    elif i == 2:
                        self.in_EN = True
                    elif i == 3:
                        self.in_back = True
        if self.in_ASME:
            self.manager.go_to(GenericVerticalPressureVesselPageASME())
        elif self.in_SIS:
            self.manager.go_to(GenericVerticalPressureVesselsPageSIS())
        elif self.in_EN:
            self.manager.go_to(GenericVerticalPressureVesselsPageEN())
        elif self.in_back:
            self.manager.go_to(PressureVesselsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Boiler Pressure Vessels Page
class BoilerVesselPage(Page):
    def __init__(self, page_manager=None):
        self.menu_texts = ["Fire Tube Boiler","Recovery Boilers","Fluidized Bed Boiler","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]
        # Existing Button Texts
        self.in_fire_tubes = False
        self.in_recovery_boilers = False
        self.in_fluidized_bed_boiler = False
        self.in_back = False
        self.manager = page_manager

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_fire_tubes = True
                    elif i == 1:
                        self.in_recovery_boilers = True
                    elif i == 2:
                        self.in_back = True

        if self.in_fire_tubes:
            self.manager.go_to(FireTubeBoilerPage())
            self.in_fire_tubes = False
        elif self.in_recovery_boilers:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_ANOVA.py")])
            self.in_anova = False
        elif self.in_back:    
            self.manager.go_to(VesselsPage())
            self.in_back = False
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Industrial Statistics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Expansion Vessels Page
class ExpansionVesselPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x / 2 - button_width / 2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]
        self.next_rect = [
            pygame.Rect(button_start_x / 2 + button_width / 2 + 900, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)

        ]

        self.back_text = ["Back"]
        self.next_text = ["Next"]
        self.input_labels = ["Temp 1 (C):", "Temp 2 (C):", " System Vol (m³):", "CAS nr:"]
        self.input_values = ["", "", "", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Expansion Volume (m³):"]
        self.output_values = [""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
        ]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.next_rect[0].collidepoint(event.pos):
                self.in_next = True
                print("Next")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(VesselsPage())
        elif self.in_next:
            self.manager.go_to(VesselsPage())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            temp2 = float(self.input_values[1])
            temp1 = float(self.input_values[0])
            system_vol = float(self.input_values[2])
            cas_nr = (self.input_values[3])

            expansion_vol = (temp2 - temp1) * system_vol * 0.0001

            self.output_values = [f"{expansion_vol:.2f}"]
        except ValueError:
            self.output_values = [""]
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        # Draw the expansion liquid in the tank
        tank_color = (173, 216, 230)  # Light Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 4))
        # Draw the tank border
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)  
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 5
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)
     # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Fluid Expansion Volume", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height//2),(tank_x-tank_width/2+100,tank_y+tank_height),"V2")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+50,tank_y+tank_height//2 + tank_height//4),(tank_x-tank_width/2+50,tank_y+tank_height),"V1")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.next_rect[0])
        next_button = font.render(self.next_text[0], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.next_rect[0].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define the Open Tanks Sizing Page 1
class OpenTankSizingPage1(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x //2 - button_width //2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]
        self.next_rect = [
            pygame.Rect(button_start_x //4 + screen_width*0.65, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)

        ]

        self.back_text = ["Back"]
        self.next_text = ["Next"]
        self.input_labels = ["In Flow (m3/s):", "Out Flow (m3/s):", "Retention Time (s):", "L/D (m/m):"]
        self.input_values = ["", "", "", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Req. Volume (m³):", "Length (m):", "Diameter (m):"]
        self.output_values = ["", "", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height)
        ]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.next_rect[0].collidepoint(event.pos):
                self.in_next = True
                print("Next")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(VesselsPage())
        elif self.in_next:
            self.manager.go_to(OpenTankSizingPage2())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            retention_time = float(self.input_values[2])
            out_flow = float(self.input_values[1])
            ld_ratio = float(self.input_values[3])

            req_volume = retention_time * out_flow
            diameter = (1/3)*(4 * req_volume*9 / (math.pi))**(1/3)
            length = diameter * ld_ratio

            self.output_values = [f"{req_volume:.2f}", f"{length:.2f}", f"{diameter:.2f}"]
        except ValueError:
            self.output_values = ["", "", ""]

    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)

        
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Storage Tank Sizing Based off Retention Time: Assumed Cylindrical", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.next_rect[0])
        next_button = font.render(self.next_text[0], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.next_rect[0].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define the open Tanks Sizing Page 2
class OpenTankSizingPage2(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x / 2 - button_width / 2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]

        self.back_text = [
            "Back"
        ]
        self.input_labels = ["Outside Temp (C):","Outside Sun [lux]","Wind Speed (m/s)", "Liq Temp (C)", "Media CAS:", "Surf. Vel. (m/s):","Insul Conductivity (W/mK):"]
        self.input_values = ["", "", "", "","","",""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Insul. Thickness [mm]:", "Heat Loss (kW):", "Heat Loss (Btu/hr):"]
        self.output_values = ["", "", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height)
        ]
        self.in_back = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
                self.update_output_fields()
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(OpenTankSizingPage1())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        insulation_thickness_mm, heat_loss_kw, heat_loss_btuh = self.calculate_insulation_thickness_and_heat_loss()
        
        if isinstance(insulation_thickness_mm, (int, float)):
            self.output_values[0] = f"{insulation_thickness_mm:.2f}"
        else:
            self.output_values[0] = ""
            
        if isinstance(heat_loss_kw, (int, float)):
            self.output_values[1] = f"{heat_loss_kw:.2f}"
        else:
            self.output_values[1] = ""
            
        if isinstance(heat_loss_btuh, (int, float)):
            self.output_values[2] = f"{heat_loss_btuh:.2f}"
        else:
            self.output_values[2] = ""

    def calculate_insulation_thickness_and_heat_loss(self):
        try:
            T_outside = float(self.input_values[0])
            lux = float(self.input_values[1])
            wind_speed = float(self.input_values[2])
            T_liquid = float(self.input_values[3])
            media_CAS = float(self.input_values[4])
            surf_vel = float(self.input_values[5])
            insul_conductivity = float(self.input_values[6])

            # Add your specific formulas for calculating insulation thickness and heat loss
            insulation_thickness = (T_liquid - T_outside) / (wind_speed * insul_conductivity)
            heat_loss_kw = insul_conductivity * (T_liquid - T_outside) * surf_vel / insulation_thickness
            heat_loss_btuh = heat_loss_kw * 3412.142

            return insulation_thickness * 1000, heat_loss_kw, heat_loss_btuh
        except ValueError:
            return "", "", ""

    def draw_storage_tank(self,screen, x, y, width, height, border_width):
        """
        Draws a schematic of a liquid open storage tank with a hemispherical top and a blue lower half
        and a white upper half with black border around it.
        
        Arguments:
        screen -- the Pygame surface on which to draw the schematic
        x -- the x-coordinate of the top-left corner of the tank
        y -- the y-coordinate of the top-left corner of the tank
        width -- the width of the tank
        height -- the height of the tank
        border_width -- the width of the border around the tank
        """
        
        # Draw the tank body
        tank_color = (255, 255, 255) # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        tank_color = (0, 0, 255) # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0) # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)
        #pygame.draw.circle(screen, tank_border_color, (x + width // 2, y), width // 2 + border_width, border_width)

    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Storage Tank: Assumed Cylindrical", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)

        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define the Ventilated Tanks Page
class VentilatedTanksPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x / 2 - button_width / 2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]
        self.next_rect = [
            pygame.Rect(button_start_x / 2 + button_width / 2 + 900, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)

        ]

        self.back_text = ["Back"]
        self.next_text = ["Next"]
        self.input_labels = ["In Flow (m3/s):", "Out Flow (m3/s):", "Retention Time (s):", "L/D (m/m):"]
        self.input_values = ["", "", "", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Req. Volume (m³):", "Length (m):", "Diameter (m):"]
        self.output_values = ["", "", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height)
        ]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.next_rect[0].collidepoint(event.pos):
                self.in_next = True
                print("Next")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(VesselsPage())
        elif self.in_next:
            self.manager.go_to(OpenTankSizingPage2())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            retention_time = float(self.input_values[2])
            out_flow = float(self.input_values[1])
            ld_ratio = float(self.input_values[3])

            req_volume = retention_time * out_flow
            diameter = (1/3)*(4 * req_volume*9 / (math.pi))**(1/3)
            length = diameter * ld_ratio

            self.output_values = [f"{req_volume:.2f}", f"{length:.2f}", f"{diameter:.2f}"]
        except ValueError:
            self.output_values = ["", "", ""]

    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)

        
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Storage Tank Sizing Based off Retention Time: Assumed Cylindrical", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.next_rect[0])
        next_button = font.render(self.next_text[0], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.next_rect[0].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define Heat Exchanger Page
class HeatExchangerPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
        self.menu_texts = [
        "Shell and Tube",
        "Plate",
        "Spiral",
        "Single Tube",
        "Double Pipe",
        "Back"
    ]
        self.in_shell_tube_heat_exchanger= False
        self.in_plate_heat_exchanger = False
        self.in_spiral_heat_exchanger = False
        self.in_SingleTube_heat_exchanger = False
        self.in_double_pipe_heat_exchanger = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_shell_tube_heat_exchanger = True
                    elif i == 1:
                        self.in_plate_heat_exchanger = True
                    elif i == 2:
                        self.in_spiral_heat_exchanger = True
                    elif i == 3:
                        self.in_SingleTube_heat_exchanger = True
                    elif i == 4:
                        self.in_double_pipe_heat_exchanger = True
                    elif i == 5:
                        self.in_back = True
        if self.in_shell_tube_heat_exchanger:
            self.manager.go_to(ShellTubeHexPage())
        elif self.in_plate_heat_exchanger:
            self.manager.go_to(PlateHexPage())
        elif self.in_spiral_heat_exchanger:
            self.manager.go_to(SpiralHexPage())
        elif self.in_SingleTube_heat_exchanger:
            self.manager.go_to(SingleTubeHexPage())
        elif self.in_double_pipe_heat_exchanger:
            self.manager.go_to(DoublePipeHexPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Heat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Shell and Tube Hex Page
class ShellTubeHexPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.menu_rects = [
            pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
            pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20)
        ]
        self.menu_texts = [
            "Start Shell and Tube Heat Exchanger Designer By Örjan Johansson",
            "Back"
        ]

        self.back_text = ["Back"]
        
    
        self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson = False
        self.in_back = False
        self.manager = page_manager
        self.subprocess_obj = None  # Store the subprocess object here
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson = True
                    elif i == 1:
                        self.in_back = True
        if self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson:
            subprocess.run(["python3", os.path.join(current_directory,"apps/ST_HEX_Orjan_GUI.py")])
            self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson = False
        elif self.in_back:
            self.manager.go_to(HeatExchangerPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Shell and Tube Heat Exchanger Design an Sizing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Plate Heat Exchanger Page
class PlateHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Plate Heat Exchangers", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Spiral Heat Exchanger Page
class SpiralHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Spirial Heat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define SingeTube Heat Exchanger Page
class SingleTubeHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Single Tube HEat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Double Pipe Heat Exchanger Page
class DoublePipeHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Jacked Single pipe Aka Double Pipe Heat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Separations Page
class SeparationsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Gas",
        "Liquid",
        "Solid",
        "Back"
    ]
        self.in_gas_separations= False
        self.in_liquid_separations = False
        self.in_solid_separations = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_gas_separations = True
                    elif i == 1:
                        self.in_liquid_separations = True
                    elif i == 2:
                        self.in_solid_separations = True
                    elif i == 3:
                        self.in_back = True
                        self.in_back = True
        if self.in_gas_separations:
            self.manager.go_to(GasSeparationsPage())
        elif self.in_liquid_separations:
            self.manager.go_to(LiquidSeparationsPage())
        elif self.in_solid_separations:
            self.manager.go_to(SolidSeparationsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Select Separation Type", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Gas Separations Page
class GasSeparationsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Liquid Separations Page
class LiquidSeparationsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Solids Separations Page
class SolidSeparationsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pumps Page
class PumpsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
    ]
        self.menu_texts = [
        "Centrifugal",
        "Positive Displacement",
        "Ejectors",
        "Back"
    ]
        self.in_centrifugal_pump= False
        self.in_positive_displacement_pump = False
        self.in_ejectors = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_centrifugal_pump = True
                    elif i == 1:
                        self.in_positive_displacement_pump = True
                    elif i == 2:
                        self.in_ejectors = True
                    elif i == 3:
                        self.in_back = True
        if self.in_centrifugal_pump:
            self.manager.go_to(CentrifugalPumpsPage())
        elif self.in_positive_displacement_pump:
            self.manager.go_to(PositiveDisplacementPumpsPage())
        elif self.in_ejectors:
            self.manager.go_to(EjectorPumpsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Pumps", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Centrifugal Pumps Page
class CentrifugalPumpsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
        "Multi-Stage",
        "Single Stage",
        "Magnetic Drive",
        "Submersible",
        "Back"
    ]
        self.in_multi_stage_pump= False
        self.in_single_stage_pump = False
        self.in_magnetic_drive_pump = False
        self.in_submersible_pump = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_multi_stage_pump = True
                    elif i == 1:
                        self.in_single_stage_pump = True
                    elif i == 2:
                        self.in_magnetic_drive_pump = True
                    elif i == 3:
                        self.in_submersible_pump = True
                    elif i == 4:
                        self.in_back = True
        if self.in_multi_stage_pump:
            self.manager.go_to(MultiStagePumpsPage())
        elif self.in_single_stage_pump:
            self.manager.go_to(SingleStagePumpsPage())
        elif self.in_magnetic_drive_pump:
            self.manager.go_to(MagneticDrivePumpsPage())
        elif self.in_submersible_pump:
            self.manager.go_to(SubmersiblePumpsPage())
        elif self.in_back:
            self.manager.go_to(PumpsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Centrifugal Pumps", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Multi-Stage Centrifugal Pumps Page
class MultiStagePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the single stage centrifugal pumps page
class SingleStagePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Magnetic Pumps Page
class MagneticDrivePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Submersible Pumps Page
class SubmersiblePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Positive Displacement Pumps Page
class PositiveDisplacementPumpsPage(Page):
    def __init__(self,page_manager = None):
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        
        # Existing Button Texts
        self.menu_texts = ["Piston",
        "Rotary lobe",
        "Progressive Cavity",
        "Gear",
        "Diaphragm",
        "Peristaltic",
        "Screw Pump",
        "Back"]
        
        # Existing Buttons
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed

        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))
        ]
        
        


        self.in_piston_pump= False
        self.in_rotary_lobe_pump = False
        self.in_progressive_cavity_pump = False
        self.in_gear_pump = False
        self.in_diaphragm_pump = False
        self.in_peristaltic_pump = False
        self.in_screw_pump = False

        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.manager.go_to(PistonPumpPage())
                    elif i == 1:
                        self.manager.go_to(RotaryLobePumpPage())
                    elif i == 2:
                        self.manager.go_to(ProgressiveCavityPumpPage())
                    elif i == 3:
                        self.manager.go_to(GearPumpPage())
                    elif i == 4:
                        self.manager.go_to(DiaphragmPumpPage())
                    elif i == 5:
                        self.manager.go_to(PeristalticPumpPage())
                    elif i == 6:
                        self.manager.go_to(ScrewPumpPage())
                    elif i == 7:
                        self.manager.go_to(PumpsPage())  # Assuming 'Back' should go to the previous page
            self.manager.go_to(PumpsPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Positive Displacement Pump Menu", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Piston Pump Page
class PistonPumpPage(Page):
    pass
# Define Rotary Lobe Pump Page
class RotaryLobePumpPage(Page):
    pass
# Define Progressive Cavity Pump Page
class ProgressiveCavityPumpPage(Page):
    pass
# Define Gear Pump Page
class GearPumpPage(Page):
    pass
# Define Diaphragm Pump Page
class DiaphragmPumpPage(Page):
    pass
# Define Peristaltic Pump Page
class PeristalticPumpPage(Page):
    pass
# Define Screw Pump Page
class ScrewPumpPage(Page):
    pass
# Define Ejector Pumps Page
class EjectorPumpsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Reactors Page
class ReactorsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "CSTR",
        "PFR",
        "PBR",
        "Back"
    ]
        self.in_cstr= False
        self.in_pfr = False
        self.in_pbr = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_cstr = True
                    elif i == 1:
                        self.in_pfr = True
                    elif i == 2:
                        self.in_pbr = True
                    elif i == 3:
                        self.in_back = True
        if self.in_cstr:
            self.manager.go_to(HeatExchangerPage())
        elif self.in_pfr:
            self.manager.go_to(SeparationsPage())
        elif self.in_pbr:
            self.manager.go_to(PumpsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Reactor Selection", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Controls Page
class ControlsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
        "Valves",
        "Sensors & Instr.",
        "Transfer Fn",
        "Fluid Handling",
        "Back"
    ]
        self.in_valves= False
        self.in_sensors = False
        self.in_transfer_fn = False
        self.in_fluid_handling = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_valves = True
                    elif i == 1:
                        self.in_sensors = True
                    elif i == 2:
                        self.in_transfer_fn = True
                    elif i == 3:
                        self.in_fluid_handling = True
                    elif i == 4:
                        self.in_back = True
        if self.in_valves:
            self.manager.go_to(ValvesPage())
        elif self.in_sensors:
            self.manager.go_to(SensorsPage())
        elif self.in_transfer_fn:
            self.manager.go_to(TransferFnPage())
        elif self.in_fluid_handling:
            self.manager.go_to(FluidHandlingPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Controls Page", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Valves Page
class ValvesPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Valves!!", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Transfer Functions Page
class TransferFnPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Transfer Functions o_0", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Fluid Handling Page
class FluidHandlingPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    
    ]
        self.menu_texts = [
        "Pipes",
        "Bends",
        "Orifices",
        "Tees",
        "Vena Contracta",
        "Back"

    ]
        self.in_pipes= False
        self.in_bends = False
        self.in_orifices = False
        self.in_tees = False
        self.in_vena_contracta = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_pipes = True
                    elif i == 1:
                        self.in_bends = True
                    elif i == 2:
                        self.in_orifices = True
                    elif i == 3:
                        self.in_tees = True
                    elif i == 4:
                        self.in_vena_contracta = True
                    elif i == 5:
                        self.in_back = True
        if self.in_pipes:
            print("Pipes Page")
            self.manager.go_to(PipesPage())
        elif self.in_bends:
            print("Bends Page")
            self.manager.go_to(BendsPage())
        elif self.in_orifices:
            print("Orifice Page")
            self.manager.go_to(OrificePage())
        elif self.in_tees:
            print("Tees Page")
            self.manager.go_to(TeesPage())
        elif self.in_vena_contracta:
            print("Vena Contracta Page")
            self.manager.go_to(VenaContractaPage())
        elif self.in_back:
            print("Back")
            self.manager.go_to(ControlsPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Fluid Handling", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipes Page
class PipesPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
            "Diameter Sizing",
            "Pipe Pressure Drop",
            "Pipe Wall Thickness",
            "Pipe Heat Transfer",
            "Back"
        ]
        self.in_diameter_sizing = False
        self.in_pipe_pressure_drop = False
        self.in_pipe_wall_thickness = False
        self.in_pipe_heat_transfer = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_diameter_sizing = True
                    elif i == 1:
                        self.in_pipe_pressure_drop = True
                    elif i == 2:
                        self.in_pipe_wall_thickness = True
                    elif i == 3:
                        self.in_pipe_heat_transfer = True
                    elif i == 4:
                        self.in_back = True

        if self.in_diameter_sizing:
            print("Diameter Sizing Page")
            self.manager.go_to(PipeDiameterSizingPage())
        elif self.in_pipe_pressure_drop:
            print("Pipe Pressure Drop Page")
            self.manager.go_to(PipePressureDropPage())
        elif self.in_pipe_wall_thickness:
            print("Pipe Wall Thickness Page")
            self.manager.go_to(PipeWallThicknessPage())
        elif self.in_pipe_heat_transfer:
            print("Pipe Heat Transfer Page")
            self.manager.go_to(PipeHeatTransferPage())
        elif self.in_back:
            print("Back")
            self.manager.go_to(FluidHandlingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipes", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Pressure Drop Page
class PipePressureDropPage(Page):
    def __init__(self,page_manager=None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipesPage())
        elif self.in_next:
            print("Next")
            self.manager.go_to(PipePressureDropPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipe Pressure Drop", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Wall Thickness Page
class PipeWallThicknessPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipe Wall Thickness?", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Heat Transfer Page
class PipeHeatTransferPage(Page):
    def __init__(self,page_manager =None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipesPage())
        elif self.in_next:
            print("Next")
            self.manager.go_to(PipeHeatTransferPage2())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Heat transfer in Pipe", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Heat Transfer Page 2
class PipeHeatTransferPage2(Page):
    def __init__(self,pygame_screen=None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipeHeatTransferPage())
        elif self.in_next:
            print("Next")
            self.manager.go_to(PipeHeatTransferPage3())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipe Heat Transfer", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Diameter Sizing Page
class PipeDiameterSizingPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.input_labels = ["Flow (l/s):","Econ. Velocity (m/s):"]
        self.input_values = ["", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height)]
        self.output_labels = ["Econ. Diameter (mm):","Econ. Diameter (in):"]
        self.output_values = ["", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height)]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.menu_rects[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.menu_rects[1].collidepoint(event.pos):
                self.in_next = True
                print("Next")
        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode     
        if self.in_back:
            self.manager.go_to(PipesPage())
        elif self.in_next:
            self.manager.go_to(PipeDiameterSizingPage2())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            flow = float(self.input_values[0])
            flow = flow/1000
            velocity = float(self.input_values[1])
            diameter_mm = ((flow*4/(velocity*math.pi))**(1/2))*1000
            diameter_in = diameter_mm/25.4

            self.output_values = [f"{diameter_mm:.2f}",f"{diameter_in:.2f}"]
        except ValueError:
            self.output_values = ["",""]
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)     
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)
    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Diameter Sizing: Knowns: Flow, Economic Velocity", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
        

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.menu_rects[0])
        back_button = font.render(self.menu_texts[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.menu_rects[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.menu_rects[1])
        next_button = font.render(self.menu_texts[1], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.menu_rects[1].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(FluidHandlingPage())

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()     
# Define Pipe Diameter Sizing Page 2
class PipeDiameterSizingPage2(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height)]
        self.menu_texts = [
            "Back"]
        self.input_labels = ["Flow (l/s):","Diameter (mm):"]
        self.input_values = ["", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height)]
        self.output_labels = ["Velocity (m/s):"]
        self.output_values = [""]
        self.output_rects = [pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height)]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.menu_rects[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(PipeDiameterSizingPage())
        elif self.in_next:
            self.manager.go_to(PipesPage())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            flow = float(self.input_values[0])
            flow = flow/1000
            diameter_mm = float(self.input_values[1])
            velocity = flow*4/(math.pi*(diameter_mm/1000)**2)

            self.output_values = [f"{velocity:.2f}"]
        except ValueError:
            self.output_values = [""]
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)     
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)
    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Diameter Sizing: Knowns: Flow, Pipe Diameter", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
        

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.menu_rects[0])
        back_button = font.render(self.menu_texts[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.menu_rects[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(PipeDiameterSizingPage())

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define Pipe Velocity Page
class PipeVelocityPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipeDiameterSizingPage())
        elif self.in_next:
            print("Calculate")
            self.manager.go_to(PipeVelocityPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Average Pipe Velocity", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Bends Page
class BendsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Orifices Page
class OrificePage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Tees Page
class TeesPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Vena Contracta Page
class VenaContractaPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Sensors Page
class SensorsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
        self.menu_texts = [
        "Temperature",
        "Pressure",
        "Flow",
        "Level",
        "Composition",
        "Back"
    ]
        self.in_temp_sensor= False
        self.in_pressure_sensor = False
        self.in_flow_sensor = False
        self.in_level_sensor = False
        self.in_composition_sensor = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_temp_sensor = True
                    elif i == 1:
                        self.in_pressure_sensor = True
                    elif i == 2:
                        self.in_flow_sensor = True
                    elif i == 3:
                        self.in_level_sensor = True
                    elif i == 4:
                        self.in_composition_sensor = True
                    elif i == 5:
                        self.in_back = True
        if self.in_temp_sensor:
            self.manager.go_to(TemperatureSensorsPage())
        elif self.in_pressure_sensor:
            self.manager.go_to(PressureSensorsPage())
        elif self.in_flow_sensor:
            self.manager.go_to(FlowSensorsPage())
        elif self.in_level_sensor:
            self.manager.go_to(LevelSensorsPage())
        elif self.in_composition_sensor:
            self.manager.go_to(CompositionSensorsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Main Types of Sensors", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Temperature Sensors Page
class TemperatureSensorsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pressure Sensors Page
class PressureSensorsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Flow Sensors Page
class FlowSensorsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Level Sensors Page
class LevelSensorsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Chemical Composition Sensors Page
class CompositionSensorsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "pH",
        "Conductivity",
        "FT-IR",
        "Back"
    ]
        self.in_ph_sensor= False
        self.in_conductivity_sensor = False
        self.in_ft_ir_sensor = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_ph_sensor = True
                    elif i == 1:
                        self.in_conductivity_sensor = True
                    elif i == 2:
                        self.in_ft_ir_sensor = True
                    elif i == 3:
                        self.in_back = True
        if self.in_ph_sensor:
            self.manager.go_to(pHSensorsPage())
        elif self.in_conductivity_sensor:
            self.manager.go_to(ConductivitySensorsPage())
        elif self.in_ft_ir_sensor:
            self.manager.go_to(FtIrPage())
        elif self.in_back:
            self.manager.go_to(SensorsPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Composition Analyzers", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define the pH Sensors Page
class pHSensorsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Conductivity Sensors Page
class ConductivitySensorsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define FT-IR Sensors Page
class FtIrPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Gas Alarms Page
class GasAlarmsPage(Page):
    def __init__(self) -> None:
        super().__init__()
        self.menu_rects = []
        self.menu_texts = []
        self.in_back = False
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
        if self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Gas Alarms", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Flare System Page
class FlareSystemPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Capital Cost Page
class ProcessEconomicsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
        pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Estimate Flowsheet Capital Cost",
        "Edit Capital Cost Estimation Factors",
        "Optimal Pipe Diameter",
        "Back"
    ]
        self.in_estimate_flowsheet_capital_cost= False
        self.in_edit_capital_cost_estimation_factors = False
        self.in_optimal_pipe_diameter = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_estimate_flowsheet_capital_cost = True
                    elif i == 1:
                        self.in_edit_capital_cost_estimation_factors = True
                    elif i == 2:
                        self.in_optimal_pipe_diameter = True
                    elif i == 3:
                        self.in_back = True
        if self.in_estimate_flowsheet_capital_cost:
            self.manager.go_to(EstimateFlowsheetCapitalCostEstimationPage())
        elif self.in_edit_capital_cost_estimation_factors:
            self.manager.go_to(EditCapitalCostEstimationParametersPage())
        elif self.in_optimal_pipe_diameter:
            self.manager.go_to(OptimalPipeDiameterPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Process Economics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Estimate 
class OptimalPipeDiameterPage(Page):
    def __init__(self) -> None:
        self.menu_rects = [
        pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
        pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20)
    ]
        self.menu_texts = [
        "Start Optimal Pipe Diameter Calculation Program for Sweden By Oscar Rexid",
        "Back"
    ]
        self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit = False
        self.in_back = False
        self.manager = page_manager
        self.subprocess_obj = None  # Store the subprocess object here
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit = True
                    elif i == 1:
                        self.in_back = True
        if self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Pipe_Cost_Optimization.py")])
            self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit = False
        elif self.in_back:
            self.manager.go_to(ProcessEconomicsPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Optimal Pipe Diameter", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Estimate Flowsheet Capital Cost Estimation Page
class EstimateFlowsheetCapitalCostEstimationPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Edit Capital Cost Estimation Parameters Page
class EditCapitalCostEstimationParametersPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Process Safety Page
class ProcessSafetyPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x-(button_width)//8, button_start_y, button_width*1.25, button_height),
        pygame.Rect(button_start_x-(button_width)//4, button_start_y + button_height + button_padding, button_width*1.5, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = process_safety_submenu_texts = [
        "Estimate Flowsheet Safety",
        "Find Chemical Safety Properties",
        "Safety Instr.",
        "Back"
    ]
        self.in_estimate_flowsheet_safety= False
        self.in_find_chemical_safety = False
        self.in_safety_instrumentation = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_estimate_flowsheet_safety = True
                    elif i == 1:
                        self.in_find_chemical_safety = True
                    elif i == 2:
                        self.in_safety_instrumentation = True
                    elif i == 3:
                        self.in_back = True
        if self.in_estimate_flowsheet_safety:
            self.manager.go_to(EstimateFlowsheetSafetyPage())
        elif self.in_find_chemical_safety:
            self.manager.go_to(FindChemicalSafetyPage())
        elif self.in_safety_instrumentation:
            self.manager.go_to(SafetyInstrumentationPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Process Safety", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Flowsheet Safety Estimation Page
class EstimateFlowsheetSafetyPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Chemical Safety Page
class FindChemicalSafetyPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.menu_rects = [
            pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
            pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20)
        ]
        self.menu_texts = [
            "Start Industrial Hygeine Finder",
            "Back"
        ]

        self.back_text = ["Back"]
        
    
        self.in_start_industrial_hygeine_finder = False
        self.in_back = False
        self.manager = page_manager
        self.subprocess_obj = None  # Store the subprocess object here
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_start_industrial_hygeine_finder = True
                    elif i == 1:
                        self.in_back = True
        if self.in_start_industrial_hygeine_finder:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Safety_Chemical_Safety.py")])
            self.in_start_industrial_hygeine_finder = False
        elif self.in_back:
            self.manager.go_to(ProcessSafetyPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("TWA STEL Data", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Safety Instrument Page
class SafetyInstrumentationPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Rupture Disks",
        "relief Valves",
        "Gas Alarms",
        "Back"
    ]
        self.in_rupture_disk= False
        self.in_relief_valves = False
        self.in_gas_alarms = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_rupture_disk = True
                    elif i == 1:
                        self.in_relief_valves = True
                    elif i == 2:
                        self.in_gas_alarms = True
                    elif i == 3:
                        self.in_back = True
        if self.in_rupture_disk:
            self.manager.go_to(RuptureDiskPage())
        elif self.in_relief_valves:
            self.manager.go_to(ReliefValvesPage())
        elif self.in_gas_alarms:
            self.manager.go_to(GasAlarmsPage())
        elif self.in_back:
            self.manager.go_to(ProcessSafetyPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Equipment Sizing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Rupture Disk Page
class RuptureDiskPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define relief Valves Page
class ReliefValvesPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Physical Properties Page
class PhysicalPropertiesPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
        self.menu_texts = [
        "Add a Chemical",
        "Find a Chemical",
        "Back"
    ]
        self.in_add_chemical= False
        self.in_find_chemical = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_add_chemical = True
                    elif i == 1:
                        self.in_find_chemical = True
                    elif i == 2:
                        self.in_back = True
                        self.in_back = True
        if self.in_add_chemical:
            screen_dimensions = (WINDOW_SIZE[0], WINDOW_SIZE[1])
            self.manager.go_to(AddChemicalPage())
        elif self.in_find_chemical:
            self.manager.go_to(FindChemicalPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())

    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Physical Properties of Chemicals", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define a basic button class
class Button:
    def __init__(self, x, y, width, height, text, font_size, bg_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont(None, font_size)
        self.bg_color = bg_color
        self.text_color = BLACK

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg_color, self.rect)
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)
# Define InputFieldWithUnits Class
class InputFieldWithUnits:
    def __init__(self, font, font_name, label='', max_length=None, var_name='', units=''):
        self.label = label
        self.var_name = var_name
        self.units = units
        self.max_length = max_length
        self.font = font
        self.bold_font = pygame.font.SysFont(font_name, font.get_height(), bold=True)  # Create a bold font using font_name
        self.rect = pygame.Rect(0, 0, 0, 60)
        self.active = False
        self.update_surface()

    def render(self, surface, pos):
        self.update_surface()
        self.rect = self.surface.get_rect(center=pos)
        surface.blit(self.surface, self.rect)


    def get_text(self):
        return self.label

    def add_character(self, char):
        if self.max_length is not None and len(self.label) >= self.max_length:
            return
        self.label += char
        self.update_surface()

    def remove_character(self):
        self.label = self.label[:-1]
        self.update_surface()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = True
            self.label = ""
            self.update_surface()
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.remove_character()
            elif event.unicode.isprintable():
                self.add_character(event.unicode)

    def update_surface(self):
        fixed_surface_width = 300

        # Use the bold_font for rendering the variable name
        var_name_surf = self.bold_font.render(self.var_name, True, BLACK)
        self.surface = pygame.Surface((fixed_surface_width, 60), pygame.SRCALPHA)

        fixed_surface_width = 500
        box_start_x = 150
        var_name_surf = self.font.render(self.var_name, True, BLACK)

        units_numerator, units_denominator = self.units.split('/')
        units_numerator_surf = self.font.render(units_numerator, True, BLACK)
        units_denominator_surf = self.font.render(units_denominator, True, BLACK)

        self.surface = pygame.Surface((fixed_surface_width, 60), pygame.SRCALPHA)

        # Clear the surface with a transparent color
        self.surface.fill((0, 0, 0, 0))

        pygame.draw.rect(self.surface, BLACK, (box_start_x, 20, int(125 * 1.50), 40), 2)
        label_surf = self.font.render(self.label, True, BLACK)
        self.surface.blit(label_surf, (box_start_x + 5, 25))

        self.surface.blit(var_name_surf, (fixed_surface_width // 2 - var_name_surf.get_width() // 2, -5))

        units_x_offset = -80  # Change this value to move the units label closer or farther from the input box
        max_unit_width = max(units_numerator_surf.get_width(), units_denominator_surf.get_width())
        units_x = fixed_surface_width - max_unit_width - 10 + units_x_offset

        current_unit_width = max(units_numerator_surf.get_width(), units_denominator_surf.get_width())
        additional_offset = max_unit_width - current_unit_width

        self.surface.blit(units_numerator_surf, (units_x + additional_offset // 2 + (max_unit_width - units_numerator_surf.get_width()) // 2, 5))
        pygame.draw.line(self.surface, BLACK, (units_x - 5 + additional_offset // 2, 30), (units_x + max_unit_width + 5 + additional_offset // 2, 30), 1)
        self.surface.blit(units_denominator_surf, (units_x + additional_offset // 2, 35))
# Define the InputField class
class InputField:
    def __init__(self, font, label='', max_length=None):
        self.label = label
        self.max_length = max_length
        self.font = font
        self.surface = pygame.Surface((250, 40))
        self.rect = self.surface.get_rect()
        self.active = False

    def get_text(self):
        return self.label

    def add_character(self, char):
        if self.max_length is not None and len(self.label) >= self.max_length:
            return
        self.label += char
        self.update_surface()

    def remove_character(self):
        self.label = self.label[:-1]
        self.update_surface()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = True
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.remove_character()
            elif event.unicode.isprintable():
                self.add_character(event.unicode)

    def update_surface(self):
        self.surface.fill(WHITE)
        pygame.draw.rect(self.surface, GRAY, (0, 0, 250, 40), 2)
        label_surf = self.font.render(self.label, True, BLACK)
        self.surface.blit(label_surf, (5, 5))

    def render(self, surface, pos, height):
        self.rect.center = pos
        self.update_surface()
        surface.blit(self.surface, self.rect)

    def render_value(self, surface, rect):
        pygame.draw.rect(surface, WHITE, rect)
        pygame.draw.rect(surface, GRAY, rect, 2)

        # Split the label into label and unit
        label, unit = self.label.rsplit(' ', 1)

        # Draw the input field label on the right side of the input box
        label_surface = font.render(label, True, BLACK)
        label_rect = label_surface.get_rect()
        label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
        label_rect.centery = rect.centery
        surface.blit(label_surface, label_rect)

        # Draw the input field text
        text_surface = font.render(self.label, True, BLACK)
        text_rect = text_surface.get_rect()
        text_rect.centerx = rect.centerx
        text_rect.centery = rect.centery
        surface.blit(text_surface, text_rect)

        # Draw the input field unit on the left side of the input box
        unit_surface = font.render(unit, True, BLACK)
        unit_rect = unit_surface.get_rect()
        unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
        unit_rect.centery = rect.centery
        surface.blit(unit_surface, unit_rect)
# Define the AddChemicalPage class
class AddChemicalPage(Page):
    def __init__(self, page_manager = None):
        super().__init__()
        self.cas_entry = InputField(font,label ="CAS Number",max_length = 50)
        self.chemical_name_entry = InputField(font, label="Chemical Name", max_length = 20)
        self.molecular_formula_entry = InputField(font, label ="Molecular Formula", max_length=20)
        pos = (button_start_x, button_start_y)
        self.submit_button = Button(pos[0], pos[1], button_width+100, button_height+50, 'Submit', font_size=30, bg_color=BLUE)
        self.back_button = Button(pos[0], pos[1] + button_height + button_padding, button_width, button_height, "Back", font_size=30, bg_color=GRAY)

    def handle_event(self, event):
        # Handle events for input fields and submit button
        self.cas_entry.handle_event(event)
        self.chemical_name_entry.handle_event(event)
        self.molecular_formula_entry.handle_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if self.submit_button.is_clicked(mouse_pos):
                # Do something when the button is clicked
                print("Entry Submitted")
            elif self.back_button.is_clicked(mouse_pos):
                self.manager.go_to(MainMenuPage())


    # Render method
    def render(self, pygame_screen):
        screen.fill(WHITE)
        text = font.render("Add Chemical Data", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)

        input_y_offset = button_start_y + button_height + button_padding
        input_fields = [(self.cas_entry, (button_start_x * 1.5, input_y_offset)),
                        (self.chemical_name_entry, (button_start_x * 1.5, input_y_offset + 75)),
                        (self.molecular_formula_entry, (button_start_x * 1.5, input_y_offset + 150))]
        for field, pos in input_fields:
            field.render(pygame_screen, pos, 100)

        pygame.draw.rect(pygame_screen, GRAY, self.submit_button.rect)
        pygame.draw.rect(pygame_screen,GRAY, self.back_button.rect)

        pygame.display.update()
# Define Find Chemical Page
class FindChemicalPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Data Pre-Processing Page
class DataPreProcessingPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Quality check and Basic Cleaning Page
class DataQualityCheckAndBasicCleaningPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the temporal consistency Page
class DataTemporalConsistencyPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Temporal Data", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Normalization and Standardization Page
class DataNormalizationAndStandardizationPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Dimensionality Reduction Page
class DataDimensionalityReductionPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Noise Reduction and Outlier Handling Page
class DataNoiseReductionAndOutlierHandlingPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Flowsheet Renderer Class
class FlowsheetRenderer:
    def __init__(self,flowsheet):
        self.flowsheet = flowsheet
    # Draw a line
    def draw_line(screen, start_pos, end_pos, color):
        pygame.draw.line(screen, color, start_pos, end_pos, 5)
    # Draw a rectangle
    def draw_rect(screen, rect, color):
        pygame.draw.rect(screen, color, rect)
    # Draw a filled rectangle
    def draw_filled_rect(screen, rect, color):
        pygame.draw.rect(screen, color, rect, 0)
    # Draw a circle
    def draw_circle(screen, center_pos, radius, color):
        pygame.draw.circle(screen, color, center_pos, radius, 5)
    # Draw a filled circle
    def draw_filled_circle(screen, center_pos, radius, color):
        pygame.draw.circle(screen, color, center_pos, radius, 0)
    # Draw an ellipse
    def draw_ellipse(screen, rect, color):
        pygame.draw.ellipse(screen, color, rect, 5)
    # Draw a filled ellipse
    def draw_filled_ellipse(screen, rect, color):
        pygame.draw.ellipse(screen, color, rect, 0)
    # Draw a polygon
    def draw_polygon(screen, point_list, color):
        pygame.draw.polygon(screen, color, point_list, 5)
    # Draw a filled polygon
    def draw_filled_polygon(screen, point_list, color):
        pygame.draw.polygon(screen, color, point_list, 0)
    # Draw a valve
    def draw_valve(screen, center_pos):
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 20, center_pos[1] - 5, 40, 10))
    # Draw a capsule with points
    def capsule_points(a, r, n, tolerance):
        top_points = []
        bottom_points = []

        for y in np.linspace(0, r, num=n // 2):
            x_minus_term = np.sqrt(r ** 2 - y ** 2)
            x_plus_term = x_minus_term

            for x in np.linspace(-a - x_minus_term, a + x_plus_term, num=n // 2):
                equation = (x - a - x_minus_term) * (x + a + x_plus_term) * y * (y - r)

                if abs(equation) <= tolerance:
                    top_points.append((x, y))
                    bottom_points.append((x, -y))

        return top_points, bottom_points
    # Draw Capsule
    def draw_capsule(screen, center_pos, width, height, color, thickness, rotation=0):
        a = width / 2
        r = height / 2
        n = 200
        tolerance = 10 ** (-n)

        top_points, bottom_points = capsule_points(a, r, n, tolerance)

        if rotation == 90:
            translated_top_points = [(int(center_pos[0] - y), int(center_pos[1] - x)) for x, y in top_points]
            translated_bottom_points = [(int(center_pos[0] - y), int(center_pos[1] - x)) for x, y in bottom_points]
        else:
            translated_top_points = [(int(center_pos[0] + x), int(center_pos[1] - y)) for x, y in top_points]
            translated_bottom_points = [(int(center_pos[0] + x), int(center_pos[1] - y)) for x, y in bottom_points]

        pygame.draw.lines(screen, color, False, translated_top_points, thickness)
        pygame.draw.lines(screen, color, False, translated_bottom_points, thickness)
    # Draw a horizontal flash tank
    def draw_horizontal_flash_tank(screen, center_pos, label):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the capsule shape based on the text size
        padding = 10
        capsule_width = text_width + 2 * padding
        capsule_height = text_height + 2 * padding

        # Draw capsule shape
        draw_capsule(screen, center_pos, capsule_width, capsule_height, BLACK, 5)
        draw_capsule(screen, center_pos, capsule_width-3, capsule_height-3, WHITE, 5)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)
    # Draw a vertical flash tank
    def draw_vertical_flash_tank(screen, center_pos, label):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the capsule shape based on the text size
        padding = 10
        capsule_width = text_height + 2 * padding
        capsule_height = text_width + 2 * padding

        # Draw capsule shape
        draw_capsule(screen, center_pos, capsule_width, capsule_height, BLACK, 5, rotation=90)
        draw_capsule(screen, center_pos, capsule_width - 3, capsule_height - 3, WHITE, 5, rotation=90)

        # Draw label
        # Rotate the text 90 degrees counterclockwise
        #rotated_text = pygame.transform.rotate(text, 90)
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)
    # Draw arc
    def draw_arc(screen, color, rect, start_angle, stop_angle, width):
        n_points = 100
        pointlist = []

        for i in range(n_points + 1):
            angle = math.radians(start_angle + (stop_angle - start_angle) * i / n_points)
            x = rect[0] + rect[2] // 2 + int(rect[2] // 2 * math.cos(angle))
            y = rect[1] + rect[3] // 2 - int(rect[3] // 2 * math.sin(angle))
            pointlist.append((x, y))

        pygame.draw.lines(screen, color, False, pointlist, width)
    # Draw online instrumentation
    def draw_online_instrumentation(self,screen, center_pos, label):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the pill shape based on the text size
        padding = 10
        pill_width = text_width + 2 * padding
        pill_height = text_height + 2 * padding

        # Draw pill shape
        rect = (center_pos[0] - pill_width // 2, center_pos[1] - pill_height // 2, pill_width, pill_height)
        pygame.draw.rect(screen, BLACK, rect, 2)
        draw_arc(screen, BLACK, rect, 0, 180, 2)
        draw_arc(screen, BLACK, rect, 180, 360, 2)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)
    # Draw a distillation column
    def draw_distillation_column(self,screen, center_pos, label, column_width, column_height, tray_thickness=2):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the column based on the text size
        padding = 1

        # Calculate the number of trays and tray width based on column dimensions
        num_trays = int((column_height - 2 * padding) / (text_width + 2 * padding))
        tray_width = int(column_width / 2)

        # Draw column
        column_rect = pygame.Rect(center_pos[0] - column_width / 2, center_pos[1] - column_height / 2,
                                column_width, column_height)
        pygame.draw.rect(screen, BLACK, column_rect, 5)
        pygame.draw.rect(screen, WHITE, column_rect.inflate(-3, -3))

        # Draw trays
        tray_spacing = (column_height - 2 * padding) / (num_trays + 1)
        for i in range(num_trays):
            y = center_pos[1] - column_height / 2 + padding + (i + 1) * tray_spacing
            if i % 2 == 0:  # left side
                pygame.draw.line(screen, BLACK, (center_pos[0] - column_width / 2, y),
                                (center_pos[0] - column_width / 2 + tray_width, y), tray_thickness)
            else:  # right side
                pygame.draw.line(screen, BLACK, (center_pos[0] + column_width / 2, y),
                                (center_pos[0] + column_width / 2 - tray_width, y), tray_thickness)

        # Draw label
        rotated_text = pygame.transform.rotate(text, 90)
        rotated_text_rect = text.get_rect(center=center_pos)
        screen.blit(text, rotated_text_rect)
    # Draw a shell and tube heat exchanger
    def draw_shell_and_tube_heat_exchanger(self,screen, center_pos):
        # Draw shell
        pygame.draw.ellipse(screen, BLACK, (center_pos[0] - 50, center_pos[1] - 20, 100, 40), 2)

        # Draw tube bundle
        for i in range(5):
            pygame.draw.line(screen, BLACK, (center_pos[0] - 40 + i * 20, center_pos[1] - 15), (center_pos[0] - 40 + i * 20, center_pos[1] + 15), 2)

        # Draw channel heads
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 60, center_pos[1] - 20, 10, 40), 2)
        pygame.draw.rect(screen, BLACK, (center_pos[0] + 50, center_pos[1] - 20, 10, 40), 2)
    # Draw a heat exchanger
    def draw_heat_exchanger(self,screen, center_pos, radius=20):
        # Draw circle
        pygame.draw.circle(screen, BLACK, center_pos, radius, 2)

        # Calculate the scale factor
        scale_factor = radius / 20

        # Draw zigzag line
        zigzag_points = [
            (center_pos[0] - radius, center_pos[1]),
            (center_pos[0] - radius + 5 * scale_factor, center_pos[1]),
            (center_pos[0] - 10 * scale_factor, center_pos[1] - 10 * scale_factor),
            (center_pos[0], center_pos[1] + 10 * scale_factor),
            (center_pos[0] + 10 * scale_factor, center_pos[1] - 10 * scale_factor),
            (center_pos[0] + radius - 5 * scale_factor, center_pos[1]),
            (center_pos[0] + radius, center_pos[1])
        ]
        pygame.draw.lines(screen, BLACK, False, zigzag_points, 2)
    # Draw a filter press
    def draw_filter_press(self,screen, center_pos, label, column_width, column_height, tray_thickness=2):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Draw column
        column_rect = pygame.Rect(center_pos[0] - column_width / 2, center_pos[1] - column_height / 2,
                                column_width, column_height)
        pygame.draw.rect(screen, BLACK, column_rect, 5)
        pygame.draw.rect(screen, WHITE, column_rect.inflate(-3, -3))

        # Calculate the number of trays based on column_height and text_height
        padding = 1
        num_trays = (column_height - text_width - 4 * padding) // (2 * padding)

        # Draw trays
        tray_spacing = (column_height - 2 * padding) / (num_trays + 1)
        for i in range(num_trays):
            y = center_pos[1] - column_height / 2 + padding + (i + 1) * tray_spacing
            pygame.draw.line(screen, BLACK, (center_pos[0] - column_width / 2, y),
                            (center_pos[0] + column_width / 2, y), tray_thickness)

        # Draw label
        rotated_text = pygame.transform.rotate(text, 90)
        rotated_text_rect = rotated_text.get_rect(center=center_pos)
        screen.blit(rotated_text, rotated_text_rect)
    # Draw a dome roof tank
    def draw_dome_roof_tank(self,screen, center_pos, width, height, color=BLACK):
        # Draw the tank body (rectangle)
        tank_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1], width, height)
        pygame.draw.rect(screen, color, tank_rect, 2)

        # Draw the dome (half-circle)
        dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, width, height)
        pygame.draw.arc(screen, color, dome_rect, 0, math.pi, 2)
    # Draw a centrifugal pump
    def draw_centrifugal_pump(self,screen, center_pos, radius=20):
        pygame.draw.circle(screen, BLACK, center_pos, radius, 2)

        start_angle1 = math.pi / 2
        start_angle2 = 3 * math.pi / 2
        end_angle = 2 * math.pi

        start_point1 = (int(center_pos[0] + radius * math.cos(start_angle1)),
                        int(center_pos[1] - radius * math.sin(start_angle1)))
        start_point2 = (int(center_pos[0] + radius * math.cos(start_angle2)),
                        int(center_pos[1] - radius * math.sin(start_angle2)))
        end_point = (int(center_pos[0] + radius * math.cos(end_angle)),
                    int(center_pos[1] - radius * math.sin(end_angle)))

        pygame.draw.line(screen, BLACK, start_point1, end_point, 2)
        pygame.draw.line(screen, BLACK, start_point2, end_point, 2)
    # Draw a tank
    def draw_tank(screen, center_pos):
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 25, center_pos[1] - 50, 50, 100), 2)
    # Draw a storage tank
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)

    # Draw a right swing check valve
    def draw_left_swing_check_valve(screen, center_pos, triangle_base, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, fill_color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)
    # Draw a right swing check valve
    def draw_right_swing_check_valve(screen, center_pos, triangle_base, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, fill_color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)

    def draw_relief_valve(screen, center_pos):
        pygame.draw.polygon(screen, BLACK, [(center_pos[0] - 20, center_pos[1] + 20), (center_pos[0], center_pos[1] - 20), (center_pos[0] + 20, center_pos[1] + 20)], 2)
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 5, center_pos[1] + 20, 10, 20), 2)

    def draw_flow_control_valve(screen, center_pos):
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 25, center_pos[1] - 5, 50, 10), 2)
        pygame.draw.polygon(screen, BLACK, [(center_pos[0], center_pos[1] - 10), (center_pos[0] - 10, center_pos[1] - 20), (center_pos[0] + 10, center_pos[1] - 20)], 2)

    def draw_pipe_with_arrow(screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        point_a = (end_pos[0] - arrow_length * math.cos(angle + arrow_angle),
                end_pos[1] - arrow_length * math.sin(angle + arrow_angle))
        point_b = (end_pos[0] - arrow_length * math.cos(angle - arrow_angle),
                end_pos[1] - arrow_length * math.sin(angle - arrow_angle))

        # Draw arrowhead
        pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))
        
        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    def draw_pipe_with_double_arrow(screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    def draw_dashed_line(screen, start_pos, end_pos, color, dash_length=5, gap_length=5):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        dashes = int(distance / (dash_length + gap_length))
        dash_step = dx / (dashes * 2)
        gap_step = dy / (dashes * 2)

        for i in range(dashes):
            x_dash_start = x1 + (i * 2 * (dash_step + gap_step))
            y_dash_start = y1 + (i * 2 * (dash_step + gap_step))
            x_dash_end = x_dash_start + dash_step
            y_dash_end = y_dash_start + gap_step
            pygame.draw.line(screen, color, (x_dash_start, y_dash_start), (x_dash_end, y_dash_end), 2)

    def draw_digital_signal(screen, start_pos, end_pos, dash_length=10, color=BLACK):
        length = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        num_dashes = int(length / (dash_length * 2))

        for i in range(num_dashes):
            start = (
                start_pos[0] + (i * 2 * dash_length / length) * (end_pos[0] - start_pos[0]),
                start_pos[1] + (i * 2 * dash_length / length) * (end_pos[1] - start_pos[1])
            )
            end = (
                start_pos[0] + ((2 * i + 1) * dash_length / length) * (end_pos[0] - start_pos[0]),
                start_pos[1] + ((2 * i + 1) * dash_length / length) * (end_pos[1] - start_pos[1])
            )
            pygame.draw.line(screen, color, start, end, 2)

    def draw_pneumatic_line(screen, start_pos, end_pos, dash_length=20, color=BLACK):
        length = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        num_dashes = int(length / (dash_length * 2))
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        diagonal_angle = angle + math.pi / 4
        
        # Draw the main solid line
        pygame.draw.line(screen, color, start_pos, end_pos, 2)

        for i in range(num_dashes):
            dash_start_fraction = (i * 2 * dash_length / length)
            dash_end_fraction = ((2 * i + 1) * dash_length / length)

            center = (
                start_pos[0] + dash_start_fraction * (end_pos[0] - start_pos[0]),
                start_pos[1] + dash_start_fraction * (end_pos[1] - start_pos[1])
            )
            start = (
                center[0] - (dash_length / 4) * math.cos(diagonal_angle),
                center[1] - (dash_length / 4) * math.sin(diagonal_angle)
            )
            end = (
                center[0] + (dash_length / 4) * math.cos(diagonal_angle),
                center[1] + (dash_length / 4) * math.sin(diagonal_angle)
            )
            pygame.draw.line(screen, color, start, end, 2)
    
    def draw_instrumentation(screen, center_pos, symbol_type):
        pygame.draw.circle(screen, BLACK, center_pos, 20, 2)

        font = pygame.font.Font(None, 24)
        text = font.render(symbol_type, True, BLACK)
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)

    def draw_vertical_drum(screen, center_pos, width, height, label, arc_height_ratio=2, color=BLACK):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)

        # Calculate the height of the enclosing rectangle for the arcs
        arc_height = height / arc_height_ratio

        # Draw the top dome (arc)
        top_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2 - arc_height/2, width, arc_height)
        pygame.draw.arc(screen, color, top_dome_rect, 0, math.pi, 2)

        # Draw the bottom dome (arc)
        bottom_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] + height / 2 - arc_height/2, width, arc_height)
        pygame.draw.arc(screen, color, bottom_dome_rect, math.pi, 2 * math.pi, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2, center_pos[1] + height / 2)
        top_right_tangent = (center_pos[0] + width / 2, center_pos[1] - height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, bottom_left_tangent, 2)
        pygame.draw.line(screen, color, top_right_tangent, bottom_right_tangent, 2)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)

    def draw_venturi(screen, center_pos, width, height, arc_height_ratio=1.75, color=BLACK):
        # Calculate the width of the enclosing rectangle for the arcs
        arc_width = width / arc_height_ratio

        # Draw the left dome (arc)
        left_dome_rect = pygame.Rect(center_pos[0] - width / 2 - arc_width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, left_dome_rect, -math.pi / 2, math.pi / 2, 2)

        # Draw the right dome (arc)
        right_dome_rect = pygame.Rect(center_pos[0] + width / 2 - arc_width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, right_dome_rect, math.pi / 2, 3 * math.pi / 2, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2, center_pos[1] - height / 2)
        top_right_tangent = (center_pos[0] + width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2, center_pos[1] + height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, top_right_tangent, 2)
        pygame.draw.line(screen, color, bottom_left_tangent, bottom_right_tangent, 2)

    def draw_sine_wave(screen, center_pos, width, height, arc_height_ratio=2, color=BLACK):
        # Calculate the width of the enclosing rectangle for the arcs
        arc_width = width / arc_height_ratio

        # Draw the left dome (arc)
        left_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, left_dome_rect, 0, math.pi, 2)

        # Draw the right dome (arc)
        right_dome_rect = pygame.Rect(center_pos[0] + width / 2 - arc_width, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, right_dome_rect, math.pi, 2 * math.pi, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] - height / 2)
        top_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] + height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, top_right_tangent, 2)
        pygame.draw.line(screen, color, bottom_left_tangent, bottom_right_tangent, 2)

    def draw_horizontal_drum(screen, center_pos, width, height, label, arc_height_ratio=2, color=BLACK):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)

        # Calculate the width of the enclosing rectangle for the arcs
        arc_width = width / arc_height_ratio

        # Draw the left dome (arc)
        left_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, left_dome_rect, math.pi / 2, 3 * math.pi / 2, 2)

        # Draw the right dome (arc)
        right_dome_rect = pygame.Rect(center_pos[0] + width / 2 - arc_width, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, right_dome_rect, -math.pi / 2, math.pi / 2, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] - height / 2)
        top_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] + height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, top_right_tangent, 2)
        pygame.draw.line(screen, color, bottom_left_tangent, bottom_right_tangent, 2)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)

    def draw_cstr_with_heating_jacket(screen, center_pos, width, height, color=BLACK):
        # Draw the outer rectangle for the CSTR
        outer_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, width, height)
        pygame.draw.rect(screen, color, outer_rect, 2)

        # Draw the inner rectangle for the heating jacket
        inner_width = width * 0.8
        inner_height = height * 0.9
        inner_rect = pygame.Rect(center_pos[0] - inner_width / 2, center_pos[1] - inner_height / 2, inner_width, inner_height)
        pygame.draw.rect(screen, color, inner_rect, 2)

        # Draw the mixing paddle (vertical line)
        paddle_start = (center_pos[0], center_pos[1] - inner_height / 2)
        paddle_end = (center_pos[0], center_pos[1] + inner_height // 4)
        pygame.draw.line(screen, color, paddle_start, paddle_end, 2)

        # Draw the mixing paddle ellipses
        ellipse_width = inner_width // 3
        ellipse_height = inner_height // 10
        left_ellipse_rect = pygame.Rect(paddle_end[0] - ellipse_width, paddle_end[1] - ellipse_height / 2, ellipse_width, ellipse_height)
        right_ellipse_rect = pygame.Rect(paddle_end[0], paddle_end[1] - ellipse_height / 2, ellipse_width, ellipse_height)

        pygame.draw.ellipse(screen, color, left_ellipse_rect, 2)
        pygame.draw.ellipse(screen, color, right_ellipse_rect, 2)

        # Draw the motor box on top of the paddle
        motor_width = width / 4
        motor_height = height / 8
        motor_rect = pygame.Rect(center_pos[0] - motor_width / 2, paddle_start[1] - motor_height, motor_width, motor_height)
        pygame.draw.rect(screen, color, motor_rect, 2)

        # Draw motor fins
        num_fins = 4
        fin_spacing = motor_width / (num_fins + 1)
        for i in range(num_fins):
            fin_start_x = motor_rect.left + (i + 1) * fin_spacing
            fin_start = (fin_start_x, paddle_start[1] - motor_height)
            fin_end = (fin_start_x, paddle_start[1])
            pygame.draw.line(screen, color, fin_start, fin_end, 2)

    def draw_open_hand_valve(screen, center_pos, triangle_base, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, fill_color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, fill_color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)

    def draw_closed_hand_valve(screen, center_pos, triangle_base, color=BLACK):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with black color
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the right triangle filled with black color
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

    def draw_open_on_off_valve(screen, center_pos, triangle_base, symbol_type, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, fill_color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, fill_color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)

        # Draw the short line connecting touching point to the instrumentation symbol
        line_end = (center_pos[0], center_pos[1] - triangle_height // 2 - 10)
        pygame.draw.line(screen, color, touching_point, line_end, 2)

        # Draw the instrumentation symbol
        instrument_center = (center_pos[0], center_pos[1] - triangle_height // 2 - triangle_base)
        draw_instrumentation(screen, instrument_center, symbol_type)

    def draw_closed_on_off_valve(screen, center_pos, triangle_base, symbol_type, color=BLACK):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the right triangle filled
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the short line connecting touching point to the instrumentation symbol
        line_end = (center_pos[0], center_pos[1] - triangle_height // 2 - 10)
        pygame.draw.line(screen, color, touching_point, line_end, 2)

        # Draw the instrumentation symbol
        instrument_center = (center_pos[0], center_pos[1] - triangle_height // 2 - triangle_base)
        draw_instrumentation(screen, instrument_center, symbol_type)
    
    def render(self, screen):
        # Draw connections
        for connection in self.flowsheet.connections:
            # You might need to get the position and size of the connection elements
            # to draw them correctly, either by adding attributes or by using separate
            # position/size dictionaries
            source_pos = connection.source.position
            destination_pos = connection.destination.position

            # Choose the appropriate draw_* method depending on the type of connection
            self.draw_pipe_with_arrow(screen, source_pos, destination_pos)

        # Draw blocks
        for block in self.flowsheet.blocks:
            # You might need to get the position and size of the block to draw it correctly
            block_pos = block.position
            block_size = block.size

            # Choose the appropriate draw_* method depending on the type of block
            if isinstance(block, StorageTank):
                self.draw_storage_tank(screen, block_pos, block_size)
            # Add more conditions for other block types

        # Draw streams
        for stream in self.flowsheet.streams:
            # You might need to get the position and size of the stream to draw it correctly
            stream_pos = stream.position
            stream_size = stream.size

            # Choose the appropriate draw_* method depending on the type of stream
            self.draw_arrow_with_label(screen, stream_pos, stream_size, stream.name)

        # You can also draw other elements like labels, grid, or background here
# Define the Flowhsheet Class, This is where all the info will be saved
class Flowsheet:
    def __init__(self, name):
        self.name = name
        self.blocks = []
        self.connections = []
        self.streams = []

    def add_block(self, block):
        self.blocks.append(block)

    def add_connection(self, connection):
        self.connections.append(connection)

    def add_stream(self, stream):
        self.streams.append(stream)

    def print_info(self):
        print(f"Components in {self.name}:")
        for block in self.blocks:
            for input_stream in block.inputs:
                for component in input_stream.components:
                    print(f"- {component.name}")
            for output_stream in block.outputs:
                for component in output_stream.components:
                    print(f"- {component.name}")

        print(f"Streams in {self.name}:")
        for stream in self.streams:
            print(f"- {stream.name}")

        print(f"Blocks in {self.name}:")
        for block in self.blocks:
            print(f"- {block.name}")
# The Block is an abstract class that can take on the form of other equiptment, it take input and output
class Block:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

    def add_input(self, input_stream):
        self.inputs.append(input_stream)

    def add_output(self, output_stream):
        self.outputs.append(output_stream)
# This class connects Streams and blocks
class Connection:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
# The Stream class containes components which are connected to blocks via the connections
class Stream:
    def __init__(self, name, components):
        self.name = name
        self.components = components
# Components are the molecules are chemicals that make up streams
class Component:
    def __init__(self, name):
        self.name = name
# ---------------Block subclasses also equiptment---------------
# The StorageTank class is a subclass of Block
class Tank(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The CentrifugalPump class is a subclass of Block
class CentrifugalPump(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The Hand Valve class is a subclass of Block
class HandValve(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The ReliefValve class is a subclass of Block
class ReliefValve(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The ControlValve class is a subclass of Block
class ControlValve(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The FlashTank class is a subclass of Block
class FlashTank(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The DistillationColumn class is a subclass of Block
class DistillationColumn(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The StrippingColumn class is a subclass of Block
class StrippingColumn(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The AbsorptionColumn class is a subclass of Block
class AbsorptionColumn(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The Pipe class is a subclass of Block
class Pipe(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The GeneralHeatExchanger class is a subclass of Block
class GeneralHeatExchanger(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The STHeatExchanger class is a subclass of Block
class STHeatExchanger(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The PlateHeatExchanger class is a subclass of Block
class PlateHeatExchanger(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# Enums for block types
class BlockType(Enum):
    Tank = "Tank"
    CentrifugalPump = "Centrifugal Pump"
    HandValve = "Hand Valve"
    ControlValve = "Control Valve"
    ReliefValve = "Relief Valve"
    FlashTank = "Flash Tank"
    DistillationColumn = "Distillation Column"  # Add this line
    StrippingColumn = "Stripping Column"
    AbsorptionColumn = "Absorption Column"
    Pipe = "Pipe"
    GeneralHeatExchanger = "Heat Exchanger"
    STHeatExchanger = "Shell and Tube Heat Exchanger"
    PlateHeatExchanger = "Plate Heat Exchanger"
    CSTR = "Continuous Stirred Tank Reactor"
    PFR = "Plug Flow Reactor"
    BatchReactor = "Batch Reactor"

# Create the MainMenuPage object
main_menu_page = MainMenuPage()

# Create the PageManager object with MainMenuPage as the initial page
page_manager = PageManager(initial_page=main_menu_page)

clock = pygame.time.Clock()
# Main game loop
while page_manager.running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            page_manager.running = False
            pygame.quit()
        else:
            # Pass the event to the current page
            page_manager.handle_event(event)

    # Check if a new flowsheet has been created
    if isinstance(page_manager.current_page, NewFlowsheetPage) and page_manager.current_page.new_flowsheet_created:
        filename = page_manager.current_page.text_input.strip() + ".pkl"
        print("Flowsheet created: ", filename)
        flowsheet_sim = RunFlowsheetSimulation(filename, screen, page_manager)
        flowsheet_sim.run()

        if flowsheet_sim.go_to_main_menu:
            page_manager.current_page = MainMenuPage()

    # Render the current page
    page_manager.render(screen)
    pygame.display.flip()
    clock.tick(60)

print("Thanks for using Franks Chemical Simulator")

```

