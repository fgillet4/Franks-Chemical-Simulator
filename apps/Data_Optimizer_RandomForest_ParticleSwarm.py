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
