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
