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
