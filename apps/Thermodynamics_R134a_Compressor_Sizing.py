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
