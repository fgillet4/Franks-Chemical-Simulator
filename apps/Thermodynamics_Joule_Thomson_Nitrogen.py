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
