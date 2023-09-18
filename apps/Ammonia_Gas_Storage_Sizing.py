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
