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
