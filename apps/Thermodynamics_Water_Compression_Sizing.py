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
