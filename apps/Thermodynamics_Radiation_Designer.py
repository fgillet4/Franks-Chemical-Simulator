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
