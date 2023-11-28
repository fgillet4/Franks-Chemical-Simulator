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
