import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import chemicals

class ChemicalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chemical Properties GUI")
        self.geometry("900x700")
        
        # Entry for CASRN
        ttk.Label(self, text="Enter CASRN:").pack(pady=10)
        self.cas_entry = ttk.Entry(self)
        self.cas_entry.pack(pady=10)

        # Entry for Common Name
        ttk.Label(self, text="Or, Enter Common Name:").pack(pady=10)
        self.common_name_entry = ttk.Entry(self)
        self.common_name_entry.pack(pady=10)
        
        # Button to Fetch and Display Data
        ttk.Button(self, text="Fetch Data", command=self.fetch_data).pack(pady=20)

        # Treeview to Display Results
        self.data_tree = ttk.Treeview(self, columns=("Property", "Value"), show="headings")
        self.data_tree.heading("Property", text="Property")
        self.data_tree.heading("Value", text="Value")
        self.data_tree.pack(pady=20, fill=tk.BOTH, expand=True)

    def fetch_data(self):
        CASRN = self.cas_entry.get().strip()
        
        # If CAS is empty, try to fetch CAS from common name
        if not CASRN:
            common_name = self.common_name_entry.get().strip()
            if not common_name:
                messagebox.showerror("Error", "Please enter either a CAS number or a common name.")
                return

            try:
                CASRN = chemicals.identifiers.CAS_from_any(common_name)
            except Exception:
                messagebox.showerror("Error", "Unable to identify the CAS number for the provided name.")
                return
        
        # Clear previous entries
        for row in self.data_tree.get_children():
            self.data_tree.delete(row)

        # Tc data
        tc_methods = chemicals.critical.Tc_methods(CASRN)
        for method in tc_methods:
            tc_value = chemicals.critical.Tc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Tc ({method})", tc_value))

        # Pc data
        pc_methods = chemicals.critical.Pc_methods(CASRN)
        for method in pc_methods:
            pc_value = chemicals.critical.Pc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Pc ({method})", pc_value))

        # Vc data
        vc_methods = chemicals.critical.Vc_methods(CASRN)
        for method in vc_methods:
            vc_value = chemicals.critical.Vc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Vc ({method})", vc_value))

        # Zc data
        zc_methods = chemicals.critical.Zc_methods(CASRN)
        for method in zc_methods:
            zc_value = chemicals.critical.Zc(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Zc ({method})", zc_value))

        
    def create_sidebar(self):
        ttk.Label(self.sidebar, text="Select Property:").pack(padx=10, pady=5)
        
        properties = ['Tc', 'Pc', 'Vc', 'Zc']
        for prop in properties:
            ttk.Button(self.sidebar, text=prop, command=lambda p=prop: self.update_content(p)).pack(fill=tk.X, padx=10, pady=5)

    def update_content(self, property_selected):
        # Clear current content
        for widget in self.content.winfo_children():
            widget.destroy()
        
        ttk.Label(self.content, text=f"Inputs for {property_selected}:").pack(pady=10)
        
        # Add input fields based on the selected property
        if property_selected == 'Tc':
            # CASRN Input
            ttk.Label(self.content, text="CASRN:").pack(pady=5)
            self.casrn_entry = ttk.Entry(self.content)
            self.casrn_entry.pack(pady=5)
            
            # Method Selection
            self.tc_methods = tk.StringVar(self)
            self.tc_methods.set("HEOS")  # Default value
            methods_dropdown = ttk.OptionMenu(self.content, self.tc_methods, *chemicals.critical.Tc_all_methods)
            ttk.Label(self.content, text="Method:").pack(pady=5)
            methods_dropdown.pack(pady=5)
        
        # TODO: Add more conditions for other properties Pc, Vc, Zc, and add their respective input fields and methods dropdown
        
        # Add Plot button and Results display
        ttk.Button(self.content, text="Plot", command=self.plot_results).pack(pady=20)
        self.results_label = ttk.Label(self.content, text="Results will be displayed here...")
        self.results_label.pack(pady=20)

    def plot_results(self):
        # Here, process the selected property, method, and input values to generate the plot
        
        # Placeholder for the process
        casrn_value = self.casrn_entry.get()
        method_selected = self.tc_methods.get()
        
        # Call the respective method from chemicals.critical based on inputs and get the results
        
        # TODO: This is a placeholder; replace it with actual function call
        results = f"Results for CASRN: {casrn_value} using {method_selected}"
        self.results_label.config(text=results)
        
        # TODO: Replace this example plotting with actual plotting based on inputs and methods
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(np.linspace(0, 10), np.sin(np.linspace(0, 10)))

        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

# Execute the GUI
app = ChemicalGUI()
app.mainloop()
