import tkinter as tk
from tkinter import ttk, messagebox
from fpdf import FPDF
import chemicals
import pandas as pd

# Read the AFS data
#afs_data = pd.read_excel("AFS2018merged_data.xlsx")
class App:
    def __init__(self, root):
        self.root = root
        root.title("Chemicals Safety Data")

        # Layout elements
        ttk.Label(root, text="Enter CAS Number:").pack(pady=10)
        self.cas_entry = ttk.Entry(root)
        self.cas_entry.pack(pady=10)

        ttk.Label(root, text="Or Enter Common Name:").pack(pady=10)
        self.common_name_entry = ttk.Entry(root)
        self.common_name_entry.pack(pady=10)

        ttk.Button(root, text="Fetch Data", command=self.fetch_data).pack(pady=10)

        # Results frame
        self.results_frame = ttk.LabelFrame(root, text="Results", padding="10")
        self.results_frame.pack(pady=20, padx=20, fill='both', expand=True)

        # To show data
        self.data_tree = ttk.Treeview(self.results_frame, columns=("Parameter", "Value"), show="headings")
        self.data_tree.heading("Parameter", text="Parameter")
        self.data_tree.heading("Value", text="Value")
        self.data_tree.pack(fill='both', expand=True)

        ttk.Button(root, text="Export to PDF", command=self.export_pdf).pack(pady=10)

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

        # Check in AFS Data
        #afs_row = afs_data[afs_data['CAS­nr'] == CASRN]
        #if not afs_row.empty:
        #    self.data_tree.insert("", "end", values=("Chemical Name", afs_row.iloc[0]['Chemical Name']))
        #    self.data_tree.insert("", "end", values=("TWA (ppm)", afs_row.iloc[0]['TWA (ppm)']))
        #    self.data_tree.insert("", "end", values=("TWA (mg/m^3)", afs_row.iloc[0]['TWA (mg/m^3)']))
        #    self.data_tree.insert("", "end", values=("STEL (ppm)", afs_row.iloc[0]['STEL (ppm)']))
        #    self.data_tree.insert("", "end", values=("STEL (mg/m3)", afs_row.iloc[0]['STEL (mg/m3)']))
        #    self.data_tree.insert("", "end", values=("Anm.", afs_row.iloc[0]['Anm.']))
        #    self.data_tree.insert("", "end", values=("Notes", afs_row.iloc[0]['Notes']))

        # Fetch and populate data
        # STEL data
        stel_methods = chemicals.safety.STEL_methods(CASRN)
        for method in stel_methods:
            stel_value = chemicals.safety.STEL(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"STEL ({method})", stel_value))
        
        # TWA data
        twa_methods = chemicals.safety.TWA_methods(CASRN)
        for method in twa_methods:
            twa_value = chemicals.safety.TWA(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"TWA ({method})", twa_value))

        # Ceiling data
        ceiling_methods = chemicals.safety.Ceiling_methods(CASRN)
        for method in ceiling_methods:
            ceiling_value = chemicals.safety.Ceiling(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Ceiling ({method})", ceiling_value))

        # Skin data
        skin_methods = chemicals.safety.Skin_methods(CASRN)
        for method in skin_methods:
            skin_value = chemicals.safety.Skin(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Skin ({method})", skin_value))

        # Carcinogen data
        carcinogen_methods = chemicals.safety.Carcinogen_methods(CASRN)
        for method in carcinogen_methods:
            carcinogen_value = chemicals.safety.Carcinogen(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Carcinogen ({method})", carcinogen_value))

        # Flash temperature data
        t_flash_methods = chemicals.safety.T_flash_methods(CASRN)
        for method in t_flash_methods:
            t_flash_value = chemicals.safety.T_flash(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Flash Temp. ({method})", t_flash_value))

        # Autoignition temperature data
        t_autoignition_methods = chemicals.safety.T_autoignition_methods(CASRN)
        for method in t_autoignition_methods:
            t_autoignition_value = chemicals.safety.T_autoignition(CASRN, method=method)
            self.data_tree.insert("", "end", values=(f"Autoignition Temp. ({method})", t_autoignition_value))

        # The other methods, such as those related to LFL, UFL, etc. require additional parameters 
        # (e.g., Hc, atoms). For this simple GUI, these parameters would either need to be hardcoded, 
        # or additional input fields would be required.


    def export_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        for row in self.data_tree.get_children():
            parameter, value = self.data_tree.item(row, "values")
            pdf.cell(200, 10, f"{parameter}: {value}", ln=True)

        pdf.output("chemical_data.pdf")
        messagebox.showinfo("Success", "Data exported to chemical_data.pdf")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
