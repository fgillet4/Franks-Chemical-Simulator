
import tkinter as tk
import platform
from tkinter import ttk, DoubleVar, StringVar, IntVar
import ht.air_cooler
class AirCoolerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Air Cooler Design Tool")

        self.vars = {}

        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        self.container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.create_geometry_input_section()
        self.create_temp_correction_section()
        self.create_noise_estimation_section()
        self.create_pressure_drop_section()
        self.create_heat_transfer_coeff_section()
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=500, height=500)
        self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._on_mousewheel))
        self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=500, height=500)
        self.container.bind("<Map>", lambda e: self._on_canvas_configure(None))
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.container.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Cross-platform mouse wheel scrolling
        if platform.system() == "Windows":
            self.canvas.bind('<MouseWheel>', self._on_mousewheel)
            self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._on_mousewheel))
            self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
        elif platform.system() == "Linux":
            self.canvas.bind('<Button-4>', self._on_mousewheel)
            self.canvas.bind('<Button-5>', self._on_mousewheel)
            self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<Button-4><Button-5>', self._on_mousewheel))
            self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<Button-4><Button-5>'))
        elif platform.system() == "Darwin":
            self.canvas.bind('<MouseWheel>', self._on_mousewheel)
            self.container.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._on_mousewheel))
            self.container.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
            
        self.container.bind("<Configure>", self._on_frame_configure)
        self._on_canvas_configure(None)

    def _on_frame_configure(self, event):
        """Update the scrollbars to match the size of the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        if platform.system() == "Windows" or platform.system() == "Darwin":
            self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
        elif platform.system() == "Linux":
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

    def _on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def _create_frame(self, text, row):
        frame = ttk.LabelFrame(self.container, text=text, padding=(10, 5))
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
        return frame

    def _create_input_fields(self, frame, labels):
        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w")
            self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

    def create_geometry_input_section(self):
        frame = self._create_frame("Geometry Input", 0)
        labels = ["Surface Area (A) [m^2]", "Tube Diameter [m]", "A_min [m^2]", "A_increase", "A_fin [m^2]",
                  "A_tube_showing [m^2]", "Fin Diameter [m]", "Fin Thickness [m]", "Bare Length [m]",
                  "Pitch Parallel [m]", "Pitch Normal [m]", "Tube Rows", "Density [kg/m^3]", "Heat Capacity [J/kg/K]",
                  "Viscosity [Pa*s]", "Thermal Conductivity [W/m/K]", "Fin Thermal Conductivity [W/m/K]"]

        self._create_input_fields(frame, labels)

    def create_temp_correction_section(self):
        frame = self._create_frame("Temperature Correction Factor", 1)
        labels = ["Thi", "Tho", "Tci", "Tco", "Ntp", "Rows"]
        self._create_input_fields(frame, labels)

        row_end = len(labels)
        ttk.Button(frame, text="Calculate", command=self.calc_temp_correction).grid(row=row_end, columnspan=2)
        self.temp_correction_result_var = StringVar()
        ttk.Label(frame, text="Result").grid(row=row_end + 1, column=0)
        ttk.Label(frame, textvariable=self.temp_correction_result_var).grid(row=row_end + 1, column=1)

    def calc_temp_correction(self):
        result = ht.air_cooler.Ft_aircooler(
            self.vars["Thi"].get(),
            self.vars["Tho"].get(),
            self.vars["Tci"].get(),
            self.vars["Tco"].get(),
            self.vars["Ntp"].get(),
            self.vars["Rows"].get()
        )
        self.temp_correction_result_var.set(result)

    def create_noise_estimation_section(self):
        frame = ttk.LabelFrame(self.container, text="Noise Estimation", padding=(10, 5))
        frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        labels = ["Tip Speed", "Power", "Fan Diameter"]
        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0)
            self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

        self.noise_method_var = StringVar(value="GPSA")
        ttk.Radiobutton(frame, text="GPSA", variable=self.noise_method_var, value="GPSA").grid(row=3, column=0)
        ttk.Radiobutton(frame, text="Mukherjee", variable=self.noise_method_var, value="Mukherjee").grid(row=3, column=1)

        self.induced_var = IntVar()
        ttk.Checkbutton(frame, text="Induced", variable=self.induced_var).grid(row=4, column=0)

        ttk.Button(frame, text="Calculate Noise", command=self.calc_noise).grid(row=5, columnspan=2)

        self.noise_result_var = StringVar()
        ttk.Label(frame, text="Noise Result").grid(row=6, column=0)
        ttk.Label(frame, textvariable=self.noise_result_var).grid(row=6, column=1)

    def calc_noise(self):
        if self.noise_method_var.get() == "GPSA":
            result = ht.air_cooler.air_cooler_noise_GPSA(self.vars["Tip Speed"].get(), self.vars["Power"].get())
        else:
            result = ht.air_cooler.air_cooler_noise_Mukherjee(self.vars["Tip Speed"].get(), self.vars["Power"].get(),
                                                              self.vars["Fan Diameter"].get(), bool(self.induced_var.get()))
        self.noise_result_var.set(result)

   
    def create_pressure_drop_section(self):
        frame = ttk.LabelFrame(self.container, text="Pressure Drop Estimation", padding=(10, 5))
        frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        labels = ["Mass Flow Rate (m)", "A_min", "A_increase", "Flow Area Contraction Ratio", 
                  "Tube Diameter", "Fin Height (Only for Low Fin)", "Bare Length", "Pitch Parallel",
                  "Pitch Normal", "Tube Rows", "Density", "Viscosity"]

        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0)
            if label not in self.vars:
                self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

        self.pressure_drop_method_var = StringVar(value="High Fin")
        ttk.Radiobutton(frame, text="High Fin", variable=self.pressure_drop_method_var, value="High Fin").grid(row=len(labels), column=0)
        ttk.Radiobutton(frame, text="Low Fin", variable=self.pressure_drop_method_var, value="Low Fin").grid(row=len(labels), column=1)

        ttk.Button(frame, text="Calculate Pressure Drop", command=self.calc_pressure_drop).grid(row=len(labels) + 1, columnspan=2)

        self.pressure_drop_result_var = StringVar()
        ttk.Label(frame, text="Pressure Drop Result").grid(row=len(labels) + 2, column=0)
        ttk.Label(frame, textvariable=self.pressure_drop_result_var).grid(row=len(labels) + 2, column=1)

    def calc_pressure_drop(self):
        if self.pressure_drop_method_var.get() == "High Fin":
            result = ht.air_cooler.dP_ESDU_high_fin(
                self.vars["Mass Flow Rate (m)"].get(),
                self.vars["A_min"].get(),
                self.vars["A_increase"].get(),
                self.vars["Flow Area Contraction Ratio"].get(),
                self.vars["Tube Diameter"].get(),
                self.vars["Pitch Parallel"].get(),
                self.vars["Pitch Normal"].get(),
                self.vars["Tube Rows"].get(),
                self.vars["Density"].get(),
                self.vars["Viscosity"].get()
            )
        else:
            result = ht.air_cooler.dP_ESDU_low_fin(
                self.vars["Mass Flow Rate (m)"].get(),
                self.vars["A_min"].get(),
                self.vars["A_increase"].get(),
                self.vars["Flow Area Contraction Ratio"].get(),
                self.vars["Tube Diameter"].get(),
                self.vars["Fin Height (Only for Low Fin)"].get(),
                self.vars["Bare Length"].get(),
                self.vars["Pitch Parallel"].get(),
                self.vars["Pitch Normal"].get(),
                self.vars["Tube Rows"].get(),
                self.vars["Density"].get(),
                self.vars["Viscosity"].get()
            )
        self.pressure_drop_result_var.set(result)

    def create_heat_transfer_coeff_section(self):
        frame = ttk.LabelFrame(self.container, text="Heat Transfer Coefficient Estimation", padding=(10, 5))
        frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        labels = ["Mass Flow Rate (m)", "A", "A_min", "A_increase", "A_fin", "A_tube_showing",
                  "Tube Diameter", "Fin Diameter", "Fin Thickness", "Bare Length", "Pitch Parallel",
                  "Pitch Normal", "Tube Rows", "Density", "Cp", "Viscosity", "Thermal Conductivity (k)",
                  "Fin Conductivity (k_fin)"]

        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=i, column=0)
            if label not in self.vars:
                self.vars[label] = DoubleVar()
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=i, column=1)

        methods = ["Briggs Young", "ESDU High Fin", "ESDU Low Fin", "Ganguli VDI"]
        self.htc_method_var = StringVar(value=methods[0])
        
        for i, method in enumerate(methods):
            ttk.Radiobutton(frame, text=method, variable=self.htc_method_var, value=method).grid(row=len(labels) + i, column=0, columnspan=2)

        ttk.Button(frame, text="Calculate Heat Transfer Coefficient", command=self.calc_htc).grid(row=len(labels) + len(methods), columnspan=2)

        self.htc_result_var = StringVar()
        ttk.Label(frame, text="Heat Transfer Coefficient Result").grid(row=len(labels) + len(methods) + 1, column=0)
        ttk.Label(frame, textvariable=self.htc_result_var).grid(row=len(labels) + len(methods) + 1, column=1)

    def calc_htc(self):
        method = self.htc_method_var.get()
        # Variables that might be used by multiple methods, fetched once for efficiency
        m = self.vars["Mass Flow Rate (m)"].get()
        A = self.vars["A"].get()
        A_min = self.vars["A_min"].get()
        A_increase = self.vars["A_increase"].get()
        A_fin = self.vars["A_fin"].get()
        A_tube_showing = self.vars["A_tube_showing"].get()
        Tube_Diameter = self.vars["Tube Diameter"].get()
        Fin_Diameter = self.vars["Fin Diameter"].get()
        Fin_Thickness = self.vars["Fin Thickness"].get()
        Bare_Length = self.vars["Bare Length"].get()
        Density = self.vars["Density"].get()
        Cp = self.vars["Cp"].get()
        Viscosity = self.vars["Viscosity"].get()
        k = self.vars["Thermal Conductivity (k)"].get()
        k_fin = self.vars["Fin Conductivity (k_fin)"].get()
        # Assuming all methods require these parameters, but you can adjust accordingly.

        if method == "Briggs Young":
            result = ht.air_cooler.h_Briggs_Young(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Density, Cp, Viscosity, k, k_fin
            )

        elif method == "ESDU High Fin":
            result = ht.air_cooler.h_ESDU_high_fin(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Density, Cp, Viscosity, k, k_fin
            )

        elif method == "ESDU Low Fin":
            # Assuming the ESDU low fin method requires an extra parameter, say "Fin_Height", but you can adjust accordingly.
            Fin_Height = self.vars["Fin Height (Only for Low Fin)"].get()
            result = ht.air_cooler.h_ESDU_low_fin(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Fin_Height, Density, Cp, Viscosity, k, k_fin
            )

        elif method == "Ganguli VDI":
            Pitch_Parallel = self.vars["Pitch Parallel"].get()
            Pitch_Normal = self.vars["Pitch Normal"].get()
            Tube_Rows = self.vars["Tube Rows"].get()
            
            # This is just an example. Assuming the Ganguli VDI method requires these parameters. 
            # Adjust accordingly based on your function's signature.
            result = ht.air_cooler.h_Ganguli_VDI(
                m, A, A_min, A_increase, A_fin, A_tube_showing, Tube_Diameter, Fin_Diameter,
                Fin_Thickness, Bare_Length, Pitch_Parallel, Pitch_Normal, Tube_Rows, Density, Cp, Viscosity, k, k_fin
            )
        else:
            result = "Method not recognized"

        self.htc_result_var.set(result)

if __name__ == "__main__":
    app = AirCoolerApp()
    app.mainloop()
