import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class WaterVolumeChangeCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Volume Change of Water Calculator")
        self.geometry("600x250")

        # Label and Entry for Initial Temperature
        ttk.Label(self, text="Enter Initial Temperature (°C):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.init_temp_entry = ttk.Entry(self)
        self.init_temp_entry.grid(row=0, column=1, padx=10, pady=5)

        # Label and Entry for Final Temperature
        ttk.Label(self, text="Enter Final Temperature (°C):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.final_temp_entry = ttk.Entry(self)
        self.final_temp_entry.grid(row=1, column=1, padx=10, pady=5)

        # Label and Entry for Initial Volume
        ttk.Label(self, text="Enter Initial Volume (L):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.init_vol_entry = ttk.Entry(self)
        self.init_vol_entry.grid(row=2, column=1, padx=10, pady=5)

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate Volume Change", command=self.calculate_volume_change)
        self.calculate_btn.grid(row=3, column=0, columnspan=2, pady=20)

        # Label to Display the Result
        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def calculate_volume_change(self):
        try:
            T_initial = float(self.init_temp_entry.get())
            T_final = float(self.final_temp_entry.get())
            V_initial = float(self.init_vol_entry.get())

            beta = 0.000207  # coefficient of volume expansion for water in 1/°C
            delta_T = T_final - T_initial

            delta_V = V_initial * beta * delta_T

            self.result_label["text"] = f"The change in volume due to the temperature change is: {delta_V:.4g} L.\n The final volume is: {V_initial + delta_V:.4g} L."

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values!")

if __name__ == "__main__":
    app = WaterVolumeChangeCalculator()
    app.mainloop()
