import chemicals.air

class AirProperties:
    def __init__(self):
        pass

    def rho(self, T, P):
        return chemicals.air.lemmon2000_rho(T, P)

    def P(self, T, rho):
        return chemicals.air.lemmon2000_P(T, rho)

    def P_dew(self, T):
        return chemicals.air.lemmon2000_air_P_dew(T)

    def P_bubble(self, T):
        return chemicals.air.lemmon2000_air_P_bubble(T)

    def rho_dew(self, T):
        return chemicals.air.lemmon2000_air_rho_dew(T)

    def rho_bubble(self, T):
        return chemicals.air.lemmon2000_air_rho_bubble(T)

    def A0(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_A0(tau, delta)

    def dA0_dtau(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_dA0_dtau(tau, delta)

    def d2A0_dtau2(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d2A0_dtau2(tau, delta)

    # ... Continuing the methods:
    def d3A0_dtau3(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d3A0_dtau3(tau, delta)

    def d4A0_dtau4(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d4A0_dtau4(tau, delta)

    def Ar(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_Ar(tau, delta)

    def dAr_dtau(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_dAr_dtau(tau, delta)

    # ... Continuing in the same pattern for all the properties:

    # (You can continue adding functions in this manner. I'll provide a pattern for a couple more, but you can continue this for all functions):

    def d2Ar_dtau2(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d2Ar_dtau2(tau, delta)

    def d3Ar_dtau3(self, T, rho):
        tau = chemicals.air.lemmon2000_air_T_reducing / T
        delta = rho / chemicals.air.lemmon2000_air_rho_reducing
        return chemicals.air.lemmon2000_air_d3Ar_dtau3(tau, delta)

    # ... and so on for the rest of the functions ...

    # As an example for how to handle functions that don't require rho, here's a method for the Henry's law constant:
    def iapws04_Henry_air(self, T):
        return chemicals.air.iapws04_Henry_air(T)

import tkinter as tk
from tkinter import ttk, messagebox

# The previous AirProperties class code should be here

class AirPropertiesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Air Properties Calculator")
        
        # Initialize AirProperties
        self.air = AirProperties()
        
        # User Input
        ttk.Label(root, text="Temperature (K)").grid(row=0, column=0, padx=10, pady=5)
        self.temperature = ttk.Entry(root)
        self.temperature.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(root, text="Pressure (Pa) / Density (mol/m^3)").grid(row=1, column=0, padx=10, pady=5)
        self.pressure_density = ttk.Entry(root)
        self.pressure_density.grid(row=1, column=1, padx=10, pady=5)

        # Buttons for calculations
        self.calculate_rho_button = ttk.Button(root, text="Calculate Density", command=self.calculate_rho)
        self.calculate_rho_button.grid(row=2, column=0, padx=10, pady=5)

        self.calculate_P_button = ttk.Button(root, text="Calculate Pressure", command=self.calculate_P)
        self.calculate_P_button.grid(row=2, column=1, padx=10, pady=5)

        # More Buttons for calculations
        self.calculate_P_dew_button = ttk.Button(root, text="Calculate Dew Point Pressure", command=self.calculate_P_dew)
        self.calculate_P_dew_button.grid(row=4, column=0, padx=10, pady=5)

        self.calculate_P_bubble_button = ttk.Button(root, text="Calculate Bubble Point Pressure", command=self.calculate_P_bubble)
        self.calculate_P_bubble_button.grid(row=4, column=1, padx=10, pady=5)

        self.calculate_rho_dew_button = ttk.Button(root, text="Calculate Dew Point Density", command=self.calculate_rho_dew)
        self.calculate_rho_dew_button.grid(row=5, column=0, padx=10, pady=5)

        self.calculate_rho_bubble_button = ttk.Button(root, text="Calculate Bubble Point Density", command=self.calculate_rho_bubble)
        self.calculate_rho_bubble_button.grid(row=5, column=1, padx=10, pady=5)

        # Output
        self.result_label = ttk.Label(root, text="")
        self.result_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def calculate_rho(self):
        try:
            T = float(self.temperature.get())
            P = float(self.pressure_density.get())
            rho = self.air.rho(T, P)
            self.result_label.config(text=f"Density: {rho} mol/m^3")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please provide valid input values for Temperature and Pressure.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def calculate_P(self):
        try:
            T = float(self.temperature.get())
            rho = float(self.pressure_density.get())
            P = self.air.P(T, rho)
            self.result_label.config(text=f"Pressure: {P} Pa")
        except ValueError:
            messagebox.showerror("Error", "Please provide valid input values for Temperature and Density.")
    def calculate_P_dew(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            P_dew = self.air.P_dew(T)
            self.result_label.config(text=f"Dew Point Pressure: {P_dew} Pa")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")


    # Add this method to the AirPropertiesApp class:
    def validate_float_input(self, input_str):
        """Validates if the input string is a valid float."""
        try:
            if input_str:  # Check if the string is not empty
                float(input_str)
            return True
        except ValueError:
            return False
        
    def calculate_P_bubble(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            P_bubble = self.air.P_bubble(T)
            self.result_label.config(text=f"Bubble Point Pressure: {P_bubble} Pa")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def calculate_rho_dew(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            rho_dew = self.air.rho_dew(T)
            self.result_label.config(text=f"Dew Point Density: {rho_dew} mol/m^3")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def calculate_rho_bubble(self):
        try:
            T = float(self.temperature.get())
            if T < 59.75 or T > 132.6312:
                messagebox.showwarning("Invalid Input", "Please input a temperature within the range of 59.75 K to 132.6312 K.")
                return
            rho_bubble = self.air.rho_bubble(T)
            self.result_label.config(text=f"Bubble Point Density: {rho_bubble} mol/m^3")
        except ValueError as e:
            if str(e) == "could not convert string to float":
                messagebox.showerror("Error", "Please input a valid number for Temperature.")
            else:
                messagebox.showerror("Error", f"An error occurred: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AirPropertiesApp(root)
    root.mainloop()
