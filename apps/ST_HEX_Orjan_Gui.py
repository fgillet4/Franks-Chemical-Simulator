import os
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime
import numpy as np

# Assuming current_directory is defined as:
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'results' directory
results_dir = os.path.join(current_directory, 'results')

# Check if 'results' directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Generate the filename based on the current date including seconds
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"Shell_Tube_Heat_Exchanger_Results_{current_time}.xlsx"
excel_filename = f"Pipe_Cost_Results_{current_time}.xlsx"
csv_filename = f"Pipe_Cost_Results_{current_time}.csv"

excel_file_path = os.path.join(results_dir, excel_filename)
csv_file_path = os.path.join(results_dir, csv_filename)


def ST_HEX_solver_Orjan(tubeside_flow_l_s, tubeside_temp_in_C, tubeside_temp_out,
                        shellside_flow_l_s, shellside_temp_in_C, shellside_temp_out,
                        tubeside_pressure_in_Pa, shellside_pressure_in_Pa,
                        shell_length_mm, shell_diameter_mm, tube_length_mm, tube_diameter_mm,
                        tube_thickness_mm, tube_fouling_thickness_inner_mm, tube_fouling_thickness_outer_mm,
                        k_fouling_inner, k_fouling_outer, k_tube_material, nr_baffles, nr_tubes,
                        distance_between_tubes, angle_between_tubes, tube_row_length, tubeside_fluid_CAS,
                        shellside_fluid_CAS, tubeside_fluid_common_name, shellside_fluid_common_name,
                        parts, tolerance, cocurrent_or_countercurrent_flow, nr_HEX_passes, nr_heat_exchangers,
                        extra_area_procent,zig_zag_straight):
    
    def Cp_Orjan_etylenglykol(tempC, glykolhalt_procent):
        Cp = Cp_Orjan_vatten(tempC) - (Cp_Orjan_vatten(tempC) - 0.41) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100
        return Cp

    def Cp_Orjan_propylenglykol(tempC, glykolhalt_procent):
        Cp = (Cp_Orjan_vatten(tempC) * 1000 - (Cp_Orjan_vatten(tempC) * 1000 - 3590 + 3.875 * (tempC - 20)) * glykolhalt_procent / 50) / 1000
        return Cp

    def Cp_Orjan_vatten(tempC):
        if tempC < 40:
            Cp = 4.218 - 0.0018 * (tempC - 0) + 0.00004 * (tempC - 0) * (tempC - 20)
        else:
            Cp = 4.178 + 0.00045 * (tempC - 40) + 9.6875 * 10 ** -6 * (tempC - 40) * (tempC - 80) + 4.42708 * 10 ** -8 * (tempC - 40) * (tempC - 80) * (tempC - 120)
        return Cp

    def densitet_Orjan_etylenglykol(tempC, glykol_halt_procent):
        rho = densitet_Orjan_vatten(tempC) + (1080 - 0.75 * (tempC - 20) - densitet_Orjan_vatten(tempC)) * (0.8 * (glykol_halt_procent - 0) + 0.03 * (glykol_halt_procent - 0) * (glykol_halt_procent - 10)) / 100
        return rho

    def densitet_Orjan_propylenglykol(tempC):
        rho = densitet_Orjan_vatten(tempC) + 30
        return rho

    def densitet_Orjan_vatten(tempC):
        if tempC < 100:
            rho = 999.84 - 0.13966667 * (tempC - 0) - 0.004583333 * (tempC - 0) * (tempC - 30) + 1.73457 * 10 ** -5 * (tempC - 0) * (tempC - 30) * (tempC - 60)
        else:
            rho = 958.35 - 0.785 * (tempC - 100) - 0.002138889 * (tempC - 100) * (tempC - 130) - 9.25926 * 10 ** -7 * (tempC - 100) * (tempC - 130) * (tempC - 160)
        return rho

    def dynamisk_visk_Orjan_etylenglykol(tempC):
        mu = densitet_Orjan_etylenglykol(tempC) * kinematisk_visk_Orjan_etylenglykol(tempC)
        return mu

    def dynamisk_visk_Orjan_propylenglykol(tempC, glykolhalt_procent):
        mu = densitet_Orjan_propylenglykol(tempC) * kinematisk_visk_Orjan_propylenglykol(tempC, glykolhalt_procent)
        return mu

    def dynamisk_visk_Orjan_vatten(tempC):
        mu = (1792 - 33.16666667 * (tempC - 0) + 0.36944444 * (tempC - 0) * (tempC - 30) - 0.003006173 * (tempC - 0) * (tempC - 30) * (tempC - 60)) * 10 ** -6
        return mu

    def kinematisk_visk_Orjan_etylenglykol(tempC, glykolhalt_procent):
        if tempC < 80:
            mu = (3.8 - 0.075 * (tempC - 20) + 0.001 * (tempC - 20) * (tempC - 40) - 0.0000104167 * (tempC - 20) * (tempC - 40) * (tempC - 60) - kinematisk_visk_Orjan_vatten(tempC) * 10 ** 6) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        else:
            mu = (1.2 - (0.4 / 20) * (tempC - 80) - kinematisk_visk_Orjan_vatten(tempC) * 10 ** 6) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        return mu

    def kinematisk_visk_Orjan_propylenglykol(tempC, glykolhalt_procent):
        if tempC < 80:
            mu = 5.9 - 0.15 * (tempC - 20) + 0.002375 * (tempC - 20) * (tempC - 40) - 0.0000291667 * (tempC - 20) * (tempC - 40) * (tempC - 60) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        else:
            mu = 1.5 - (0.7 / 20) * (tempC - 80) - kinematisk_visk_Orjan_vatten(tempC) * 10 ** 6 * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100 * 10 ** -6 + kinematisk_visk_Orjan_vatten(tempC)
        return mu

    def kinematisk_visk_Orjan_vatten(tempC):
        mu = dynamisk_visk_Orjan_vatten(tempC) / densitet_Orjan_vatten(tempC)
        return mu

    def D_eq_triangular_pitch(Pt, Do):
        D_eq = 4 * ((math.sqrt(3) / 4 * Pt ** 2 - math.pi * Do ** 2 / 8)) / (math.pi * Do * 0.5)
        return D_eq

    def D_eq_square_pitch(Pt, Do):
        D_eq = 4 * (Pt ** 2 - math.pi * Do ** 2 / 4) / (math.pi * Do)
        return D_eq

    def tube_orientation(S_T, S_L, diameter, zig_zag_or_straight):
        x = S_T / diameter
        y = S_L / diameter

        if zig_zag_or_straight == "straight":
            p00 = -0.5943
            p10 = 0.166
            p01 = 1.62
            p20 = 0.2007
            p11 = -0.3674
            p02 = -0.6684
            p30 = -0.05379
            p21 = 0.05296
            p12 = 0.02012
            p03 = 0.1051
            n = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3

            p00 = 3.447
            p10 = -1.878
            p01 = -2.832
            p20 = 0.3357
            p11 = 0.7697
            p02 = 1.117
            p30 = 0.0145
            p21 = -0.1731
            p12 = 0.01476
            p03 = -0.1903
            C = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3
        else:
            p00 = 0.8551
            p10 = -0.03898
            p01 = -0.7839
            p20 = 0.09826
            p11 = -0.1391
            p02 = 0.8411
            p30 = -0.02162
            p21 = -0.0209
            p12 = 0.06605
            p03 = -0.3466
            p31 = 0.008917
            p22 = -0.00415
            p13 = -0.007945
            p04 = 0.04896
            n = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p30 * x ** 3 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3 + p31 * x ** 3 * y + p22 * x ** 2 * y ** 2 + p13 * x * y ** 3 + p04 * y ** 4

            p00 = 0.9872
            p10 = 2.321
            p01 = -6.566
            p20 = -0.628
            p11 = -5.315
            p02 = 15.27
            p21 = 1.335
            p12 = 3.608
            p03 = -13.23
            p22 = -0.8294
            p13 = -0.7861
            p04 = 4.799
            p23 = 0.1474
            p14 = 0.03581
            p05 = -0.6194
            C = p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3 + p22 * x ** 2 * y ** 2 + p13 * x * y ** 3 + p04 * y ** 4 + p23 * x ** 2 * y ** 3 + p14 * x * y ** 4 + p05 * y ** 5

        return C, n

    def dittus_forced_convection(Re, Pr, n):
        Nu = 0.023 * Re ** 0.8 * Pr ** n
        return Nu

    def Gr_D(T, Ts, Tinf, D, nu):
        g = 9.81
        beta = 1 / T
        Gr_D = g * beta * (Ts - Tinf) * D ** 3 / nu ** 2
        return Gr_D

    def Ra_L(beta, Ts, Tinf, L, nu):
        Ra_L = 9.81 * beta * (Ts - Tinf) * L ** 3 / nu ** 2
        return Ra_L

    def k_Orjan_etylenglykol(tempC, glykolhalt_procent):
        k = k_Orjan_vatten(tempC) - (k_Orjan_vatten(tempC) - 0.41) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100
        return k

    def k_Orjan_propylenglykol(tempC, glykolhalt_procent):
        k = k_Orjan_vatten(tempC) - (k_Orjan_vatten(tempC) - 0.38) * (0.8 * (glykolhalt_procent - 0) + 0.03 * (glykolhalt_procent - 0) * (glykolhalt_procent - 10)) / 100
        return k

    def k_Orjan_vatten(tempC):
        k = 0.553 + 0.002033 * (tempC - 0) - 1.277778 * 10 ** -5 * (tempC - 0) * (tempC - 30) + 5.55556 * 10 ** -8 * (tempC - 0) * (tempC - 30) * (tempC - 60)
        return k

    def Nu_D(h, D, k):
        Nu_D = h * D / k
        return Nu_D

    def Pe_D(velocity, D, L):
        Pe_D = velocity / (D / L)
        return Pe_D

    def Pr(mu, Cp, k):
        Pr = mu * Cp / k
        return Pr

    def Ra_L(Pr, Gr_L):
        Ra_L = Gr_L * Pr
        return Ra_L

    def Re_D(rho, V, D, mu):
        Re = rho * V * D / mu
        return Re

    def Re_L(rho, V, L, mu):
        Re = rho * V * L / mu
        return Re
    # Make a first guess for shellside outlet temperature
    shellside_temp_out_guess = shellside_temp_in_C * 1.5

    # Geometry Calculations
    S_L = math.cos(angle_between_tubes * math.pi / 180) * distance_between_tubes
    S_T = math.sin(angle_between_tubes * math.pi / 180) * distance_between_tubes * 2
    nr_horizontal_tubes = tube_row_length / S_T
    nr_tube_rows = nr_tubes / nr_horizontal_tubes
    width_of_tube_package = 2 * S_L * nr_tube_rows + S_L
    free_area = (distance_between_tubes - tube_diameter_mm) / 1000 * (tube_length_mm / 1000) * nr_horizontal_tubes / (nr_baffles + 1)
    shell_side_media_velocity = shellside_flow_l_s / 1000 / free_area
    tubeside_heat_transfer_area = nr_tubes * tube_diameter_mm * math.pi / 1000 * tube_length_mm / 1000

    heat_transfer_area_1 = tube_diameter_mm * math.pi / 1000 * tube_length_mm * nr_heat_exchangers * nr_HEX_passes / 1000 * nr_tubes / (1 + extra_area_procent)
    heat_transfer_area_2 = (tube_diameter_mm - (tube_diameter_mm - 2 * tube_thickness_mm)) / math.log(tube_diameter_mm / (tube_diameter_mm - 2 * tube_thickness_mm)) * math.pi / 1000 * tube_length_mm * nr_heat_exchangers * nr_HEX_passes / 1000 * nr_tubes / (1 + extra_area_procent)
    heat_transfer_area_ln = (heat_transfer_area_1 - heat_transfer_area_2) / (math.log(heat_transfer_area_1 / heat_transfer_area_2))
    active_area = heat_transfer_area_ln / parts

    # Create an empty list to store the calculated shellside temperatures
    shellside_tempC = []

    guesses = 0
    shellside_tempC_final = 0
    shellside_tempC = []

    while (abs(shellside_temp_in_C - shellside_tempC_final) / shellside_temp_in_C > tolerance):

        tubeside_velocity = (tubeside_flow_l_s / 1000) / (math.pi / 4 * (tube_diameter_mm / 1000 - 2 * tube_thickness_mm / 1000) ** 2 * nr_tubes)

        if cocurrent_or_countercurrent_flow in ["counter_current", "counter", "countercurrent", "CounterCurrent"]:
            shellside_velocity = (shellside_flow_l_s / 1000) / (math.pi / 4 * (shell_diameter_mm / 1000) ** 2 / nr_HEX_passes - 3.14 / 4 * (tube_diameter_mm / 1000) ** 2 * nr_tubes)
        else:
            shellside_velocity = shell_side_media_velocity

        result = [["Del av växlare", "tubeside_tempC", "mu_dynamic_tubeside_fluid", "rho_tubeside_fluid", "Cp_tubeside_fluid", "k_tubeside_fluid", "mu_kinematic_tubeside_fluid", "Re_tubeside", "Pr_tubeside", "tubeside_h",
                   "shellside_tempC", "mu_dynamic_shellside_fluid", "rho_shellside_fluid", "Cp_shellside_fluid", "k_shellside_fluid", "mu_kinematic_shellside_fluid", "Re_shellside", "Pr_shellside", "shellside_h", "active_area", "U_value", "my_v", "P"]]

        if not shellside_tempC:
            shellside_tempC.append(shellside_temp_out_guess)
        else:
            shellside_temp_out_guess = shellside_temp_out_guess + 0.1 * (shellside_temp_in_C - shellside_tempC[-1])
            shellside_tempC.append(shellside_temp_out_guess)

        i = 0
        tubeside_tempC = tubeside_temp_in_C

        while i <= parts:
            # ST_HEX: This function is for calculating the shell and tube heat ex.
            # This model takes into consideration...

            tubeside_pressure_Pa = tubeside_pressure_in_Pa
            shellside_pressure_Pa = shellside_pressure_in_Pa

            # Use specific libraries if the chemical is water
            if tubeside_fluid_CAS == "7732-18-5" or tubeside_fluid_common_name == "water":
                # rho = Mass density of water, [kg/m^3]
                rho_tubeside_fluid = densitet_Orjan_vatten(tubeside_tempC)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_tubeside_fluid = Cp_Orjan_vatten(tubeside_tempC)
                # k = Thermal Capacity, [W/mC]
                k_tubeside_fluid = k_Orjan_vatten(tubeside_tempC)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_tubeside_fluid = dynamisk_visk_Orjan_vatten(tubeside_tempC)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_tubeside_fluid = kinematisk_visk_Orjan_vatten(tubeside_tempC)
                # Velocity calculations based on the mass flux
            # Add other fluids if needed
            elif tubeside_fluid_CAS == "107-21-1" or tubeside_fluid_common_name == "etylenglykol":
                # rho = Mass density of ethylene glycol, [kg/m^3]
                rho_tubeside_fluid = densitet_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_tubeside_fluid = Cp_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_tubeside_fluid = k_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_tubeside_fluid = dynamisk_visk_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_tubeside_fluid = kinematisk_visk_Orjan_etylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
            elif tubeside_fluid_CAS == "57-55-6" or tubeside_fluid_common_name == "propylenglykol":
                # rho = Mass density of propylene glycol, [kg/m^3]
                rho_tubeside_fluid = densitet_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_tubeside_fluid = Cp_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_tubeside_fluid = k_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_tubeside_fluid = dynamisk_visk_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_tubeside_fluid = kinematisk_visk_Orjan_propylenglykol(tubeside_tempC, tubeside_fluid_glycol_procent)
            if shellside_fluid_CAS == "7732-18-5" or shellside_fluid_common_name == "water":
                # rho = Mass density of water, [kg/m^3]
                rho_shellside_fluid = densitet_Orjan_vatten(shellside_tempC[-1])
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_shellside_fluid = Cp_Orjan_vatten(shellside_tempC[-1])
                # k = Thermal Capacity, [W/mC]
                k_shellside_fluid = k_Orjan_vatten(shellside_tempC[-1])
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_shellside_fluid = dynamisk_visk_Orjan_vatten(shellside_tempC[-1])
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_shellside_fluid = kinematisk_visk_Orjan_vatten(shellside_tempC[-1])
                # Velocity calculations based on the mass flux
            elif shellside_fluid_CAS == "107-21-1" or shellside_fluid_common_name == "etylenglykol"or shellside_fluid_common_name == "ehtyleneglycol":
                # rho = Mass density of ethylene glycol, [kg/m^3]
                rho_shellside_fluid = densitet_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_shellside_fluid = Cp_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_shellside_fluid = k_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_shellside_fluid = dynamisk_visk_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_shellside_fluid = kinematisk_visk_Orjan_etylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
            elif shellside_fluid_CAS == "57-55-6" or shellside_fluid_common_name == "propylenglykol":
                # rho = Mass density of propylene glycol, [kg/m^3]
                rho_shellside_fluid = densitet_Orjan_propylenglykol(shellside_tempC[-1])
                # Cp = Isobaric heat capacity, [kJ/(kg*C)]
                Cp_shellside_fluid = Cp_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # k = Thermal Capacity, [W/mC]
                k_shellside_fluid = k_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu_dynamic = Dynamic Viscosity, [kg/ms]
                mu_dynamic_shellside_fluid = dynamisk_visk_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
                # mu=kinematic = Kinematic Viscosity, (kg/ms)
                mu_kinematic_shellside_fluid = kinematisk_visk_Orjan_propylenglykol(shellside_tempC[-1], shellside_fluid_glycol_procent)
            
            # Add other fluids if needed

            # Calculate other properties for different fluids as needed

            Re_tubeside = Re_D(rho_tubeside_fluid, tubeside_velocity, (tube_diameter_mm - 2 * tube_thickness_mm) / 1000, mu_dynamic_tubeside_fluid)
            Re_shellside = Re_D(rho_shellside_fluid, shellside_velocity, tube_diameter_mm / 1000, mu_dynamic_shellside_fluid)
            Pr_tubeside = Pr(mu_dynamic_tubeside_fluid, Cp_tubeside_fluid * 1000, k_tubeside_fluid)
            Pr_shellside = Pr(mu_dynamic_shellside_fluid, Cp_shellside_fluid * 1000, k_shellside_fluid)

            tubeside_h = k_tubeside_fluid / ((tube_diameter_mm - 2 * tube_thickness_mm) / 1000) * 0.023 * Re_tubeside ** 0.8 * Pr_tubeside ** 0.4
            C, n = tube_orientation(S_T / 1000, S_L / 1000, tube_diameter_mm / 1000, zig_zag_straight)

            if cocurrent_or_countercurrent_flow in ["counter_current", "counter", "countercurrent", "CounterCurrent"]:
                shellside_h = k_shellside_fluid / (tube_diameter_mm / 1000) * 0.036 * Re_shellside ** 0.8 * Re_shellside ** 0.33
            else:
                shellside_h = k_shellside_fluid / (tube_diameter_mm / 1000) * C * Re_shellside ** n

            U_value = 1 / (1 / shellside_h + 1 / tubeside_h + (tube_thickness_mm / 1000) / k_tube_material + (tube_fouling_thickness_inner_mm / 1000) / k_fouling_inner + (tube_fouling_thickness_outer_mm / 1000) / k_fouling_outer)
            my_v = tubeside_tempC - U_value / tubeside_h * (tubeside_tempC - shellside_tempC[-1])
            P = tubeside_h * active_area * (tubeside_tempC - my_v) / 1000

            tubeside_tempC -= P / (tubeside_flow_l_s * Cp_tubeside_fluid)
            shellside_tempC[-1] -= P / (shellside_flow_l_s * Cp_shellside_fluid)

            result.append([i, tubeside_tempC, mu_dynamic_tubeside_fluid, rho_tubeside_fluid, Cp_tubeside_fluid, k_tubeside_fluid,
                           mu_kinematic_tubeside_fluid, Re_tubeside, Pr_tubeside, tubeside_h, shellside_tempC[-1],
                           mu_dynamic_shellside_fluid, rho_shellside_fluid, Cp_shellside_fluid, k_shellside_fluid,
                           mu_kinematic_shellside_fluid, Re_shellside, Pr_shellside, shellside_h, active_area, U_value, my_v, P])

            i += 1

        shellside_tempC_final = shellside_tempC[-1]
        guesses += 1

    return result


    

class OutputFrame(tk.Frame):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.__create_widgets()

    def __create_widgets(self):
        self.resultbutton = tk.Button(self, text="Generate Results", width=30, height=5, command=self.result_button_click)
        self.resultbutton.grid(column=0, row=0, padx=5, pady=10)

        self.warninglabel = tk.Label(self, text="");
        self.warninglabel.grid(column=0, row=1, padx=5, pady=10);

    def result_button_click(self):
        failures = 0
        for data in self.master.shared_data:
            if self.master.shared_data[data].get():
                pass
            elif data=="tube_fouling_thickness_inner_mm" or data =="tube_fouling_thickness_outer_mm" or data == "extra_area_procent": #add in or data=="" for any other variable that should be allowed to be 0
                pass
            else:
                failures += 1
                print(self.master.shared_data[data])
                print(failures)
                self.master.shared_data[data].set(0)
        if failures > 0:
            self.warninglabel['text'] = "Failure, unfilled field found!"
            self.warninglabel['fg'] = "red"
        else:
            self.master.calculate()

    def print_result(self, dim, cost):
        self.result_dim_label['text'] = "The cheapest is: " + str(1000 * dim) + "mm"
        self.result_cost_label['text'] = "The cost is: " + '{:0,.2f}'.format(cost).replace(",", "X").replace(".", ",").replace("X", " ") + "kr"

class InputFrame(tk.Frame):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=5)
        self.__create_widgets()

    def __create_widgets(self):
        def create_widget_pair(column, row, text, var):
            label = tk.Label(self, text=text)
            label.grid(column=column, row=row, sticky=tk.W, padx=5, pady=(10, 0))
            entry = tk.Entry(self, width=25, textvariable=self.master.shared_data[var])
            entry.grid(column=column, row=row + 1, sticky=tk.W, padx=5, pady=(5, 10))

        # Add more input widgets here as needed for heat exchanger data
        # For example:
        create_widget_pair(column=0, row=0, text='Tubeside Flow (L/s):', var="tubeside_flow_l_s")
        create_widget_pair(column=0, row=2, text='Tubeside Inlet Temp (C):', var="tubeside_temp_in_C")
        create_widget_pair(column=0, row=4, text='Tubeside Outlet Temp (C):', var="tubeside_temp_out")
        create_widget_pair(column=0, row=6, text='Shellside Flow (L/s):', var="shellside_flow_l_s")
        create_widget_pair(column=0, row=8, text='Shellside Inlet Temp (C):', var="shellside_temp_in_C")
        create_widget_pair(column=0, row=10, text='Shellside Outlet Temp (C):', var="shellside_temp_out")
        create_widget_pair(column=0, row=12, text='Tubeside Pressure (Pa):', var="tubeside_pressure_in_Pa")
        create_widget_pair(column=0, row=14, text='Shellside Pressure (Pa):', var="shellside_pressure_in_Pa")
        create_widget_pair(column=0, row=16, text='Shell Length (mm):', var="shell_length_mm")
        create_widget_pair(column=0, row=18, text='Shell Diameter (mm):', var="shell_diameter_mm")
        create_widget_pair(column=1, row=0, text='Tube Length (mm):', var="tube_length_mm")
        create_widget_pair(column=1, row=2, text='Tube Diameter (mm):', var="tube_diameter_mm")
        create_widget_pair(column=1, row=4, text='Tube Thickness (mm):', var="tube_thickness_mm")
        create_widget_pair(column=1, row=6, text='Tube Fouling Thickness Inner (mm):', var="tube_fouling_thickness_inner_mm")
        create_widget_pair(column=1, row=8, text='Tube Fouling Thickness Outer (mm):', var="tube_fouling_thickness_outer_mm")
        create_widget_pair(column=1, row=10, text='K Fouling Inner:', var="k_fouling_inner")
        create_widget_pair(column=1, row=12, text='K Fouling Outer:', var="k_fouling_outer")
        create_widget_pair(column=1, row=14, text='K Tube Material:', var="k_tube_material")
        create_widget_pair(column=1, row=16, text='Nr Baffles:', var="nr_baffles")
        create_widget_pair(column=1, row=18, text='Nr Tubes:', var="nr_tubes")
        create_widget_pair(column=2, row=0, text='Distance Between Tubes (mm):', var="distance_between_tubes")
        create_widget_pair(column=2, row=2, text='Angle Between Tubes (degrees):', var="angle_between_tubes")
        create_widget_pair(column=2, row=4, text='Tube Row Length (mm):', var="tube_row_length")
        create_widget_pair(column=2, row=6, text='Tubeside Fluid CAS:', var="tubeside_fluid_CAS")
        create_widget_pair(column=2, row=8, text='Shellside Fluid CAS:', var="shellside_fluid_CAS")
        create_widget_pair(column=2, row=10, text='Tubeside Fluid Common Name:', var="tubeside_fluid_common_name")
        create_widget_pair(column=2, row=12, text='Shellside Fluid Common Name:', var="shellside_fluid_common_name")
        create_widget_pair(column=2, row=14, text='Parts:', var="parts")
        create_widget_pair(column=2, row=16, text='Tolerance:', var="tolerance")
        create_widget_pair(column=2, row=18, text='Co-Current or Counter-Current Flow:', var="cocurrent_or_countercurrent_flow")
        create_widget_pair(column=3, row=0, text='Nr HEX Passes:', var="nr_HEX_passes")
        create_widget_pair(column=3, row=2, text='Nr Heat Exchangers:', var="nr_heat_exchangers")
        create_widget_pair(column=3, row=4, text='Extra Area Procent:', var="extra_area_procent")
        create_widget_pair(column=3, row=6, text='Zig Zag Straight:', var="zig_zag_straight")

        # Add more input widgets as needed

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        self.shared_data ={
            "tubeside_flow_l_s": tk.DoubleVar(),
            "tubeside_temp_in_C": tk.DoubleVar(),
            "tubeside_temp_out": tk.DoubleVar(),
            "shellside_flow_l_s": tk.DoubleVar(),
            "shellside_temp_in_C": tk.DoubleVar(),
            "shellside_temp_out": tk.DoubleVar(),
            "tubeside_pressure_in_Pa": tk.DoubleVar(),
            "shellside_pressure_in_Pa": tk.DoubleVar(),
            "shell_length_mm": tk.DoubleVar(),
            "shell_diameter_mm": tk.DoubleVar(),
            "tube_length_mm": tk.DoubleVar(),
            "tube_diameter_mm": tk.DoubleVar(),
            "tube_thickness_mm": tk.DoubleVar(),
            "tube_fouling_thickness_inner_mm": tk.DoubleVar(),
            "tube_fouling_thickness_outer_mm": tk.DoubleVar(),
            "k_fouling_inner": tk.DoubleVar(),
            "k_fouling_outer": tk.DoubleVar(),
            "k_tube_material": tk.DoubleVar(),
            "nr_baffles": tk.DoubleVar(),
            "nr_tubes": tk.DoubleVar(),
            "distance_between_tubes": tk.DoubleVar(),
            "angle_between_tubes": tk.DoubleVar(),
            "tube_row_length": tk.DoubleVar(),
            "tubeside_fluid_CAS": tk.StringVar(),
            "shellside_fluid_CAS": tk.StringVar(),
            "tubeside_fluid_common_name": tk.StringVar(),
            "shellside_fluid_common_name": tk.StringVar(),
            "parts": tk.DoubleVar(),
            "tolerance": tk.DoubleVar(),
            "cocurrent_or_countercurrent_flow": tk.StringVar(),
            "nr_HEX_passes": tk.DoubleVar(),
            "nr_heat_exchangers": tk.DoubleVar(),
            "extra_area_procent": tk.DoubleVar(),
            "zig_zag_straight": tk.StringVar()
        }
        self.__set_default_values()
        self.__create_widgets()
    def __set_default_values(self):
        self.shared_data["tubeside_flow_l_s"].set(317.0)
        self.shared_data["tubeside_temp_in_C"].set(84)
        self.shared_data["tubeside_temp_out"].set(76.5)
        self.shared_data["shellside_flow_l_s"].set(83)
        self.shared_data["shellside_temp_in_C"].set(49)
        self.shared_data["shellside_temp_out"].set(77.7)
        self.shared_data["tubeside_pressure_in_Pa"].set(201325)
        self.shared_data["shellside_pressure_in_Pa"].set(201325)
        self.shared_data["shell_length_mm"].set(7000)
        self.shared_data["shell_diameter_mm"].set(1440)
        self.shared_data["tube_length_mm"].set(5500)
        self.shared_data["tube_diameter_mm"].set(25)
        self.shared_data["tube_thickness_mm"].set(1)
        self.shared_data["tube_fouling_thickness_inner_mm"].set(0)
        self.shared_data["tube_fouling_thickness_outer_mm"].set(0)
        self.shared_data["k_fouling_inner"].set(0.9)
        self.shared_data["k_fouling_outer"].set(0.9)
        self.shared_data["k_tube_material"].set(16)
        self.shared_data["nr_baffles"].set(22)
        self.shared_data["nr_tubes"].set(240)
        self.shared_data["distance_between_tubes"].set(31.8)
        self.shared_data["angle_between_tubes"].set(30)
        self.shared_data["tube_row_length"].set(1520)
        self.shared_data["tubeside_fluid_CAS"].set("7732-18-5")
        self.shared_data["shellside_fluid_CAS"].set("7732-18-5")
        self.shared_data["tubeside_fluid_common_name"].set("water")
        self.shared_data["shellside_fluid_common_name"].set("water")
        self.shared_data["parts"].set(40)
        self.shared_data["tolerance"].set(0.005)
        self.shared_data["cocurrent_or_countercurrent_flow"].set("counter_current")
        self.shared_data["nr_HEX_passes"].set(2)
        self.shared_data["nr_heat_exchangers"].set(1)
        self.shared_data["extra_area_procent"].set(0)
        self.shared_data["zig_zag_straight"].set("straight")
    def __create_widgets(self):
        self.input_frame = InputFrame(self)
        self.input_frame.grid(column=0, row=0)

        self.output_frame = OutputFrame(self)
        self.output_frame.grid(column=1, row=0)
    def calculate(self):
        # Call the heat exchanger calculation function with the provided data   
        tubeside_flow_l_s= self.shared_data['tubeside_flow_l_s'].get()
        tubeside_temp_in_C=self.shared_data['tubeside_temp_in_C'].get()
        tubeside_temp_out=self.shared_data['tubeside_temp_out'].get()
        shellside_flow_l_s=self.shared_data['shellside_flow_l_s'].get()
        shellside_temp_in_C=self.shared_data['shellside_temp_in_C'].get()
        shellside_temp_out=self.shared_data['shellside_temp_out'].get()
        tubeside_pressure_in_Pa=self.shared_data['tubeside_pressure_in_Pa'].get()
        shellside_pressure_in_Pa=self.shared_data['shellside_pressure_in_Pa'].get()
        shell_length_mm=self.shared_data['shell_length_mm'].get()
        shell_diameter_mm=self.shared_data['shell_diameter_mm'].get()
        tube_length_mm=self.shared_data['tube_length_mm'].get()
        tube_diameter_mm=self.shared_data['tube_diameter_mm'].get()
        tube_thickness_mm=self.shared_data['tube_thickness_mm'].get()
        tube_fouling_thickness_inner_mm=self.shared_data['tube_fouling_thickness_inner_mm'].get()
        tube_fouling_thickness_outer_mm=self.shared_data['tube_fouling_thickness_outer_mm'].get()
        k_fouling_inner=self.shared_data['k_fouling_inner'].get()
        k_fouling_outer=self.shared_data['k_fouling_outer'].get()
        k_tube_material=self.shared_data['k_tube_material'].get()
        nr_baffles=self.shared_data['nr_baffles'].get()
        nr_tubes=self.shared_data['nr_tubes'].get()
        distance_between_tubes=self.shared_data['distance_between_tubes'].get()
        angle_between_tubes=self.shared_data['angle_between_tubes'].get()
        tube_row_length=self.shared_data['tube_row_length'].get()
        tubeside_fluid_CAS=self.shared_data['tubeside_fluid_CAS'].get()
        shellside_fluid_CAS=self.shared_data['shellside_fluid_CAS'].get()
        tubeside_fluid_common_name=self.shared_data['tubeside_fluid_common_name'].get()
        shellside_fluid_common_name=self.shared_data['shellside_fluid_common_name'].get()
        parts=self.shared_data['parts'].get()
        tolerance=self.shared_data['tolerance'].get()
        cocurrent_or_countercurrent_flow=self.shared_data['cocurrent_or_countercurrent_flow'].get()
        nr_HEX_passes=self.shared_data['nr_HEX_passes'].get()
        nr_heat_exchangers=self.shared_data['nr_heat_exchangers'].get()
        extra_area_procent=self.shared_data['extra_area_procent'].get()
        zig_zag_straight=self.shared_data['zig_zag_straight'].get()

        self.results = ST_HEX_solver_Orjan(
            tubeside_flow_l_s,
            tubeside_temp_in_C,
            tubeside_temp_out,
            shellside_flow_l_s,
            shellside_temp_in_C,
            shellside_temp_out,
            tubeside_pressure_in_Pa,
            shellside_pressure_in_Pa,
            shell_length_mm,
            shell_diameter_mm,
            tube_length_mm,
            tube_diameter_mm,
            tube_thickness_mm,
            tube_fouling_thickness_inner_mm,
            tube_fouling_thickness_outer_mm,
            k_fouling_inner,
            k_fouling_outer,
            k_tube_material,
            nr_baffles,
            nr_tubes,
            distance_between_tubes,
            angle_between_tubes,
            tube_row_length,
            tubeside_fluid_CAS,
            shellside_fluid_CAS,
            tubeside_fluid_common_name,
            shellside_fluid_common_name,
            parts,
            tolerance,
            cocurrent_or_countercurrent_flow,
            nr_HEX_passes,
            nr_heat_exchangers,
            extra_area_procent,
            zig_zag_straight
        )

        # Extract the relevant columns from the result for plotting and saving as CSV
        T = pd.DataFrame(self.results[1:], columns=self.results[0])
        T_data = T[["Del av växlare", "tubeside_tempC", "shellside_tempC"]]
        self.plot_graph(T_data)

        # Save the result as a CSV file
        T_data.to_csv(csv_file_path, index=False)
        # Save the result as an Excel file
        T_data.to_excel(excel_file_path, sheet_name='sheet1', index=False)
    
    def plot_graph(self, data):
        import matplotlib.pyplot as plt

        x_data = data["Del av växlare"]
        y_data_1 = data["tubeside_tempC"]
        y_data_2 = data["shellside_tempC"]

        tubeside_inlet_temp = float(self.shared_data["tubeside_temp_in_C"].get())
        shellside_inlet_temp = float(self.shared_data["shellside_temp_in_C"].get())

        if tubeside_inlet_temp > shellside_inlet_temp:
            warm_data, cold_data = y_data_1, y_data_2
            warm_label = "Tube Side (Warm Side)"
            cold_label = "Shell Side (Cold Side)"
        else:
            warm_data, cold_data = y_data_2, y_data_1
            warm_label = "Shell Side (Warm Side)"
            cold_label = "Tube Side (Cold Side)"

        flow_type = self.shared_data["cocurrent_or_countercurrent_flow"].get()
        print(flow_type)
        plt.plot(x_data, warm_data, 'r', label=warm_label)
        plt.plot(x_data, cold_data, 'b', label=cold_label)

        arrow_spacing = len(x_data) // 10
        arrow_dx = (x_data.iloc[-1] - x_data.iloc[0]) / 10
        arrow_properties = dict(shape='full', lw=0, length_includes_head=True, head_width=2, head_length=1)

        for i in range(0, len(x_data) - arrow_spacing, arrow_spacing):
            # Warm side arrow follows the line
            dx_warm = x_data[i + arrow_spacing] - x_data[i]
            dy_warm = warm_data[i + arrow_spacing] - warm_data[i]
            plt.arrow(x_data[i], warm_data[i], dx_warm, dy_warm, color='r', **arrow_properties)
            
            # Compute the dx and dy for the cold side regardless of the flow type
            dx_cold = x_data[i + arrow_spacing] - x_data[i]
            dy_cold = cold_data[i + arrow_spacing] - cold_data[i]

            # Check flow type for cold side
            if flow_type in ["counter_current", "counter", "countercurrent", "CounterCurrent"]:
                # Cold side arrow starts from the future point and points back for counter-current flow
                plt.arrow(x_data[i + arrow_spacing], cold_data[i + arrow_spacing], -dx_cold, -dy_cold, color='b', **arrow_properties)
            else:
                # Cold side arrow follows the line for co-current flow
                plt.arrow(x_data[i], cold_data[i], dx_cold, dy_cold, color='b', **arrow_properties)
        # Annotate the starting temperatures
        # Left Side warm text
        plt.annotate(f"{warm_data.iloc[0]:.2f}°C", 
                    (x_data.iloc[0], warm_data.iloc[0]),
                    textcoords="offset points", 
                    xytext=(-40,-7), 
                    ha='center',
                    color='r')
        # Left Side cold text
        plt.annotate(f"{cold_data.iloc[0]:.2f}°C", 
                    (x_data.iloc[0], cold_data.iloc[0]),
                    textcoords="offset points", 
                    xytext=(-40,-5), 
                    ha='center',
                    color='b')

        # Annotate the end temperatures
        # Right Side warm text
        plt.annotate(f"{warm_data.iloc[-1]:.2f}°C", 
                    (x_data.iloc[-1], warm_data.iloc[-1]),
                    textcoords="offset points", 
                    xytext=(40,-10), 
                    ha='center',
                    color='r')
        # Right Side cold text
        plt.annotate(f"{cold_data.iloc[-1]:.2f}°C", 
                    (x_data.iloc[-1], cold_data.iloc[-1]),
                    textcoords="offset points", 
                    xytext=(40,10), 
                    ha='center',
                    color='b')
        plt.title("S&T HEX Temperature Profile")
        plt.xlabel("Slice of Heat Exchanger")
        plt.ylabel("Temperature C")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    app = App()
    app.mainloop()
