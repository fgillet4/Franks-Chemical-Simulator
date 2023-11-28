chemicals.dipole.dipole_moment(CASRN, method=None)
chemicals.dipole.dipole_moment_methods(CASRN)
chemicals.dipole.dipole_moment_all_methods = ('CCCBDB', 'MULLER', 'POLING', 'PSI4_2022A')¶
chemicals.dippr.EQ100(T, A=0, B=0, C=0, D=0, E=0, F=0, G=0, order=0)
chemicals.dippr.EQ101(T, A, B, C=0.0, D=0.0, E=0.0, order=0)
chemicals.dippr.EQ102(T, A, B, C=0.0, D=0.0, order=0)
chemicals.dippr.EQ104(T, A, B, C=0.0, D=0.0, E=0.0, order=0)
chemicals.dippr.EQ105(T, A, B, C, D, order=0)
chemicals.dippr.EQ106(T, Tc, A, B, C=0.0, D=0.0, E=0.0, order=0)
chemicals.dippr.EQ107(T, A=0, B=0, C=0, D=0, E=0, order=0)
chemicals.dippr.EQ114(T, Tc, A, B, C, D, order=0)
chemicals.dippr.EQ115(T, A, B, C=0, D=0, E=0, order=0)
chemicals.dippr.EQ116(T, Tc, A, B, C, D, E, order=0)
chemicals.dippr.EQ127(T, A, B, C, D, E, F, G, order=0)
chemicals.dippr.EQ101_fitting_jacobian(Ts, A, B, C, D, E)
chemicals.dippr.EQ102_fitting_jacobian(Ts, A, B, C, D)
chemicals.dippr.EQ105_fitting_jacobian(Ts, A, B, C, D)
chemicals.dippr.EQ106_fitting_jacobian(Ts, Tc, A, B, C, D, E)
chemicals.dippr.EQ107_fitting_jacobian(Ts, A, B, C, D, E)
chemicals.heat_capacity.TRCCp(T, a0, a1, a2, a3, a4, a5, a6, a7)
chemicals.heat_capacity.TRCCp_integral(T, a0, a1, a2, a3, a4, a5, a6, a7, I=0)
chemicals.heat_capacity.TRCCp_integral_over_T(T, a0, a1, a2, a3, a4, a5, a6, a7, J=0)
chemicals.heat_capacity.Shomate(T, A, B, C, D, E)
chemicals.heat_capacity.Shomate_integral(T, A, B, C, D, E)
chemicals.heat_capacity.Shomate_integral_over_T(T, A, B, C, D, E)
class chemicals.heat_capacity.ShomateRange(coeffs, Tmin, Tmax)
calculate(T)

Return heat capacity as a function of temperature.

calculate_integral(Ta, Tb)

Return the enthalpy integral of heat capacity from Ta to Tb.

calculate_integral_over_T(Ta, Tb)

Return the entropy integral of heat capacity from Ta to Tb.

calculate(T)[source]
Return heat capacity as a function of temperature.

Parameters
Tfloat
Temperature, [K]

Returns
Cpfloat
Liquid heat capacity as T, [J/mol/K]

calculate_integral(Ta, Tb)[source]
Return the enthalpy integral of heat capacity from Ta to Tb.

Parameters
Tafloat
Initial temperature, [K]

Tbfloat
Final temperature, [K]

Returns
dHfloat
Enthalpy difference between Ta and Tb, [J/mol]

calculate_integral_over_T(Ta, Tb)[source]
Return the entropy integral of heat capacity from Ta to Tb.

Parameters
Tafloat
Initial temperature, [K]

Tbfloat
Final temperature, [K]

Returns
dSfloat
Entropy difference between Ta and Tb, [J/mol/K]

chemicals.heat_capacity.Poling(T, a, b, c, d, e)
chemicals.heat_capacity.Poling_integral(T, a, b, c, d, e)
chemicals.heat_capacity.Poling_integral_over_T(T, a, b, c, d, e)
chemicals.heat_capacity.PPDS2(T, Ts, C_low, C_inf, a1, a2, a3, a4, a5)
chemicals.heat_capacity.Lastovka_Shaw(T, similarity_variable, cyclic_aliphatic=False, MW=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_integral(T, similarity_variable, cyclic_aliphatic=False, MW=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_integral_over_T(T, similarity_variable, cyclic_aliphatic=False, MW=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_T_for_Hm(Hm, MW, similarity_variable, T_ref=298.15, factor=1.0, cyclic_aliphatic=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_T_for_Sm(Sm, MW, similarity_variable, T_ref=298.15, factor=1.0, cyclic_aliphatic=None, term_A=None)
chemicals.heat_capacity.Lastovka_Shaw_term_A(similarity_variable, cyclic_aliphatic)
chemicals.heat_capacity.Cpg_statistical_mechanics(T, thetas, linear=False)
chemicals.heat_capacity.Cpg_statistical_mechanics_integral(T, thetas, linear=False)
chemicals.heat_capacity.Cpg_statistical_mechanics_integral_over_T(T, thetas, linear=False)
chemicals.heat_capacity.vibration_frequency_cm_to_characteristic_temperature(frequency, scale=1)
chemicals.heat_capacity.Zabransky_quasi_polynomial(T, Tc, a1, a2, a3, a4, a5, a6)
chemicals.heat_capacity.Zabransky_quasi_polynomial_integral(T, Tc, a1, a2, a3, a4, a5, a6)
chemicals.heat_capacity.Zabransky_quasi_polynomial_integral_over_T(T, Tc, a1, a2, a3, a4, a5, a6)
chemicals.heat_capacity.Zabransky_cubic(T, a1, a2, a3, a4)
chemicals.heat_capacity.Zabransky_cubic_integral(T, a1, a2, a3, a4)
chemicals.heat_capacity.Zabransky_cubic_integral_over_T(T, a1, a2, a3, a4)
class chemicals.heat_capacity.ZabranskySpline(coeffs, Tmin, Tmax)
class chemicals.heat_capacity.ZabranskyQuasipolynomial(coeffs, Tc, Tmin, Tmax)
chemicals.heat_capacity.PPDS15(T, Tc, a0, a1, a2, a3, a4, a5)
chemicals.heat_capacity.TDE_CSExpansion(T, Tc, b, a1, a2=0.0, a3=0.0, a4=0.0)
chemicals.heat_capacity.Rowlinson_Poling(T, Tc, omega, Cpgm)
chemicals.heat_capacity.Rowlinson_Bondi(T, Tc, omega, Cpgm)
chemicals.heat_capacity.Dadgostar_Shaw(T, similarity_variable, MW=None, terms=None)
chemicals.heat_capacity.Dadgostar_Shaw_integral(T, similarity_variable, MW=None, terms=None)
chemicals.heat_capacity.Dadgostar_Shaw_integral_over_T(T, similarity_variable, MW=None, terms=None)
chemicals.heat_capacity.Dadgostar_Shaw_terms(similarity_variable)
chemicals.heat_capacity.Perry_151(T, a, b, c, d)
chemicals.heat_capacity.Lastovka_solid(T, similarity_variable, MW=None)
chemicals.heat_capacity.Lastovka_solid_integral(T, similarity_variable, MW=None)
chemicals.heat_capacity.Lastovka_solid_integral_over_T(T, similarity_variable, MW=None)
class chemicals.heat_capacity.PiecewiseHeatCapacity(models)
chemicals.heat_capacity.Cp_data_Poling¶
chemicals.heat_capacity.TRC_gas_data¶
chemicals.heat_capacity.CRC_standard_data¶
chemicals.heat_capacity.Cp_dict_PerryI¶
chemicals.heat_capacity.zabransky_dicts¶
chemicals.heat_capacity.Cp_dict_characteristic_temperatures_adjusted_psi4_2022a¶
chemicals.heat_capacity.Cp_dict_characteristic_temperatures_psi4_2022a¶
chemicals.interface.Brock_Bird(T, Tb, Tc, Pc)
chemicals.interface.Pitzer_sigma(T, Tc, Pc, omega)
chemicals.interface.Sastri_Rao(T, Tb, Tc, Pc, chemicaltype=None)
chemicals.interface.Zuo_Stenby(T, Tc, Pc, omega)
chemicals.interface.Hakim_Steinberg_Stiel(T, Tc, Pc, omega, StielPolar=0.0)
chemicals.interface.Miqueu(T, Tc, Vc, omega)
chemicals.interface.Aleem(T, MW, Tb, rhol, Hvap_Tb, Cpl)
chemicals.interface.Mersmann_Kind_sigma(T, Tm, Tb, Tc, Pc, n_associated=1)
chemicals.interface.sigma_Gharagheizi_1(T, Tc, MW, omega)
chemicals.interface.sigma_Gharagheizi_2(T, Tb, Tc, Pc, Vc)
chemicals.interface.Winterfeld_Scriven_Davis(xs, sigmas, rhoms)
chemicals.interface.Weinaug_Katz(parachors, Vml, Vmg, xs, ys)
chemicals.interface.Diguilio_Teja(T, xs, sigmas_Tb, Tbs, Tcs)
chemicals.interface.sigma_IAPWS(T)
chemicals.interface.API10A32(T, Tc, K_W)
chemicals.interface.Meybodi_Daryasafar_Karimi(rho_water, rho_oil, T, Tc)
chemicals.interface.REFPROP_sigma(T, Tc, sigma0, n0, sigma1=0.0, n1=0.0, sigma2=0.0, n2=0.0)
chemicals.interface.Somayajulu(T, Tc, A, B, C)
chemicals.interface.Jasper(T, a, b)
chemicals.interface.PPDS14(T, Tc, a0, a1, a2)
chemicals.interface.Watson_sigma(T, Tc, a1, a2, a3=0.0, a4=0.0, a5=0.0)
chemicals.interface.ISTExpansion(T, Tc, a1, a2, a3=0.0, a4=0.0, a5=0.0)
All of these coefficients are lazy-loaded, so they must be accessed as an attribute of this module.

chemicals.interface.sigma_data_Mulero_Cachadina
Data from [5] with REFPROP_sigma coefficients.

chemicals.interface.sigma_data_Jasper_Lange
Data as shown in [4] but originally in [3] with Jasper coefficients.

chemicals.interface.sigma_data_Somayajulu
Data from [1] with Somayajulu coefficients.

chemicals.interface.sigma_data_Somayajulu2
Data from [2] with Somayajulu coefficients. These should be preferred over the original coefficients.

chemicals.interface.sigma_data_VDI_PPDS_11
Data from [6] with chemicals.dippr.EQ106 coefficients.