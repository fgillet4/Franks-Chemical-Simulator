import math

def LMTD(T1_in, T1_out, T2_in, T2_out):
    delta_T1 = T1_out - T2_in
    delta_T2 = T1_in - T2_out
    return (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)


T1_in = 100.0 # hot fluid inlet temperature in °C
T1_out = 80.0 # hot fluid outlet temperature in °C
T2_in = 20.0 # cold fluid inlet temperature in °C
T2_out = 30.0 # cold fluid outlet temperature in °C

lmtd = LMTD(T1_in, T1_out, T2_in, T2_out)

print("The LMTD is:", lmtd, "°C")
