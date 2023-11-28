import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 1.0, 1.0  # domain size
Nx, Ny = 128, 128  # grid size
dx, dy = Lx/Nx, Ly/Ny

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

X, Y = np.meshgrid(x, y)
# Temperature parameters
T_top = 0.0
T_bottom = 1.0
T = T_top + (T_bottom - T_top) * (Ly - Y) / Ly

# Visualize initial condition
plt.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Initial Temperature Field')
plt.show()
# Simulation parameters
dt = 0.001
Ra = 1e6  # Rayleigh number
Pr = 1.0  # Prandtl number
g = 9.81  # Gravity
alpha = 1e-4  # Thermal expansion coefficient
nu = 1.0/Pr  # Kinematic viscosity

# Abstracted simulation loop
for t in range(1000):
    # This is a gross oversimplification
    buoyancy_force = -g * alpha * (T - T_top)
    T += buoyancy_force * dt

    # Add some artificial diffusion to mimic the effects of viscosity
    T += nu * (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
               np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4*T) * dt
    if t % 100 == 0:
        plt.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Temperature Field at t={t*dt:.2f}')
        plt.pause(0.1)

