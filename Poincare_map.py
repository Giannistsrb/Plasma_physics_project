import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde

# Define the system of differential equations:
def system(z, y, V, Vb):
    psi, theta = y
    dpsi_dz = 3 * V * np.sin(3 * theta - 2 * z) + 2 * Vb * np.sin(2 * theta - z)
    dtheta_dz = 1 / (1 + 4 * psi ** 4)
    return [dpsi_dz, dtheta_dz]

# Parameters:
V  = 0
Vb = 0 
y0_initial = np.array([[0.2 + 0.05 * i, 0] for i in range(21)])  # Expanded range of initial conditions
z_span = (0, 80000)  #Integration range
t_eval = np.arange(0, 80000, 2 * np.pi)  # Evaluate at multiples of 2π

# Function to solve ODE and generate Poincare plot data:
def generate_poincare_data(V, Vb, y0):
    psipoincare = []
    thetapoincare = []
    
    for y0_single in y0:
        sol = solve_ivp(system, z_span, y0_single, args=(V, Vb), t_eval=t_eval, method='LSODA')
        psipoincare.extend(sol.y[0])
        thetapoincare.extend(sol.y[1] % (2 * np.pi))
    return np.array(psipoincare), np.array(thetapoincare)

# Additional configurations with different V and Vb values:
V_values  = [0, 1e-4, 1e-3, 1e-2, 1e-4, 1e-3, 1e-2,    0, 1e-2, 1e-3]
Vb_values = [0, 1e-4, 1e-3, 1e-2, 1e-3, 1e-2, 1e-4, 1e-4,    0,    0]

for i, (V, Vb) in enumerate(zip(V_values, Vb_values)):
    psipoincare, thetapoincare = generate_poincare_data(V, Vb, y0_initial)
    
    xy = np.vstack([thetapoincare, psipoincare])
    z = gaussian_kde(xy)(xy)
    
    # Create the plot:
    plt.figure(figsize=(10, 8))
    plt.scatter(thetapoincare, psipoincare, c=z, s=1, cmap='hsv')
    plt.colorbar(label='Density')
    plt.title(f'Poincare Surface of Section for $V_1$={V}, $V_2$={Vb}')
    plt.xlabel(r'$\theta$ (in rad)')
    plt.ylabel(r'$\psi$')
    plt.ylim([0.2, 1])  
    plt.xlim([0, 2 * np.pi])  
    plt.grid(True)
    
    # Save the plot as a PNG file:
    plt.savefig(f'poincare_plot_{i+1}.png')
    plt.close()
