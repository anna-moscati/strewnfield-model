###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

import numpy as np
import os
import matplotlib.pyplot as plt

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import element_conversion
from tudatpy.numerical_simulation.environment_setup import radiation_pressure

# Problem-specific imports
import utilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s

# Meteoroid settings
meteoroid_strength = 1e6  # Pa
meteorodid_density = 3000  # kg/m^3
diameter = 1  # m
meteoroid_mass = meteorodid_density * (4/3) * np.pi * (0.5 * diameter) ** 3 # kg
CH = 0.1  # Heat transfer coefficient
Q = 8.0e6  # Heat of ablation [J/kg]
f = 1.0  # Shape/area factor so that S0 = F*(m/rho_M)^(2/3)

###########################################################################
# CREATE ENVIRONMENT AND PROPAGATE ########################################
###########################################################################

# Initialize dictionary to store the results of the simulation
simulation_results = dict()

results = Util.propagate_once(
    simulation_start_epoch=simulation_start_epoch,
    meteoroid_mass=meteoroid_mass,
    meteoroid_strength=meteoroid_strength,
    integrator_index=1,
    settings_index=2,
    density_met=meteorodid_density,
    CH=CH,
    Q=Q,
    f=f
)

### OUTPUT OF THE SIMULATION ###
print("=== Propagation summary ===")
print(f"Stop reason:            {results['stop_reason']}")
print(f"Termination flags:      {results['termination_flags']}") 

print(f"Final time [s]:         {results['final_time']:.2f}")
print(f"Final altitude [m]:     {results['final_dependent']['altitude']:.2f}")
print(f"Final latitude [deg]:   {np.rad2deg(results['final_dependent']['geodetic_latitude']):.4f}")
print(f"Final longitude [deg]:  {np.rad2deg(results['final_dependent']['longitude']):.4f}")

print(f"Final body mass [kg]:   {results['final_dependent']['body_mass']:.3f}")
print(f"Final dyn. pressure [Pa]: {results['final_dependent']['dynamic_pressure']:.2e}")
print(f"Final Mach number:      {results['final_dependent']['mach_number']:.2f}")

def set_axes_equal(ax):
    """
    Imposta gli assi 3D con scala uguale (1:1:1).
    Funziona come per ax.axis('equal') in 2D.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range) / 2.0

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

def plot_state_history(state_history):
    # Estraggo e ordino i tempi
    tempi = np.array(sorted(state_history.keys()))
    
    # Ricostruisco array posizione e massa
    posizioni = np.array([state_history[t][:3] for t in tempi])  # (N, 3)
    masse = np.array([state_history[t][-1] for t in tempi])      # (N,)
    
    # Plot traiettoria 3D
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(posizioni[:, 0], posizioni[:, 1], posizioni[:, 2], label="Traiettoria")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Traiettoria nello spazio 3D")
    ax1.legend()
    ax1.plot(posizioni[:, 0], posizioni[:, 1], posizioni[:, 2], label="Traiettoria")
    set_axes_equal(ax1)  # 👈 forza la scala 1:1 sugli assi
    
    # Plot massa-tempo
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(tempi, masse, color="orange", label="Massa(t)")
    ax2.set_xlabel("Tempo")
    ax2.set_ylabel("Massa")
    ax2.set_title("Massa in funzione del tempo")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# plot_state_history(results["history"][0])