###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

import numpy as np
from typing import Callable, Sequence
from typing import Any, Dict

# Tudatpy imports
import tudatpy
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment, environment_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.math import root_finders as rf
from tudatpy.kernel.numerical_simulation.environment_setup import aerodynamic_coefficients as aero

###########################################################################
# PROPAGATION SETTING UTILITIES ###########################################
###########################################################################

def get_initial_state(
    simulation_start_epoch: float,
    bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies) -> np.ndarray:
    """
    Converts the initial state to inertial coordinates.

    The initial state is expressed in Earth-centered spherical coordinates.
    These are first converted into Earth-centered cartesian coordinates,
    then they are finally converted in the global (inertial) coordinate
    system.

    Parameters
    ----------
    simulation_start_epoch : float
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    initial_state_inertial_coordinates : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
    """
    

    """!!!!!!!!CHANGE TO GEODETIC AND TRUE DATA!!!!!!!!!!!!"""


    # Set initial spherical elements.
    altitude = 100.0E3
    radius = spice_interface.get_average_radius('Earth') + altitude
    latitude = np.deg2rad(0.6875)
    longitude = np.deg2rad(23.4333)
    speed = 15.0E3
    flight_path_angle = np.deg2rad(-50.0)
    heading_angle = np.deg2rad(90.0)

    # Convert spherical elements to body-fixed cartesian coordinates
    initial_cartesian_state_body_fixed = element_conversion.spherical_to_cartesian_elementwise(
        radius, latitude,  longitude, speed, flight_path_angle, heading_angle)
    # Get rotational ephemerides of the Earth
    Earth_rotational_model = bodies.get_body('Earth').rotation_model
    # Transform the state to the global (inertial) frame
    initial_state_inertial_coordinates = environment.transform_to_inertial_orientation(
        initial_cartesian_state_body_fixed,
        simulation_start_epoch,
        Earth_rotational_model)

    return initial_state_inertial_coordinates

def get_termination_settings(
    simulation_start_epoch: float,
    maximum_duration: float,
    meteoroid_strength: float,
    minimum_mass: float) \
    -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (one day)
    - meteoroid strength (=dynamic pressure at which fragmentation occurs)
    - lower altitude boundary (0 km)
    - mass run-out (complete ablation of the meteoroid)

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] at J2000.
    maximum_duration : float
        Maximum duration of the simulation [s].
    meteoroid_strength : float
        Strength of the meteoroid [Pa].
    meteoroid_mass : float
        Minimum mass of the meteoroid [kg].

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """

    # Create single PropagationTerminationSettings objects

    # Time
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=False)
    
    # Dynamic pressure
    strenght_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.dynamic_pressure('Meteoroid', 'Earth'),
        limit_value=meteoroid_strength,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False,
        # termination_root_finder_settings = rf.bisection()
    )
    
    # Altitude
    altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Meteoroid', 'Earth'),
        limit_value=0.0,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False,
        # termination_root_finder_settings = rf.bisection()
    )

    # Vehicle mass
    mass_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.body_mass('Meteoroid'),
        limit_value=minimum_mass,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False,
        # termination_root_finder_settings = rf.bisection()
    )

    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 strenght_termination_settings,
                                 altitude_termination_settings,
                                 mass_termination_settings]
    
    # Create termination settings object
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings

def get_dependent_variable_save_settings() \
 -> list:
    """
    Retrieves the dependent variables to save.

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.mach_number('Meteoroid', 'Earth'),
        propagation_setup.dependent_variable.altitude('Meteoroid', 'Earth'),
        propagation_setup.dependent_variable.dynamic_pressure('Meteoroid', 'Earth'),
        propagation_setup.dependent_variable.aerodynamic_force_coefficients('Meteoroid', 'Earth'),
        propagation_setup.dependent_variable.geodetic_latitude('Meteoroid', "Earth"),
        propagation_setup.dependent_variable.longitude('Meteoroid', 'Earth'),
        propagation_setup.dependent_variable.body_mass('Meteoroid'),
    ]

    return dependent_variables_to_save
    
def get_integrator_settings(
    integrator_index: int,
    settings_index: int,
    ) \
        -> numerical_simulation.propagation_setup.integrator.IntegratorSettings:
    """

    Integrator settings to be provided to the dynamics simulator.

    It selects a combination of integrator to be used and
    the related setting (tolerance for variable step size integrators). 

    Parameters
    ----------
    integrator_index : int
        Index that selects the integrator type.
    settings_index : int
        Index that selects the tolerance or the step size.

    Returns
    -------
    integrator_settings : tudatpy.numerical_simulation.propagation_setup.integrator.IntegratorSettings

    """
    # Define list of multi-stage integrators
    multi_stage_integrators = [propagation_setup.integrator.CoefficientSets.rkf_45,
                               propagation_setup.integrator.CoefficientSets.rkf_56,
                               propagation_setup.integrator.CoefficientSets.rkdp_87,
                               propagation_setup.integrator.CoefficientSets.rkf_1210]
    
    # Select variable-step integrator
    current_coefficient_set = multi_stage_integrators[integrator_index]
    # Compute current tolerance
    current_tolerance = 10.0 ** (-12.0 + settings_index)
    # Create integrator settings
    blockwise_control_settings = (
        propagation_setup.integrator.step_size_control_blockwise_scalar_tolerance(
            propagation_setup.integrator.standard_cartesian_state_element_blocks(6, 1),
            current_tolerance,
            current_tolerance
        ))
    
    step_size_validation_settings = propagation_setup.integrator.step_size_validation(
        minimum_step=2.0**(-8),
        maximum_step=0.5,
    )

    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=0.125,
        coefficient_set=current_coefficient_set,
        step_size_control_settings=blockwise_control_settings,
        step_size_validation_settings=step_size_validation_settings
    )
        
    return integrator_settings

def cd_of_mach(M):
    """
    Empirical fit for drag coefficient as a function of Mach number.
    Source: Ceplecha et al. (1998), "Meteor Phenomena and Bodies".
    Valid for 0.3 < M < 20.
    """

    """!!!!!CHANGE WUTH BETTER DATA!!!!!"""

    if M > 4.0:
        Cd = 4.0

    else:
        Cd = ((13.2566820069031e-003)*M^4-(101.740672494020e-003)*M^3+(186.705667397016e-003)*M^2+(104.950251795627e-003)*M+291.373797865474e-003)
    
    return Cd

def mass_rate_function(
    bodies,
    rho_M: float,                 # meteoroid material density [kg/m^3]
    CH: float,                    # heat transfer coefficient [-]
    Q: float,                     # heat of ablation [J/kg]
    F: float,                     # shape/area factor so that S0 = F*(m/rho_M)^(2/3)
    cd_of_mach: Callable[[float], float],  # user-supplied: Cd = f(Mach)
) -> Callable[[float], float]:
    """
    Returns a callable dm_dt(t) that models meteoroid ablation in Earth's atmosphere:
        S0 = F * (m / rho_M)^(2/3)
        dm/dt = -Cd(M) * (CH/Q) * rho_a * S0 * Vr^3

    Notes:
    - The vehicle is assumed to be named "Meteoroid" in `bodies`.
    - Requires flight conditions to be available for "Meteoroid" (i.e., AtmosphericFlightConditions).
    - Units assumed SI (kg, m, s, J).
    """

    """!!!CHANGE WITH BETTER MODEL!!!"""

    meteoroid = bodies.get_body("Meteoroid")

    def dm_dt(t: float) -> float:
        fc = meteoroid.flight_conditions    # AtmosphericFlightConditions at current time
        rho_a = fc.density                  # atmospheric density [kg/m^3]
        Vr    = fc.airspeed                 # airspeed [m/s]
        m     = max(meteoroid.mass, 0.0)    # current mass [kg], safeguarded
        M = float(fc.mach_number)
        Cd = float(cd_of_mach(M))

        # Effective area scaling with mass^(2/3)
        S0 = F * (m / rho_M) ** (2.0 / 3.0)

        # Ablation mass-rate
        return -Cd * (CH / Q) * rho_a * S0 * (Vr ** 3)

    return dm_dt

def atmosphere(
    body_settings, 
    central_body: str = "Earth") -> None:
    """
    Attach the US76 Standard Atmosphere to `central_body` inside `body_settings`.

    Use this *before* creating the SystemOfBodies:
        body_settings = environment_setup.get_default_body_settings([...])
        atmosphere(body_settings, central_body="Earth")
        bodies = environment_setup.create_system_of_bodies(body_settings)

    Parameters
    ----------
    body_settings : tudatpy.kernel.numerical_simulation.environment_setup.BodyListSettings
        Container of per-body settings to be passed to `create_system_of_bodies`.
    central_body : str, optional
        Name of the body that provides the atmosphere (default "Earth").

    Notes
    -----
    - This function installs **US76** only (no other models or options).
    - Aerodynamic coefficients/forces (e.g., your Cd(Mach) with S0(m)) should be set
      separately — for example using your `aerodynamics(...)` function.
    """

    """!!!CHANGE TO REAL ATMOSPHERE!!!"""

    # Set the US76 atmospheric model for the chosen central body
    body_settings.get(central_body).atmosphere_settings = environment_setup.atmosphere.us76()

def aerodynamics(
    bodies,
    rho_M: float,     # meteoroid material density [kg/m^3]
    F: float,         # shape factor in S0 = F * (m/rho_M)^(2/3)
    central_body: str = "Earth"
) -> None:
    """
    Configure aerodynamics for the body named "Meteoroid" so that the net aerodynamic force
    matches a model with reference area S0 = F * (m/rho_M)^(2/3) and Cd = cd_of_mach(Mach).

    Implementation details:
    - Set `reference_area = 1.0` and define an effective drag coefficient:
          Cd_eff = cd_of_mach(Mach) * S0(m)
    - The force coefficients are defined as [C_D_eff, C_S, C_L] = [Cd_eff, 0, 0],
      and depend on Mach via `independent_variable_names=[mach_number_dependent]`.
    - The function assumes a body called "Meteoroid" is present in `bodies`.
    - Requires a global/function-scope symbol `cd_of_mach(M)` to be available.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies containing "Meteoroid" and the central body (default "Earth").
    rho_M : float
        Meteoroid bulk density [kg/m^3].
    F : float
        Shape/area factor used in S0 = F * (m/rho_M)^(2/3).
    central_body : str, optional
        Name of the central body providing the atmosphere (default "Earth").
    """

    # Flight conditions for Meteoroid in the given atmosphere
    environment_setup.add_flight_conditions(bodies, "Meteoroid", central_body)
    meteoroid = bodies.get_body("Meteoroid")

    # Define force coefficients depending on Mach.
    # Fold the variable "area" S0(m) into Cd_eff so that S_ref can stay equal to 1.0.
    def force_coefficients():
        # Extract Mach number from the independent variables
        fc = meteoroid.flight_conditions  
        M = float(fc.mach_number)

        # Drag coefficient as a function of Mach 
        Cd = float(cd_of_mach(M))

        # Current mass of the meteoroid (updated during propagation)
        m = bodies.get_body("Meteoroid").mass

        # Effective area scaling S0 = F * (m / rho_M)^(2/3)
        S0 = F * (m / rho_M) ** (2.0 / 3.0)

        # Effective drag coefficient that embeds S0, with S_ref = 1.0
        Cd_eff = Cd * S0

        # Return [CD, CS, CL]; here we use pure drag with zero side/lift
        return [Cd_eff, 0.0, 0.0]

    # Build aerodynamic coefficient settings:
    
    aero_settings = aero.custom_aerodynamic_force_coefficients(
        force_coefficient_function=force_coefficients,
        reference_area=1.0,
        independent_variable_names=[aero.AerodynamicCoefficientsIndependentVariables.mach_number_dependent]
    )

    # Attach the aerodynamic interface to "Meteoroid"
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, "Meteoroid", aero_settings
    )

def get_propagator_settings(
        bodies,
        simulation_start_epoch,
        meteoroide_initial_mass,
        termination_settings,
        dependent_variables_to_save,
        rho_M,    
        CH,         
        Q,       
        F,         
        current_propagator = propagation_setup.propagator.cowell):
    """
    Create and configure the multi-type propagator settings for simulating the trajectory 
    and mass evolution of a meteoroid within a planetary system.

    This function sets up translational dynamics (including gravitational and aerodynamic 
    accelerations) as well as a custom mass-loss model for the meteoroid. The resulting 
    propagator settings combine both translational and mass propagation into a single 
    multi-type propagator object.

    Parameters
    ----------
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        The system of celestial bodies, including Earth, Moon, Sun, and other perturbing bodies,
        required to define accelerations and central bodies.
        
    simulation_start_epoch : float
        Initial epoch of the simulation, expressed in seconds since J2000.

    meteoroide_initial_mass : float
        Initial mass of the meteoroid [kg].

    termination_settings : tudatpy.numerical_simulation.propagation_setup.propagator.TerminationSettings
        Settings that specify the stopping conditions of the propagation (e.g., maximum time, 
        altitude thresholds, etc.).

    dependent_variables_to_save : list of tudatpy.numerical_simulation.propagation_setup.dependent_variable_save_settings
        List of dependent variables to be saved during propagation (e.g., altitude, Mach number, etc.).
    
    rho_M : float
        Meteoroid bulk density [kg/m^3].

    CH : float
        Heat transfer coefficient [-].
    
    Q : float
        Heat of ablation [J/kg].

    F : float
        Shape/area factor used in S0 = F * (m/rho_M)^(

    current_propagator : tudatpy.numerical_simulation.propagation_setup.propagator.PropagatorType, optional
        The numerical propagator to be used for the translational dynamics (default is Cowell’s method).

    Returns
    -------
    tudatpy.numerical_simulation.propagation_setup.propagator.MultiTypePropagatorSettings
        A propagator settings object containing both translational and mass propagation 
        settings, ready to be passed to a numerical simulator.

    Notes
    -----
    - Translational accelerations include:
        * Spherical harmonic gravity up to degree/order 200 for Earth and Moon.
        * Point-mass gravity from Sun, Jupiter, Mars, Venus, and Saturn.
        * Aerodynamic drag from Earth’s atmosphere.
    - Mass evolution is governed by a custom mass-rate function provided via 
      `propagation_setup.mass_rate.custom_mass_rate`.
    - The meteoroid is propagated around Earth as the central body, but subject 
      to perturbations from additional celestial bodies.
    """
    
    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Meteoroid']
    central_bodies = ['Earth']

    # Define accelerations acting on vehicle
    acceleration_settings_on_vehicle = {
        'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(200, 200),
                  propagation_setup.acceleration.aerodynamic()],
        'Moon': [propagation_setup.acceleration.spherical_harmonic_gravity(200, 200)],
        'Sun': [propagation_setup.acceleration.point_mass_gravity()],
        'Jupiter': [propagation_setup.acceleration.point_mass_gravity()],
        'Mars': [propagation_setup.acceleration.point_mass_gravity()],
        'Venus': [propagation_setup.acceleration.point_mass_gravity()],
        'Saturn': [propagation_setup.acceleration.point_mass_gravity()],
    }
        
    # Create acceleration models.
    acceleration_settings = {'Meteoroid': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Retrieve initial state
    initial_state = get_initial_state(simulation_start_epoch, bodies) 

    # Create propagation settings for the translational dynamics
    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        None,
        termination_settings,
        current_propagator,
        dependent_variables_to_save)

    # Create mass rate model
    mass_rate_settings_on_meteoroid = {'Meteoroid': [propagation_setup.mass_rate.custom_mass_rate(mass_rate_function(
        bodies,
        rho_M,   
        CH,         
        Q,        
        F,          
        cd_of_mach=cd_of_mach  
    ))]}

    mass_rate_models = propagation_setup.create_mass_rate_models(bodies,
                                                                 mass_rate_settings_on_meteoroid,
                                                                 acceleration_models)

    # Create mass propagator settings
    mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                 mass_rate_models,
                                                                 np.array([meteoroide_initial_mass]),
                                                                 simulation_start_epoch,
                                                                 None,
                                                                 termination_settings)

    # Create multi-type propagation settings list
    propagator_settings_list = [translational_propagator_settings,
                                mass_propagator_settings]

    # Create multi-type propagation settings object for translational dynamics and mass.
    propagator_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                 None,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save)

    return propagator_settings

def dynamics_system(
    bodies_to_create: Sequence[str] = ('Moon', 'Earth', 'Sun', 'Venus', 'Mars', 'Jupiter', 'Saturn'),
    global_frame_origin: str = 'Earth',
    global_frame_orientation: str = 'J2000',
    integrator_index: int = 0,
    settings_index: int = 0,
    *,
    simulation_start_epoch: float,
    meteoroid_mass: float,
    meteoroid_strength: float,
    maximum_duration: float = constants.JULIAN_DAY, 
    density_met,   
    CH,
    Q,
    f,
    minimum_mass: float = 0.0
):
    """
    Build and return a Tudat dynamics simulator for a meteoroid scenario.

    Parameters
    ----------
    bodies_to_create : Sequence[str], optional
        Celestial bodies to include in the environment model. Defaults to
        ('Moon', 'Earth', 'Sun', 'Venus', 'Mars', 'Jupiter', 'Saturn').
    global_frame_origin : str, optional
        Name of the global frame origin. Default is 'Earth'.
    global_frame_orientation : str, optional
        Orientation of the global frame. Default is 'J2000'.
    integrator_index : int, optional
        Selector passed to ``Util.get_integrator_settings`` to choose the
        integrator family (e.g., RK4, RKF7(8), etc.). Default is 1.
    settings_index : int, optional
        Selector passed to ``Util.get_integrator_settings`` to choose the
        parameterization for the chosen integrator (step size, tolerances, etc.).
        Default is 2.
    simulation_start_epoch : float (required, keyword-only)
        Start epoch of the propagation [s since J2000].
    meteoroid_mass : float (required, keyword-only)
        Mass of the meteoroid [kg].
    meteoroid_strength : float (required, keyword-only)
        Strength parameter used by your termination logic [units per your Util].
    maximum_duration : float, optional
        Maximum simulation duration [s]. Default is one Julian day.
    density_met : float, optional
        Bulk density of the meteoroid material [kg/m^3]. Default is 3000 kg/m^3.
    CH : float
        Heat transfer coefficient [-]. Default is 0.1.
    Q : float
        Heat of ablation [J/kg]. Default is 1e6 J/kg.
    f : float
        Shape/area factor used in S0 = f * (m/rho_M)^(2/3). Default is 1.0.
    minimum_mass : float, optional
        Minimum mass of the meteoroid [kg] before termination. Default is 0.0 kg.

    Returns
    -------
    numerical_simulation.SingleArcSimulator
        A fully constructed Tudat dynamics simulator instance.

    Notes
    -----
    - This function assumes the existence of a ``Util`` module that provides:
        ``get_termination_settings(start_epoch, maximum_duration, meteoroid_strength, meteoroid_mass)``,
        ``get_dependent_variable_save_settings()``, and
        ``get_propagator_settings(bodies, start_epoch, meteoroid_mass, termination_settings, dependent_variables)``,
        as well as ``get_integrator_settings(integrator_index, settings_index)``.
    - If your termination settings should explicitly use ``termination_altitude``,
      add it to the corresponding ``Util.get_termination_settings`` call.
    """

    # ----- ENVIRONMENT -------------------------------------------------------
    body_settings = environment_setup.get_default_body_settings(
        list(bodies_to_create), global_frame_origin, global_frame_orientation)
    atmosphere(body_settings, central_body="Earth")
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Meteoroid
    bodies.create_empty_body('Meteoroid')
    bodies.get_body('Meteoroid').mass = meteoroid_mass
    # aerodynamics(bodies, rho_M=density_met, F=f, central_body="Earth")
    drag_coefficient = 1.2  
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area= 1.0,  # m²
        constant_force_coefficient=[drag_coefficient, 0, 0]  # CD, CS, CL
    )
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, "Meteoroid", aero_coefficient_settings  
    )

    # ----- PROPAGATION SETUP -------------------------------------------------
    # Termination settings 
    termination_settings = get_termination_settings(
        simulation_start_epoch,
        maximum_duration,
        meteoroid_strength,
        minimum_mass
    )

    # Dependent variables to save
    dependent_variables_to_save = get_dependent_variable_save_settings()

    # Propagator settings
    propagator_settings = get_propagator_settings(
        bodies,
        simulation_start_epoch,
        meteoroid_mass,
        termination_settings,
        dependent_variables_to_save,
        density_met,
        CH,        
        Q, 
        f
    )

    # Integrator settings (select via provided indices)
    propagator_settings.integrator_settings = get_integrator_settings(
        integrator_index, settings_index
    )

    # ----- SIMULATOR ---------------------------------------------------------
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    return dynamics_simulator

def propagate_once(
    **dynamics_kwargs) -> Dict[str, Any]:
    """
    Run a single-body propagation with `Util.dynamics_system`
    and classify the stopping reason using Tudat's termination flags.

    This is a thin wrapper that:
      1) runs the dynamics for ONE meteoroid,
      2) extracts the final snapshot (time/state/dependent vars),
      3) infers WHY the propagation stopped from
         `propagation_termination_details.was_condition_met_when_stopping`,
      4) returns a compact result package for higher-level orchestration.

    Parameters
    ----------
    dynamics_kwargs : Dict[str, Any]
        Arguments passed directly to `Util.dynamics_system`, e.g.:
        - simulation_start_epoch : float
        - meteoroid_mass : float
        - meteoroid_strength : float
        - integrator_index : int
        - settings_index : int
        - density_met : float
        - CH : float
        - Q : float
        - f : float

    Returns
    -------
    Dict[str, Any]
        {
          "history": (state_history, dependent_variable_history),
          "final_time": float,
          "final_state": Any,
          "final_dependent": dict,       # parsed fields at final time
          "stop_reason": str,            # "time_limit" | "breakup_q_eq_S" | "ground" | "mass_zero" | "unknown"
          "termination_flags": list[bool]
        }
    """

    # Run the one-body dynamics
    dynamics_simulator = dynamics_system(**dynamics_kwargs)

    # Extract histories
    state_history = dynamics_simulator.state_history
    dep_history = dynamics_simulator.dependent_variable_history

    # Final snapshot
    final_time = list(state_history.keys())[-1]
    final_state = list(state_history.values())[-1]
    dep_end = list(dep_history.values())[-1]

    # Read termination flags 
    term = dynamics_simulator.propagation_termination_details.was_condition_met_when_stopping
    stop_reason = "unknown"
    if isinstance(term, (list, tuple)) and len(term) >= 4:
        if term[0]:
            stop_reason = "time_limit"
        elif term[1]:
            stop_reason = "breakup_q_eq_S"
        elif term[2]:
            stop_reason = "ground"
        elif term[3]:
            stop_reason = "mass_zero"

    # Parse dependent variables at final time 
    def _safe_get(seq, idx, default=None):
        try:
            return seq[idx]
        except Exception:
            return default

    final_dependent = {
        "mach_number": _safe_get(dep_end, 0),
        "altitude": _safe_get(dep_end, 1),
        "dynamic_pressure": _safe_get(dep_end, 2),
        "aero_coefficients": _safe_get(dep_end, 3),
        "geodetic_latitude": _safe_get(dep_end, 6),
        "longitude": _safe_get(dep_end, 7),
        "body_mass": _safe_get(dep_end, 8),
    }

    return {
        "history": (state_history, dep_history),
        "final_time": final_time,
        "final_state": final_state,
        "final_dependent": final_dependent,
        "stop_reason": stop_reason,
        "termination_flags": list(term),
    }