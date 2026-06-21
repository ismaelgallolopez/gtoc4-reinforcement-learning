# this file contains the parameters for the problem

from tudatpy import constants
from tudatpy.astro import time_representation

spacecraft_dry_mass = 500.0 #kg
spacecraft_propellant_mass = 1000.0 # kg
spacecraft_wet_mass = spacecraft_dry_mass + spacecraft_propellant_mass # kg
scape_velocity_max = 4.0e3 # m/s
Isp_engine = 3000.0 # s
thrust_max = 0.135 # N
launch_interval = [time_representation.DateTime(2015, 1, 1).to_epoch(), 
                   time_representation.DateTime(2025, 1, 1).to_epoch()]
time_mission_max = 10.0 * constants.JULIAN_YEAR # s

# earth's orbital elements in the J2000 frame at epoch 
a_earth = 0.999988049532578 * constants.ASTRONOMICAL_UNIT # m
eccentricity_earth = 1.671681164160e-2
inclination_earth = 0.8854353079654e-3 # deg
lan_earth = 175.40647696473 # deg
arg_periapsis_earth = 287.61577546182 # deg
mean_anomaly_earth = 257.606837077535 # deg
epoch = 54000.0 # MJD2000

# constraints on the rendevouz/flyby
accuracy_position = 1000.0e3 # m
accuracy_velocity = 1.0 # m/s (only for rendezvous)
