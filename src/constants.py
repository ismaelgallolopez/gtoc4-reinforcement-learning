# this file contains the parameters for the problem

from tudatpy import constants
from tudatpy.astro import time_representation

spacecraft_dry_mass = 500.0 #kg
spacecraft_propellant_mass = 1000.0 # kg
spacecraft_wet_mass = spacecraft_dry_mass + spacecraft_propellant_mass # kg
scape_velocity_max = 4.0e3 # m/s
Isp_engine = 3000.0 # s
thrust_max = 0.135 # N
# the legal launch window is MJD 57023-61041, i.e. 2015-01-01 through the end of 2025, 4018 days
# wide. The upper bound was transcribed as 2025-01-01, which is 3653 days -- 365 days of the legal
# window, ~10%, silently discarded (legality track, phase 2). Note tudatpy's DateTime(y, m, d)
# resolves to 12:00, not 00:00, so these are MJD 57023.5 and 61041.5: half a day inside the stated
# window at each end, which is conservative and therefore legal.
launch_interval = [time_representation.DateTime(2015, 1, 1).to_epoch(),
                   time_representation.DateTime(2026, 1, 1).to_epoch()]
time_mission_max = 10.0 * constants.JULIAN_YEAR # s

# earth's orbital elements in the J2000 frame at epoch
a_earth = 0.999988049532578 * constants.ASTRONOMICAL_UNIT # m
eccentricity_earth = 1.671681163160e-2
inclination_earth = 0.8854353079654e-3 # deg
lan_earth = 175.40647696473 # deg
arg_periapsis_earth = 287.61577546182 # deg
mean_anomaly_earth = 257.60683707535 # deg
epoch = 54000.0 # MJD (not MJD2000: catalog.mjd_to_et subtracts MJD_J2000 = 51544.5 from it,
                # and the asteroid catalog's own epoch column, 54800, is likewise plain MJD)

# constraints on the rendevouz/flyby
accuracy_position = 1000.0e3 # m
accuracy_velocity = 1.0 # m/s (only for rendezvous)

# constants and conversion (from the problem statement)
sun_gravitational_parameter = 132712440018e9 # m^3/s^2
