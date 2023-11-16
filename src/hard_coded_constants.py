import numpy as np


# set of reasonable whfast simulation timesteps for each planet
# (1/20 of its orbital period for terrestrial planets, 1/30 for giants)
# dt[0] is a placeholder since the planets are indexed starting
# at 1 instad of at 0
# values are in years
dt = [0.00001, 0.012, 0.03, 0.05, 0.09, 0.4, 0.98, 2.7, 5.4]


# array of GM values queried January 2022 (there isn't a way to get
# this from Horizons via API, so we just have to hard code it)
# values for giant planet systems are from Park et al. 2021 DE440
# and DE441, https://doi.org/10.3847/1538-3881/abd414
# all in km^3 s^–2
# G = 6.6743015e-20 #in km^3 kg^–1 s^–2
SS_GM = np.zeros(9)
SS_GM[0] = 132712440041.93938  # Sun
SS_GM[1] = 22031.868551  # Mercury
SS_GM[2] = 324858.592  # Venus
SS_GM[3] = 398600.435507 + 4902.800118  # Earth + Moon
SS_GM[4] = 42828.375816  # Mars system
SS_GM[5] = 126712764.10  # Jupiter system
SS_GM[6] = 37940584.8418  # Saturn system
SS_GM[7] = 5794556.4  # Uranus system
SS_GM[8] = 6836527.10058  # Neptune system

# array of physical radius values queried January 2022
# (again, not possible to pull directly via API)
kmtoau = (1000./149597870700.)  # 1000m/(jpl's au in m) = 6.68459e-9
SS_r = np.zeros(9)
SS_r[0] = 695700.*kmtoau  # Sun
SS_r[1] = 2440.53*kmtoau  # Mercury
SS_r[2] = 6051.8*kmtoau  # Venus
SS_r[3] = 6378.136*kmtoau  # Earth
SS_r[4] = 3396.19*kmtoau  # Mars
SS_r[5] = 71492.*kmtoau  # Jupiter system
SS_r[6] = 60268.*kmtoau  # Saturn system
SS_r[7] = 25559.*kmtoau  # Uranus system
SS_r[8] = 24764.*kmtoau  # Neptune system


# constants needed for accepting orbit fits from non-JPL sources
#
# Find_Orb sun GM as of Nov. 15, 2023 (km^3 s^-2)
find_orb_sunGM = 1.3271243994E+11
stoyear = 1./(365.25*24.*60.*60.)

