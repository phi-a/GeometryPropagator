"""Shared constants for the geometry and thermal-loading models."""

import numpy as np

# -- Thermal environment constants ----------------------------------------
S0 = 1361.0    # W/m^2 Solar constant (AM0)
J_IR = 240.0   # W/m^2 Earth mean OLR
A_ALB = 0.30   #       Earth Bond albedo

# -- Body face normals in body frame --------------------------------------
FACES = {
    '+X': np.array([1.0, 0.0, 0.0]),    # velocity / ram
    '-X': np.array([-1.0, 0.0, 0.0]),   # wake
    '+Y': np.array([0.0, 1.0, 0.0]),    # port / orbit normal
    '-Y': np.array([0.0, -1.0, 0.0]),   # starboard
    '+Z': np.array([0.0, 0.0, 1.0]),    # zenith
    '-Z': np.array([0.0, 0.0, -1.0]),   # nadir
}
