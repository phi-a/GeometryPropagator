"""S2 Feasible-Attitude Kernel.

Pure spherical geometry. No orbit math.

Layers:
    vec        - unit-vector arithmetic
    cap        - spherical cap primitive
    region     - boolean intersection, dominance, uncertainty
    constraint - labeled cap builders (FOV, Sun Excl, Earth Limb)
    engine     - build caps, extract State
"""
