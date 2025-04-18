import numpy as np
from scipy.optimize import minimize

# Room/Ambient temperature
T_inf = 21.4  # °C

# Dimensions and Properties of the materials
materials = {
    'brass':     dict(d=0.01265, L=0.306, k=116),
    'copper':    dict(d=0.01275, L=0.306, k=401),
    'steel':     dict(d=0.01267, L=0.306, k=14.9),
    'aluminum':  dict(d=0.01270, L=0.302, k=167),
}

# --- 2) Thermocouple positions and steady-state readings ---
#    Note: x must match the number of Texp entries for each case
experiments = {
    'brass_free':     dict(x=np.array([0, 0.0762, 0.1524, 0.2286, 0.3048]),
                           Texp=np.array([79.57, 52.94, 39.14, 32.88, 30.87])),
    'copper_free':    dict(x=np.array([0, 0.0762, 0.1524, 0.2286]),
                           Texp=np.array([68.22, 59.68, 52.91, 49.84])),
    'steel_free':     dict(x=np.array([0, 0.0762, 0.1524, 0.2286, 0.3048]),
                           Texp=np.array([80.58, 33.01, 23.44, 21.66, 21.52])),
    'aluminum_free':  dict(x=np.array([0, 0.0762, 0.1524, 0.2286, 0.3048]),
                           Texp=np.array([79.57, 52.94, 39.14, 32.88, 30.87])),
    'brass_forced':   dict(x=np.array([0, 0.0762, 0.1524, 0.2286, 0.3048]),
                           Texp=np.array([56.10, 30.95, 24.23, 22.74, 22.31])),
    'copper_forced':  dict(x=np.array([0, 0.0762, 0.1524, 0.2286]),
                           Texp=np.array([44.23, 34.85, 29.77, 28.31])),
    'steel_forced':   dict(x=np.array([0, 0.0762, 0.1524, 0.2286, 0.3048]),
                           Texp=np.array([55.85, 24.11, 22.86, 22.24, 22.17])),
    'aluminum_forced':dict(x=np.array([0, 0.0762, 0.1524, 0.2286, 0.3048]),
                           Texp=np.array([39.25, 27.85, 24.36, 23.32, 23.38])),
}

def fit_h_and_q(material, x, Texp):
    """Return best-fit h for one fin (given its material props and data)"""
    props = materials[material]
    d, L, k = props['d'], props['L'], props['k']
    P = np.pi * d
    A = np.pi * d**2 / 4
    T_b = Texp[0]  # assume first thermocouple sits at the base

    def model_T(h):
        m   = np.sqrt(h * P / (k * A))
        num = np.cosh(m*(L - x)) + (h/(m*k)) * np.sinh(m*(L - x))
        den = np.cosh(m*L)     + (h/(m*k)) * np.sinh(m*L)
        return T_inf + (T_b - T_inf) * num/den

    def objective(h_array):
        Tcalc = model_T(h_array[0])
        return np.sum((Tcalc - Texp)**2)

    res = minimize(objective, x0=30, bounds=[(1, 1e4)])
    h_fit = res.x[0]
    
    m = np.sqrt(h_fit * P / (k * A))
    numer = np.sinh(m*L) + (h_fit/(m*k)) * np.cosh(m*L)
    denom = np.cosh(m*L) + (h_fit/(m*k)) * np.sinh(m*L)
    q_fin = np.sqrt(h_fit * P * k * A) * (T_b - T_inf) * numer/denom
    
    return h_fit, q_fin

# Runs through all of the 8 cases
print(f"{'Case':17s} {'h [W/m²·K]':>12s}   {'q_fin [W]':>12s}")
print("-"*45)
for name, data in experiments.items():
    mat = name.split('_')[0]
    h_val, q_val = fit_h_and_q(mat, data['x'], data['Texp'])
    print(f"{name:17s} {h_val:12.1f}   {q_val:12.3f}")