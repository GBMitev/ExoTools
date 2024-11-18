from .dependencies import *

def partition_function(temperature, states):
    c_2 = 1.432 #cm K   https://doi.org/10.1016/0031-8914(49)90062-2
    term_value = states["E"].to_numpy()
    degeneracy = states["gns"].to_numpy()
        
    exponent = -1*c_2*term_value/temperature
    exponent = np.exp(exponent)

    partfunc = (np.sum(degeneracy*exponent))
    return partfunc

def partition_function_temperature(temperature_range, states):
    partfunc = []
    for T in temperature_range:
        partfunc.append(partition_function(T, states))
    return partfunc

def partition_function_maxJ(temperature, states):
    J_range = sorted(states.J.unique())
    partfunc = []
    for J in J_range:
        current_J_states = states[states["J"]<=J]
        partfunc.append(partition_function(temperature, current_J_states))
    return J_range,partfunc

def read_stick_output(path, predicted_shifts = True):
    
    from pandas import DataFrame
        
    with open(path) as file:
        lines = file.readlines()
    n = 0
    l = lines[n]
    while "Spectrum type = ABSORPTION" not in l:
        n+=1
        l = lines[n]

    m = n+1
    while "Total intensity" not in l:
        m+=1
        l = lines[m]

    start = n+2
    end   = m-1

    lines = lines[start:end]
    rows  = []
    for l in lines:
        rows.append(l.replace("\n","").replace("<-","").split())
    if predicted_shifts == True:
        transition_columns = ["nu","I","J_upper","E_upper","J_lower","E_lower","Unc_upper","Lifetime_upper","Lande_upper","tau_upper","e/f_upper","Manifold_upper","v_upper","Lambda_upper","Sigma_upper","Omega_upper","Type_upper","E_calc_upper","Unc_lower","Lifetime_lower","Lande_lower","tau_lower","e/f_lower","Manifold_lower","v_lower","Lambda_lower","Sigma_lower","Omega_lower","Type_lower","E_calc_lower"]
    else:
        transition_columns = ["nu","I","J_upper","E_upper","J_lower","E_lower","tau_upper","e/f_upper","Manifold_upper","v_upper","Lambda_upper","Sigma_upper","Omega_upper","tau_lower","e/f_lower","Manifold_lower","v_lower","Lambda_lower","Sigma_lower","Omega_lower"]
    
    stick = DataFrame(rows, columns = transition_columns)

    stick["nu"]            = stick["nu"]            .astype("float")
    stick["I"]             = stick["I"]             .astype("float")
    stick["J_upper"]       = stick["J_upper"]       .astype("float")
    stick["E_upper"]       = stick["E_upper"]       .astype("float")
    stick["J_lower"]       = stick["J_lower"]       .astype("float")
    stick["E_lower"]       = stick["E_lower"]       .astype("float")
    stick["tau_upper"]     = stick["tau_upper"]     .astype("str")
    stick["e/f_upper"]     = stick["e/f_upper"]     .astype("str")
    stick["Manifold_upper"]= stick["Manifold_upper"].astype("str")
    stick["v_upper"]       = stick["v_upper"]       .astype("int")
    stick["Lambda_upper"]  = stick["Lambda_upper"]  .astype("float")
    stick["Sigma_upper"]   = stick["Sigma_upper"]   .astype("float")
    stick["Omega_upper"]   = stick["Omega_upper"]   .astype("float")
    stick["tau_lower"]     = stick["tau_lower"]     .astype("str")
    stick["e/f_lower"]     = stick["e/f_lower"]     .astype("str")
    stick["Manifold_lower"]= stick["Manifold_lower"].astype("str")
    stick["v_lower"]       = stick["v_lower"]       .astype("int")
    stick["Lambda_lower"]  = stick["Lambda_lower"]  .astype("float")
    stick["Sigma_lower"]   = stick["Sigma_lower"]   .astype("float")
    stick["Omega_lower"]   = stick["Omega_lower"]   .astype("float")
    
    if predicted_shifts == True:
        stick["Unc_upper"]     = stick["Unc_upper"]     .astype("float")
        stick["Unc_lower"]     = stick["Unc_lower"]     .astype("float")
        stick["Lifetime_upper"]= stick["Lifetime_upper"].astype("float")
        stick["Lifetime_lower"]= stick["Lifetime_lower"].astype("float")
        stick["Lande_upper"]   = stick["Lande_upper"]   .astype("float")
        stick["Lande_lower"]   = stick["Lande_lower"]   .astype("float")
        stick["Type_upper"]    = stick["Type_upper"]    .astype("str")
        stick["Type_lower"]    = stick["Type_lower"]    .astype("str")
        stick["E_calc_upper"]  = stick["E_calc_upper"]  .astype("float")
        stick["E_calc_lower"]  = stick["E_calc_lower"]  .astype("float")
        
    return stick

def read_stick(path):
    stick = pd.read_csv(path, sep = "\s+",names = ["Energy","Absorption"])
    return stick

def calculate_intensity(A, g_final, nu, E_init, temperature, partfunc):
    c = 29979245800.0000 #cm s-1 from Duo
    c_2 = 1.432 #cm K   https://doi.org/10.1016/0031-8914(49)90062-2

    Intes_term_1 = (g_final*A)/(8*np.pi*c*nu**2)
    Intes_term_2 = np.exp(-c_2*E_init/temperature)*(1-np.exp(-c_2*nu/temperature))/partfunc
    Intes = Intes_term_1*Intes_term_2
    return Intes


def calculate_intensity_NLTE(A, g_final, nu, E_init):
    c = 29979245800.0000 #cm s-1 from Duo
    Intes_term_1 = (g_final*A)/(8*np.pi*c*nu**2)
    Intes = Intes_term_1
    return Intes