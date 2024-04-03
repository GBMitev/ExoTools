# %%
def states_df(path, manifold_to_state = None):
    from pandas import read_csv
    from numpy import select

    column_names = ["NN","E","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]
    df = read_csv(f"{path}", sep='\s+',names = column_names)

    tau_cond = [df["tau"] == "+", df["tau"]=="-"]
    tau_vals = [1.0,-1.0]

    df["tau"] = select(tau_cond, tau_vals)
    if manifold_to_state == None:
        return df
    else:
        manifold_cond = [df["Manifold"]==i for i in manifold_to_state.keys()]
        manifold_vals = [i for i in manifold_to_state.values()]

        df["State"] = select(manifold_cond, manifold_vals)

        return df
    
def trans_df(path):
    from pandas import read_csv
    
    df = read_csv(path, delim_whitespace = True,names=["nf","ni","A","nu"])

    return df

def get_full_trans(states, trans):

    trans  = trans  if type(trans)  != str else trans_df(trans)
    states = states if type(states) != str else states_df(states)

    full = trans.merge(states, left_on = "nf", right_on = "NN").drop(columns=["NN"])
    full = full.merge(states, left_on = "ni", right_on = "NN", suffixes = ["_upper","_lower"]).drop(columns = ["NN"])

    return full
    
def filter_trans(states_path, trans_path, label, upper, lower, trans_style = True):
    states = states_df(states_path)[["NN",label]]
    trans  = trans_df(trans_path)

    trans = trans.merge(states, left_on = "nf", right_on = "NN").drop(columns=["NN"])
    trans = trans.merge(states, left_on = "ni", right_on = "NN", suffixes = ["_upper","_lower"]).drop(columns = ["NN"])

    upper = upper if type(upper) == list else [upper]
    lower = lower if type(lower) == list else [lower]

    upper_label, lower_label = label+"_upper", label+"_lower"

    upper = upper if upper != [None] else trans[upper_label].unique()
    lower = lower if lower != [None] else trans[lower_label].unique()

    trans = trans[(trans[upper_label].isin(upper))&(trans[lower_label].isin(lower))]

    return trans.drop(columns=[upper_label, lower_label]) if trans_style == True else trans

def write_trans_file(trans, fname):
    lines = []
    for nf, ni, A, nu in trans.itertuples(index = False):
        line = \
            f"{nf:12d}"+" "+\
                f"{ni:12d}"+" "*2+\
                f"{A:10.4e}".upper()+" "*8+\
                f"{nu:12.6f}"+"\n"
        
        lines.append(line)

    with open(fname, "w") as file:  
        file.writelines(lines)  
    return "done"

def write_states_file(states, fname):
    lines = []
    for NN, E, gns, J,tau, ef, Manifold, v, Lambda, Sigma, Omega in states.itertuples(index=False):
        if J%1 == 0:
            J_line = f"{J:1d}"+" "
        else:
            J_line = f"{J:4.1f}"+" "
    
        if type(tau) != str:
            tau = "+" if tau == 1 else "-"

        line = \
            f"{NN:12d}"+" "+\
            f"{E:{12}.6f}"[:12]+" "+\
            f"{gns:6d}"+" "*4+\
            J_line+\
            f"{tau}"+" "+\
            f"{ef}"+" "+\
            f"{Manifold:<{10}}"+" "+\
            f"{v:3d}"+"  "+\
            f"{Lambda:1d}"+" "*4+\
            f"{Sigma:>4.1f}"+" "*4+\
            f"{Omega:>4.1f}"+"\n"

        lines.append(line)
    
    with open(fname, "w") as file:
        file.writelines(lines)
    
    return fname+"has been written :)"
    

def isolate_bound_transitions(states, trans,state_energy_limits):
    trans  = trans  if type(trans)  != str else trans_df(trans)
    states = states if type(states) != str else states_df(states)

    for state, energy_limits in state_energy_limits.items():
        states = states[
            ((states["Manifold"]==state)&(states["E"].between(*energy_limits))) |
            (states["Manifold"]!=state)
            ]

    trans = get_full_trans(states, trans)[["nf","ni","A","nu"]].sort_values("nu")
    return states, trans

def plot_xsec(path, log_scale = True, **kwargs):
    from matplotlib.pyplot import plot, figure
    from numpy import loadtxt, log10

    Lambda, Sigma = loadtxt(path, unpack = True)

    Sigma = Sigma if log_scale == False else log10(Sigma)

    plot(Lambda, Sigma, **kwargs)
    
def format_xsec_plot(cutoff = None, fontsize = None, print_legend = True):
    from matplotlib.pyplot import ylim, legend, xticks, yticks, xlabel, ylabel

    fontsize = fontsize if fontsize is not None else 20
    if cutoff is not None:
        ylim(bottom = cutoff)
    
    if print_legend == True:
        legend(loc = "best")
    
    xticks(fontsize = fontsize)
    yticks(fontsize = fontsize)

    xlabel(r"$\lambda$ / cm$^{-1} \times 10^{-3}$", fontsize = fontsize)
    ylabel(r"$\sigma$ / cm$^2$/ molecule", fontsize = fontsize)

def write_dat(df, fname):
    first_line = ""
    for col in df.columns:
        if len(col) > 12:
            raise ValueError("Column names must be fewer than 12 characters long")
        first_line += col[:12].rjust(12)+"\t"
    first_line = first_line[:-1]+"\n"
    lines = [first_line]

    for values in df.itertuples(index = False):
        fstring = f""
        for val in values:
            fstring += f"{val:{12}.8f}"[:12]+"\t"
        
        fstring = fstring[:-1]+"\n"
        lines.append(fstring)
    with open(fname, "w") as file:    
        file.writelines(lines)
    print("done")

def predicted_shifts(states, marvel):
    return "Hello World!"

def read_wavefunction(wavefunction_path, **kwargs):
    from pandarallel import pandarallel, core
    from pandas import read_csv

    cores = kwargs.get("cores", core.NB_PHYSICAL_CORES)

    pandarallel.initialize(nb_workers=cores, progress_bar=True, verbose=0)

    wavefunction = read_csv(wavefunction_path, sep = "\s+", skiprows = 1, skipfooter = 1, engine = "python",names = ["wavefunction","||","J","parity","NN"])
    wavefunction = wavefunction.drop(columns = ["||"])
    wavefunction["tau"] = wavefunction.parallel_apply(lambda x: 1 if x["parity"]==0 else -1, axis = 1)
    wavefunction = wavefunction[["wavefunction","J","tau","NN"]]
    return wavefunction 
    

def trim_wavefunction(R, wavefunc, thresh_delta_r):
    max_R = R[-1]
    min_R = max_R-thresh_delta_r
    data = [[R[num], wavefunc[num]] for num, r in enumerate(R) if min_R <= r <= max_R]
    R, wavefunc = zip(*data)
    return R, wavefunc

def filter_wavefunction(wf, J, tau, NN):
    wf = wf[
        (wf["J"]==J)&
        (wf["tau"]==tau)&
        (wf["NN"]==NN)
        ]["wavefunction"].to_numpy()
    return wf

def integrate_wavefunction(wf, R, thresh_delta_r, J, tau, NN):
    from scipy.integrate import trapezoid
    
    wf = filter_wavefunction(wf, J, tau, NN)
    R, wf = trim_wavefunction(R, wf, thresh_delta_r)
    dR = R[1]-R[0]
    integ = trapezoid(wf, R, dx = dR)
    return integ

def integrate_all_wavefunctions(wf, R, thresh_delta_r, **kwargs):
    from pandarallel import pandarallel, core
    cores = kwargs.get("cores",core.NB_PHYSICAL_CORES)
    pandarallel.initialize(progress_bar=True, nb_workers=cores)

    integrals = wf.groupby(["J","tau","NN"], as_index=False).size()[["J","tau","NN"]]
    integrals["Integ"] = integrals.parallel_apply(lambda x: integrate_wavefunction(wf, R, thresh_delta_r, x["J"], x["tau"], x["NN"]), axis = 1)
    return integrals

def get_NN(df):
    
    tau_p = df[df["tau"]==1].sort_values(["J","E"])
    tau_n = df[df["tau"]==-1].sort_values(["J","E"])

    Numbered = pd.DataFrame(columns = [*tau_p.columns])

    for J in df["J"].unique():
        p = tau_p[tau_p["J"]==J].reset_index(drop = True)
        n = tau_n[tau_n["J"]==J].reset_index(drop = True)

        p["NN"] = p.index.values+1
        n["NN"] = n.index.values+1

        Numbered = pd.concat([Numbered, p])
        Numbered = pd.concat([Numbered, n])
    
    return Numbered

def get_eigenvalues(output_path):
    import subprocess as sp
    from pandas import read_csv
    from numpy import select

    grep = '       J      i        Energy/cm  State   v  lambda spin   sigma   omega  parity\n'
    
    with open(output_path) as file:
        lines = file.readlines()
    
    starts_of_eigenvalues = [num for num, l in enumerate(lines) if grep in l]

    eigenvals = []
    for s in starts_of_eigenvalues:
        s+=1
        l = ""
        component = []
        while "ZPE" not in l and grep not in l:
            l = lines[s].replace("||","")
            
            component.append(l)
            s+=1
        eigenvals = eigenvals+component[:-1]

    eigenvals = [i for i in eigenvals if i not in ["\n"]]

    with open("get_eigenvalues_temp_file.txt", "w") as file:
        file.writelines(eigenvals)    

    eigenvalues = read_csv("./get_eigenvalues_temp_file.txt", sep = "\s+", names = ["J","NN","E","State","v","Lambda","Spin","Sigma","Omega","tau","Manifold"])

    running = sp.Popen("rm get_eigenvalues_temp_file.txt", shell = True, stdout=sp.PIPE)
    running.communicate()

    eigenvalues = eigenvalues[["NN","E","J","tau","Manifold","v","Lambda","Sigma","Omega"]]
    eigenvalues["E"]-=eigenvalues["E"].min()

    tau_cond = [eigenvalues["tau"] == "-",eigenvalues["tau"] == "+"]
    tau_vals = [-1, 1]
    eigenvalues["tau"] = select(tau_cond, tau_vals)
    return eigenvalues

def match_eigenvalues_and_states(eigenvalues, states):
    merged = eigenvalues.merge(states, on = ["J","tau","Manifold","v","Lambda","Sigma","Omega"], how = "inner", suffixes = ["_eigenval","_states"])
    merged["E_diff"] = abs(merged["E_eigenval"]-merged["E_states"])
    max_deviation = merged["E_diff"].max()
    if max_deviation > 1e-5:
        print(f"WARNING: Your energies don't match up, you have a maximum deviation of {max_deviation} please check your eigenvalues and states file")
        merged = eigenvalues.merge(states, on = ["J","tau","Manifold","v","Lambda","Sigma","Omega"], how = "outer", suffixes = ["_eigenval","_states"])
        return merged
    
    merged = merged[["NN_eigenval","E_states","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]]
    merged = merged.rename(columns = {"NN_eigenval":"NN","E_states":"E"})
    return merged

def reduced_mass(m1, m2):
    mu = (m1*m2)/(m1+m2)
    return mu

def rotational_factor(r, m1, m2, units):
    '''
    Will return the rotational constant B of a diatomic made of atoms with masses m1, m2 for a given nuclear separation r.

    If units = "cm-1":
        B = hbar/4*pi*c * 1/I
            where:
            
            hbar is Planck's constant in erg seconds
            c is the speed of light in cm/s
            I is mu*r^2
            mu is the reduced mass in units of grams
            r  is the nuclear separation in units of cm
    If units = "hartree":
        B = hbar/mu*r^2 
        where:
            hbar = 1
            mu is the reduced mass in units of electron masses
            r  is the nuclear separation in units of bohr

    Inputs:
        r = nuclear separation in Angstroms
        m1 = mass of atom 1 in atomic mass units  (Daltons)
        m2 = mass of atom 2 in atomic mass units  (Daltons)
        units = units of B (cm-1 or Hartree)
    '''
    from numpy import pi

    hbar_erg_s      = 6.62606957e-27/(2*pi) # erg seconds

    amu_to_g  = 1.660538921000E-24 #grams
    amu_to_me = 1822.8884861185961 #electron mass

    speed_of_light = 29979245800.0000 #cm/second

    angstrom_to_bohr = 1.88972612457 #bohr
    angstrom_to_cm   = 1e-8 #cm

    if units == "hartree":
        m1 *= amu_to_me 
        m2 *= amu_to_me
        r  *= angstrom_to_bohr
        I   = reduced_mass(m1, m2)*r**2

        B = 1/(2*I)
        return B
    elif units in ["cm", "cm-1", "wavenumber"]:
        m1 *= amu_to_g
        m2 *= amu_to_g
        r  *= angstrom_to_cm   
        I   = reduced_mass(m1, m2)*r**2
        
        B   = hbar_erg_s/(4*pi*speed_of_light)*(1/I)

        return B

def effective_potential(r,V,m1,m2,J):
    from numpy import array

    r = array([*r])
    V = array([*V])

    B = rotational_factor(r, m1, m2, "wavenumber")

    E_rot = J*(J+1)*B

    V += E_rot

    return V