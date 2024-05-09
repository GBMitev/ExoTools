# %%
def states_cols():
    return ["NN","E","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]
    
def states_df(path, manifold_to_state = None, extra_cols=None, end_cols=None):
    from pandas import read_csv
    from numpy import select
    
    if extra_cols is not None:
        Unc = "Unc" if "Unc" in extra_cols else "Empty"
        Lifetime = "Lifetime" if "Lifetime" in extra_cols else "Empty"
        Lande = "Lande" if "Lande" in extra_cols else "Empty"

        extra_cols = [i for i in [Unc, Lifetime, Lande] if i != "Empty"]
    
    column_names = ["NN","E","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]
    
    if extra_cols is not None:
        column_names = column_names[0:4]+extra_cols+column_names[4:]

    if end_cols is not None:
        column_names = column_names + end_cols
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

def get_full_trans(states, trans, columns = None):

    trans  = trans  if type(trans)  != str else trans_df(trans)
    states = states if type(states) != str else states_df(states)

    full = trans.merge(states, left_on = "nf", right_on = "NN").drop(columns=["NN"])
    full = full.merge(states, left_on = "ni", right_on = "NN", suffixes = ["_upper","_lower"]).drop(columns = ["NN"])

    return full if columns is None else full[columns]
    
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
    from tqdm import tqdm
    lines = []
    for nf, ni, A, nu in tqdm(trans.itertuples(index = False)):
        line = \
            f"{nf:12d}"+" "+\
                f"{ni:12d}"+" "+\
                f"{A:11.4e}".upper()+" "+\
                f"{nu:19.6f}"+"\n"
        
        lines.append(line)

    with open(fname, "w") as file:  
        file.writelines(lines)  
    return "done"

def reset_state_index(pruned,Number_column_name= "N"):
    pruned       = pruned.rename(columns={Number_column_name:"N_old"})
    pruned["N"] = pruned.index +1
    pruned       = pruned[["N","N_old","E","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]]
    return pruned

def merge_stf(upper,lower, trans, counting_num, full = False, counting_num_mapping=None):
    cols = [counting_num, "J","tau", "Manifold"] if counting_num_mapping is None else [counting_num, counting_num_mapping, "J","tau", "Manifold"]
    
    trans = trans.merge(upper[cols],left_on = "nf", right_on = counting_num, how = "inner")
    trans = trans.merge(lower[cols],left_on = "ni", right_on = counting_num, how = "inner", suffixes = ["_f","_i"])
    
    counting_num = counting_num if counting_num_mapping is None else counting_num_mapping
    
    N_f = counting_num+"_f"
    N_i = counting_num+"_i"

    trans = trans[[N_f,N_i,"J_f","J_i","tau_f","tau_i","Manifold_f","Manifold_i","A","nu"]].rename(columns = {N_f:"nf", N_i:"ni"})
    
    return trans if full == True else trans[["nf", "ni", "A","nu"]]

def bound_to_bound_stf(bound_states_file, trans, Number_column_name = "N", rigorous = False):
    bound_states_file = reset_state_index(bound_states_file, Number_column_name)

    bound_trans = merge_stf(bound_states_file,bound_states_file,trans,"N_old",full = True, counting_num_mapping="N")

    bound_states = bound_states_file[["N","E","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]]

    return bound_states, bound_trans


def write_states_file(states, fname):
    lines = []
    for NN, E, gns, J,tau, ef, Manifold, v, Lambda, Sigma, Omega in states[["NN","E","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]].itertuples(index=False):
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
            f"{v:3d}"+" "+\
            f"{Lambda:2d}"+" "*4+\
            f"{Sigma:>4.1f}"+" "*4+\
            f"{Omega:>4.1f}"+"\n"

        lines.append(line)
    
    with open(fname, "w") as file:
        file.writelines(lines)
    
    return fname+"has been written :)"
    
def write_full_states_file(states, fname):
    lines = []

    for NN, E, gns, J,Unc, Lifetime, Lande, tau, ef, Manifold, v, Lambda, Sigma, Omega,Type, E_Calc in states[["NN","E","gns","J","Unc","Lifetime","Lande","tau","e/f","Manifold","v","Lambda","Sigma","Omega","Type","E_Calc"]].itertuples(index=False):
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
            f"{Unc:{12}.6f}"[:12]+" "+\
            f"{Lifetime:12.4e}".upper()+" "+\
            f"{Lande:{10}.6f}"[:10]+" "+\
            f"{tau}"+" "+\
            f"{ef}"+" "+\
            f"{Manifold:<{10}}"+" "+\
            f"{v:3d}"+" "+\
            f"{Lambda:2d}"+" "*4+\
            f"{Sigma:>4.1f}"+" "*4+\
            f"{Omega:>4.1f}"+" "+\
            f"{Type}"[0:2]+" "+\
            f"{E_Calc:{12}.6f}"[:12]+"\n"

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

def integrate_all_wavefunctions_vectorize(wf, R, thresh_delta_r, **kwargs):
    from pandarallel import pandarallel, core
    from numpy import vectorize
    cores = kwargs.get("cores",core.NB_PHYSICAL_CORES)
    pandarallel.initialize(progress_bar=True, nb_workers=cores)

    integrals = wf.groupby(["J","tau","NN"], as_index=False).size()[["J","tau","NN"]]
    integrals["Integ"] = integrals.parallel_apply(lambda x: vectorize(integrate_wavefunction(wf, R, thresh_delta_r, x["J"], x["tau"], x["NN"])), axis = 1)
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
        m1_me   = m1*amu_to_me 
        m2_me   = m2*amu_to_me
        r_bohr  = r*angstrom_to_bohr
        I       = reduced_mass(m1_me, m2_me)*r_bohr**2

        B = 1/(2*I)
        return B
    elif units in ["cm", "cm-1", "wavenumber"]:
        m1_g = m1*amu_to_g
        m2_g = m2*amu_to_g
        r_cm = r*angstrom_to_cm   
        I     = reduced_mass(m1_g, m2_g)*r_cm**2
        
        B   = hbar_erg_s/(4*pi*speed_of_light)*(1/I)
        return B

def effective_potential(r,V,m1,m2,J):
    from numpy import array

    r = array([*r])
    V = array([*V])

    B = rotational_factor(r, m1, m2, "wavenumber")

    E_rot = J*(J+1)*B

    V_eff = V+E_rot

    return V_eff

def effective_potential_features(r, V_eff):
    from scipy.signal import find_peaks
    from numpy import nan, ndarray

    idx_min = find_peaks(-V_eff)[0]
    idx_max = find_peaks(V_eff)[0]

    min_type = "rep" if len(idx_min) == 0 else "bnd"
    max_type = "asm" if len(idx_max) == 0 else "bar"
    
    pec_type = min_type + "_" +  max_type

    idx_De = {"rep_asm":None,"bnd_asm":-1,"bnd_bar":idx_max}

    idx_Re = {"rep_asm":None,"bnd_asm":idx_min,"bnd_bar":idx_min}

    idx_Re = idx_Re[pec_type] if idx_Re[pec_type] is not None else None

    if idx_Re is not None:
        Re = r[idx_Re]
        Te = V_eff[idx_Re]
    else:
        Re = nan
        Te = nan

    idx_De = idx_De[pec_type] if idx_De[pec_type] is not None else None

    if idx_De is not None:
        De = V_eff[idx_De]
        RDe = r[idx_De]
    else:
        De = nan
        RDe = nan
    
    Re = Re if type(Re) != ndarray or Re is nan else Re[0]
    Te = Te if type(Te) != ndarray or Te is nan else Te[0]
    RDe = RDe if type(RDe) != ndarray or RDe is nan else RDe[0]
    De = De if type(De) != ndarray or De is nan else De[0]
    return Re, Te, RDe, De

