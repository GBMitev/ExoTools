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

class Units:
    def wavenumber_to_nm(wavenumber):
        return 10e6/wavenumber
    
    def nm_to_wavenumber(nm):
        return 10e6/nm

    def wavenumber_to_mhz(wavenumber):
        return 29979.2458*wavenumber

    def mhz_to_wavenumber(mhz):
        return mhz/29979.2458

    def wavenumber_to_energy(wavenumber, end_units = "hartree"):
        if end_units == "hartree":
            return 0.0000046*wavenumber
        elif end_units == "ev":
            return 0.00012*wavenumber

    def energy_to_wavenumber(value, starting_units="hartree"):
        if starting_units == "hartree":
            return value/0.0000046
        elif starting_units == "ev":
            return value/0.00012

    def bohr_to_angstrom(bohr):
        return 0.529177249*bohr

    def angstrom_to_bohr(angstrom):
        return angstrom/0.529177249
    

    
    
