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
        return value*219474.63
    elif starting_units == "ev":
        return value*8065.73

def bohr_to_angstrom(bohr):
    return 0.529177249*bohr

def angstrom_to_bohr(angstrom):
    return angstrom/0.529177249

def debye_to_au(debye):
    return 0.393456*debye

def au_to_debye(au):
    return au/0.393456