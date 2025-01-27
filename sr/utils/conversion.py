# -*- coding: utf-8 -*-

from scipy import constants

# global variable
_keV_ang = constants.Planck * constants.speed_of_light / constants.eV * 1e10 / 1e3

def energy_to_wavelength(E_keV):
    """ given the energy in keV return the wavelength in Ang """
    return _keV_ang / E_keV


def wavelength_to_energy(lambda_ang):
    """ given the wavelength in Ang returns the energy in keV """
    return _keV_ang / lambda_ang


def current_to_photons_per_s(current_amp,E_keV):
    bandgap = 3.6
    E_eV = E_keV*1e3
    n_e_per_photon = E_eV/bandgap
    charge_per_photon = n_e_per_photon*constants.e
    photons_per_sec = current_amp/charge_per_photon
    return photons_per_sec

def current_to_photons_per_s(current_amp,E_keV,thickness=500e-6,material="Si"):
    import xraydb
    return xraydb.ionchamber_fluxes(
            gas=material,
            length=thickness*100, # needs cm
            energy=E_keV*1000, # needs eV
            volts=current_amp,
            sensitivity=1
            )

