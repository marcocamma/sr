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

