"""
In this module:
    energies are in keV,
    densities in gr/cm3
    transmissions is calculated using PHOTOELECTRIC effect ONLY
"""
import numpy as np
import itertools
import collections
import copy

from functools import lru_cache
from .utils.conversion import energy_to_wavelength
from . import undulator

import xraydb as xdb


DENSITIES = dict(
    H=0.071,
    He=0.122,
    Li=0.534,
    Be=1.848,
    B=2.340,
    C=2.100,
    diamond=3.51,
    N=0.808,
    O=1.140,
    F=1.500,
    Ne=1.207,
    Na=0.971,
    Mg=1.738,
    Al=2.699,
    Si=2.330,
    P=1.820,
    S=2.070,
    Cl=1.560,
    Ar=1.400,
    K=0.862,
    Ca=1.550,
    Sc=2.989,
    Ti=4.540,
    V=6.110,
    Cr=7.190,
    Mn=7.330,
    Fe=7.874,
    Co=8.900,
    Ni=8.902,
    Cu=8.960,
    Zn=7.133,
    Ga=5.904,
    Ge=5.323,
    As=5.730,
    Se=4.790,
    Br=3.120,
    Kr=2.160,
    Rb=1.532,
    Sr=2.540,
    Y=4.469,
    Zr=6.506,
    Nb=8.570,
    Mo=10.220,
    Tc=11.500,
    Ru=12.410,
    Rh=12.410,
    Pd=12.020,
    Ag=10.500,
    Cd=8.650,
    In=7.310,
    Sn=7.310,
    Sb=6.691,
    Te=6.240,
    I=4.930,
    Xe=3.520,
    Cs=1.873,
    Ba=3.500,
    La=6.145,
    Ce=6.770,
    Pr=6.773,
    Nd=7.008,
    Pm=7.264,
    Sm=7.520,
    Eu=5.244,
    Gd=7.901,
    Tb=8.230,
    Dy=8.551,
    Ho=8.795,
    Er=9.066,
    Tm=9.321,
    Yb=6.966,
    Lu=9.841,
    Hf=13.310,
    Ta=16.654,
    W=19.300,
    Re=21.020,
    Os=22.570,
    Ir=22.420,
    Pt=21.450,
    Au=19.300,
    Hg=13.546,
    Tl=11.850,
    Pb=11.350,
    Bi=9.747,
    Po=9.320,
    Th=11.720,
    Pa=15.370,
    U=18.950,
    Np=20.250,
    Pu=19.840,
    Am=13.670,
    Cm=13.510,
    Bk=14.000,
    H2O=1,
    water=1,
    steel=7.85,
    kapton=1.43,
    mylar=1.4,
    ambient_air=1.19e-3,
)

CHEMICAL_FORMULAS = dict(
    steel="Fe85Cr11C4",
    h2o="H2O",
    water="H2O",
    diamond="C",
    kapton="C22H10N2O5",
    mylar="C10H8O4",
    ambient_air="N1862O418Ar9",
)

AVOGADRO = 6.02214199e23


def get_density(element):
    if element not in DENSITIES:
        raise ValueError("No default density for", element)
    return DENSITIES[element]


def get_number_density(element, density=None):
    """ returns atoms in cm^3 """

    if density is None:
        if element not in DENSITIES:
            raise ValueError("No default density for", element)
        else:
            density = DENSITIES[element]
    atomic_mass = xdb.atomic_mass(elem)
    num_density = density / atomic_mass * AVOGADRO


def get_air_density(T=293, P=1e5):
    density_kg_m3 = P * 0.0289652 / (8.31446 * T)
    density_g_cm3 = density_kg_m3 * 1e-3
    return density_kg_m3


@lru_cache(maxsize=1024)
def get_refractive_index(material, density=None, energy=10):
    m = get_material(material, density=density)
    if isinstance(energy, (float, int, np.ndarray)):
        energy = energy * 1e3
        ret = m.get_refractive_index(energy)
    else:
        # for tuples
        ret = [m.get_refractive_index(e * 1e3) for e in energy]
        ret = np.asarray(ret)
    return ret


@lru_cache(maxsize=1024)
def get_delta_beta(material, density=None, energy=10):
    if density is None:
        density = DENSITIES[material]
    delta, beta, attlen = xdb.xray_delta_beta(material, density, energy * 1e3)
    return delta, beta


def _attenuation_length_nocache(material, density=None, energy=10, kind="total"):
    """
    Calculates the attenuation length for a compound [in m]
                Transmission is then exp(-thickness/attenuation_length)
    kind can be "total" or "photo" (for photoelectric)
    """
    if density is None:
        density = DENSITIES[material]
    mu = xdb.material_mu(material, energy * 1e3, density=density, kind=kind)
    mu = mu * 1e2  # (from cm-1 to m-1)
    return 1 / mu


@lru_cache(maxsize=1024)
def attenuation_length(compound, density=None, energy=None, kind="total"):
    """
    Calculates the attenuation length for a compound [in m]
                Transmisison is then exp(-thickness/attenuation_length)
    """
    return _attenuation_length_nocache(compound, density=density, energy=energy, kind=kind)


def transmission(material="Si", thickness=1e-3, energy=10, density=None, cross_section_kind="total"):
    """ kind can be total|photo """
    w = Wafer(material=material, thickness=thickness, density=density)
    return w.calc_transmission(energy, cross_section_kind=cross_section_kind)

def absorption(material="Si", thickness=1e-3, energy=10, density=None, cross_section_kind="photo"):
    """ kind can be total|photo """
    w = Wafer(material=material, thickness=thickness, density=density)
    return 1-w.calc_transmission(energy, cross_section_kind=cross_section_kind)



class Wafer:
    def __init__(self, material="Si", thickness=10e-6, density=None):

        self.material = material
        self.thickness = thickness
        if density is None:
            density = get_density(material)
        self.density = density

    def get_att_len(self, E, kind="total"):
        """ get the attenuation length (in meter) of material
            E in keV"""
        try:
            return attenuation_length(
                self.material, density=self.density, energy=E, kind=kind
            )
        except TypeError:  # when E is unhashable (ndarray)
            return attenuation_length(
                self.material, density=self.density, energy=tuple(E), kind=kind,
            )

    def calc_transmission(self, E, cross_section_kind="total"):
        att_len = self.get_att_len(E, kind=cross_section_kind)
        return np.exp(-self.thickness / att_len)

    def calc_absorption(self, E, cross_section_kind="photo"):
        att_len = self.get_att_len(E, kind=cross_section_kind)
        return 1 - np.exp(-self.thickness / att_len)

    def photon_flux_after_wafer(self, photon_flux_before, cross_section_kind="total"):
        """ does not take into account finite size of mirrors """
        photon_flux_before = copy.deepcopy(photon_flux_before)
        t = self.calc_transmission(
            photon_flux_before.energy, cross_section_kind=cross_section_kind
        )
        photon_flux_before.spectral_photon_flux_density *= t[:, np.newaxis, np.newaxis]
        photon_flux_before.spectral_photon_flux *= t
        after = undulator._photon_flux_density_helper(photon_flux_before)
        return after

    def __repr__(self):
        return f"filter {self.material:3s}, thickness {self.thickness*1e6} um"

    def __str__(self):
        if self.thickness >= 1e-3:
            return f"{self.material}({self.thickness*1e3:.2f}mm)"
        else:
            return f"{self.material}({self.thickness*1e6:.1f}μm)"


def examples():
    Si = Wafer("Si", thickness=10e-6)
    E = np.arange(3, 50, 0.1)
    Si.calc_transmission(E)
