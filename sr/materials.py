"""
In this module:
    energies are in keV,
    densities in gr/cm3
    transmissions is calculated using PHOTOELECTRIC effect ONLY
"""

import numpy as np
import copy

from functools import lru_cache

try:
    from .utils.conversion import energy_to_wavelength
except ImportError:

    def energy_to_wavelength(energy):
        return 12.398 / energy


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
    GaAs=5.32,
    CdTe=5.85,
    Si3N4=3.17,
    silicon_nitride=3.17,
)

CHEMICAL_FORMULAS = dict(
    steel="Fe85Cr11C4",
    H2O="H2O",
    water="H2O",
    diamond="C",
    kapton="C22H10N2O5",
    mylar="C10H8O4",
    ambient_air="N1.562O0.42C0.0003Ar0.0094",
    silicon_nitride="Si3N4",
)

AVOGADRO = 6.02214199e23


def list_materials():
    print(f"{'name':15s} {'formula':25s} {'density':15s}")
    for name, formula in CHEMICAL_FORMULAS.items():
        density = DENSITIES[name]
        print(f"{name:15s} {formula:25s} {density:15.3f} gr/cm3")


def get_default_density(element):
    if element not in DENSITIES:
        raise ValueError("No default density for", element)
    return DENSITIES[element]


def get_formula(element):
    if element in CHEMICAL_FORMULAS:
        element = CHEMICAL_FORMULAS[element]
    return element


def get_material(element, density=None):
    # density has to go first else names are converted into formulas
    if density is None:
        density = get_default_density(element)
    element = get_formula(element)
    return element, density


def get_atomic_densities(element, density=None):
    """returns atoms in cm^3 for each atomic species (in dictionary)"""
    molecular_density = get_number_density(element, density=density)
    temp = xdb.materials.chemparse(element)
    ret = dict([(el, n * molecular_density) for el, n in temp.items()])
    return ret


def get_molecular_mass(element):
    molecular_mass = 0
    temp = xdb.materials.chemparse(element)
    for el, n in temp.items():
        molecular_mass += n * xdb.atomic_mass(el)
    return molecular_mass


def get_number_density(element, density=None):
    """returns number of molecules per cm^3"""
    if density is None:
        if element not in DENSITIES:
            raise ValueError("No default density for", element)
        else:
            density = DENSITIES[element]
    molecular_mass = get_molecular_mass(element)
    num_density = density / molecular_mass * AVOGADRO
    return num_density


def get_molecules_per_cm3(element, density=None):
    return get_number_density(element, density=density)


def get_molar_density(element, density=None):
    """return molar density (= number of moles per L)"""
    n = get_number_density(element, density=density)
    number_of_moles_per_cm3 = n / AVOGADRO
    number_of_moles_per_L = number_of_moles_per_cm3 * 1e3
    return number_of_moles_per_L


def get_density(element, molar_density=1e-3):
    """return density (in gr/cm3)"""
    mass_per_mole = get_molecular_mass(element)
    num_moles_per_cm3 = molar_density * 1e-3  # 1cm3 = 1e-3L
    return mass_per_mole * num_moles_per_cm3


def get_air_density(T=293, P=1e5):
    density_kg_m3 = P * 0.0289652 / (8.31446 * T)
    return density_kg_m3


@lru_cache(maxsize=1024)
def get_refractive_index(material, density=None, energy=10):
    material, density = get_material(material, density)
    if isinstance(energy, (float, int, np.ndarray)):
        energy = energy * 1e3
        ret = material.get_refractive_index(energy)
    else:
        # for tuples
        ret = [material.get_refractive_index(e * 1e3) for e in energy]
        ret = np.asarray(ret)
    return ret


@lru_cache(maxsize=1024)
def get_delta_beta(material, density=None, energy=10):
    material, density = get_material(material, density)
    delta, beta, attlen = xdb.xray_delta_beta(material, density, energy * 1e3)
    return delta, beta


def _attenuation_length_nocache(material, density=None, energy=10, kind="total"):
    """
    Calculates the attenuation length for a compound [in m]
                Transmission is then exp(-thickness/attenuation_length)
    kind can be "total" or "photo" (for photoelectric)
    """
    if isinstance(energy, (tuple, list)):
        energy = [e * 1e3 for e in energy]
    else:
        energy = energy * 1e3
    material, density = get_material(material, density)
    mu = xdb.material_mu(material, energy, density=density, kind=kind)
    mu = mu * 1e2  # (from cm-1 to m-1)
    return 1 / mu


@lru_cache(maxsize=1024)
def attenuation_length(compound, density=None, energy=None, kind="total"):
    """
    Calculates the attenuation length for a compound [in m]
                Transmisison is then exp(-thickness/attenuation_length)
    """
    return _attenuation_length_nocache(
        compound, density=density, energy=energy, kind=kind
    )


def transmission(
    material="Si", thickness=1e-3, energy=10, density=None, cross_section_kind="total"
):
    """kind can be total|photo"""
    w = Wafer(material, thickness=thickness, density=density)
    return w.calc_transmission(energy, cross_section_kind=cross_section_kind)


def absorption(
    material="Si", thickness=1e-3, energy=10, density=None, cross_section_kind="photo"
):
    """kind can be total|photo"""
    w = Wafer(material, thickness=thickness, density=density)
    return 1 - w.calc_transmission(energy, cross_section_kind=cross_section_kind)


class Wafer:
    def __init__(self, name="Si", thickness=10e-6, density=None):
        self.name = name
        formula, density = get_material(name, density)
        self.formula = formula
        self.density = float(density)
        self.thickness = thickness

    def get_att_len(self, E, kind="total"):
        """get the attenuation length (in meter) of material
        E in keV"""
        try:
            return attenuation_length(
                self.formula, density=self.density, energy=E, kind=kind
            )
        except TypeError:  # when E is unhashable (ndarray)
            return attenuation_length(
                self.formula,
                density=self.density,
                energy=tuple(E),
                kind=kind,
            )

    def calc_transmission(self, E, cross_section_kind="total"):
        att_len = self.get_att_len(E, kind=cross_section_kind)
        return np.exp(-self.thickness / att_len)

    def calc_absorption(self, E, cross_section_kind="photo"):
        att_len = self.get_att_len(E, kind=cross_section_kind)
        return 1 - np.exp(-self.thickness / att_len)

    def photon_flux_after_wafer(self, photon_flux_before, cross_section_kind="total"):
        """does not take into account finite size of mirrors"""
        from . import undulator

        photon_flux_before = copy.deepcopy(photon_flux_before)
        t = self.calc_transmission(
            photon_flux_before.energy, cross_section_kind=cross_section_kind
        )
        photon_flux_before.spectral_photon_flux_density *= t[:, np.newaxis, np.newaxis]
        photon_flux_before.spectral_photon_flux *= t
        after = undulator._photon_flux_density_helper(photon_flux_before)
        return after

    def material_str(self, with_formula=False, with_density=False, with_thickness=True):
        s = self.name
        if with_formula and self.name != self.formula:
            s += f"[{self.formula}]"
        if with_density:
            s += f", {self.density:.2e} gr/cm³"
        if with_thickness:
            if self.thickness >= 1e-3:
                s += f", {self.thickness*1e3:.2f}mm"
            else:
                s += f", {self.thickness*1e6:.2f}μm"
        return s

    def dose(self, E, f=1e12, exp_time=1, beamsize=50e-6, verbose=False):
        """returns absorbed dose in Gray (Gy), f is the photon flux (photons/s)"""
        photon_energy = 1.60218e-19 * E * 1e3
        beam_power = photon_energy * f
        A = self.calc_absorption(E)
        absorbed_energy = beam_power * exp_time * A
        density_SI = self.density * 1e3
        mass = beamsize * beamsize * self.thickness * density_SI
        if verbose:
            print(f"{beam_power=:.3e} W ,{absorbed_energy=:-4e} J ,{mass=:.4e} Kg")
        return absorbed_energy / mass

    def __repr__(self):
        return self.material_str(
            with_formula=True, with_density=True, with_thickness=True
        )

    def __str__(self):
        return self.material_str(
            with_formula=False, with_density=False, with_thickness=True
        )


def examples():
    Si = Wafer("Si", thickness=10e-6)
    E = np.arange(3, 50, 0.1)
    Si.calc_transmission(E)
