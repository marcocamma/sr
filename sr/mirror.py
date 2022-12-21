import numpy as np
import copy
from . import undulator
from datastorage import DataStorage as ds
import xrt.backends.raycing.materials as rm
from xrt.backends.raycing.physconsts import AVOGADRO
from functools import lru_cache

_bulk_densities = dict(Pt=21.45, Pd=12.02, Si=2.329)
_density_factor = dict(Pt=0.9, Pd=0.9, Si=1)

R_ELECTRON = 2.818e-13 # in cm
from xrt.backends.raycing.physconsts import PI

def coddington_meridional(p, q, theta):
    """ return radius of curvature """
    f = p * q / (p + q)
    R = 2 * f / np.sin(theta)
    return R


def coddington_meridional_R(f, theta):
    """ return radius of curvature """
    R = 2 * f / np.sin(theta)
    return R

def coddington_meridional_f(R, theta):
    """ return focal length """
    f = R/2*np.sin(theta)
    return f



def coddington_sagittal(p, q, theta):
    """ return radius of curvature """
    f = p * q / (p + q)
    R = 2 * f * np.sin(theta)
    return R

def get_material(atom="Pt", rho=None):
    """ 
    use rho='bulk' for bulk density
    use rho=None for default density of coating (90% of bulk for Pt)
    """
    if isinstance(rho,str) and rho == "bulk":
        rho = _bulk_densities[atom]
    elif rho is None:
        rho = _bulk_densities[atom] * _density_factor[atom]
    return rm.Material(atom, rho=rho)


def mirror_reflectivity(E, material="Si", angle=2e-3, rho=None):
    m = Mirror(material,rho=rho)
    return m.reflectivity(E,angle)


class Mirror:
    def __init__(self, material="Si", rho=None):
        """ 
        use rho='bulk' for bulk density
        use rho=None for default density of coating (90% of bulk for Pt)
        """
        self.material_name = material
        self.material = get_material(material,rho=rho)
        self.rho = self.material.rho
        atom = self.material.elements[0]
        edens_elem = atom.Z*self.rho/atom.mass*AVOGADRO
        self.edens = edens_elem


    @lru_cache(maxsize=4096*10)
    def _reflectivity_float(self, E, angle):
        field_reflectivity = self.material.get_amplitude(E * 1e3, np.sin(angle))[0]
        intensity_reflectivity = np.abs(field_reflectivity) ** 2
        return intensity_reflectivity

    def reflectivity(self, E, angle):
        """ single bounce reflectivity """
        try:
            E = float(E)
            return self._reflectivity_float(E, angle)
        except TypeError:
            ret = np.empty_like(E)
            for i,e in enumerate(E):
                ret[i]=self._reflectivity_float(e,angle)
            return ret

    def reflectivity2(self, E, angle):
        """ reflectivity for double bounce """
        return self.reflectivity(E, angle) ** 2

    def critical_angle(self,energy):
        """ energy in keV, returns angle in rad """
        n = self.material.get_refractive_index(energy*1e3)
        delta = 1-n.real
        return np.sqrt(2*delta)

    def cutoff_energy(self,angle):
        """ angle is in radians, returns cutoff_energy in keV """
        def f(E):
            return self.critical_angle(E)-angle
        from scipy.optimize import bisect
        try:
            ret = bisect(f,4,50,xtol=1e-4,rtol=1e-4)
        except ValueError:
            print("No cutoff energy found in the 4,50 keV range")
            ret = np.nan
        return float(ret)
    
    def photon_flux_after_mirror(self,photon_flux_before,angle,nbounces=2):
        """ does not take into account finite size of mirrors """
        photon_flux_before = copy.deepcopy(photon_flux_before)
        r = self.reflectivity(photon_flux_before.energy,angle)**nbounces
        photon_flux_before.spectral_photon_flux_density *= r[:,np.newaxis,np.newaxis]
        photon_flux_before.spectral_photon_flux *= r
        after = undulator._photon_flux_density_helper(photon_flux_before)
        return after

    def __str__(self):
        return f"{self.material_name} Mirror, density {self.rho} gr/cm3"

    def __repr__(self):
        return self.__str__()

id18_mirrors = [
    Mirror(material="Si", rho=_bulk_densities["Si"] * _density_factor["Si"]),
    Mirror(material="Pd", rho=_bulk_densities["Pd"] * _density_factor["Pd"]),
    Mirror(material="Pt", rho=_bulk_densities["Pt"] * _density_factor["Pt"]),
]


def find_mirror(
    E,
    mirrors=id18_mirrors,
    reflectivity=0.8,
    angles=np.linspace(2.5e-3, 4.5e-3, 201),
    reduction_factor=0.95,
    max_3E_reflectivity=1e-5,
    verbose=True,
):
    """ mirrors should be ordered from lighter to heavier element """
    for mirror in mirrors:
        r = np.empty_like(angles)
        r3 = np.empty_like(angles)
        for i, a in enumerate(angles):
            r[i] = mirror.reflectivity2(E, a)
            r3[i] = mirror.reflectivity2(E * 3, a)
        idx1 = r > reflectivity
        idx3 = r3 < max_3E_reflectivity
        if (idx1 & idx3).sum() > 0:
            idx = np.argwhere(idx1 & idx3).ravel()[-1]  # take biggest angle
            return ds(
                surface=mirror.material_name,
                angle=angles[idx],
                two_bounces_reflectivity=r[idx],
                two_bounces_reflectivity_3E=r3[idx],
            )
    if verbose:
        print(
            f"Could not find any mirror with reflectivity@E>{reflectivity:.2f} and reflectivity@3E < {max_3E_reflectivity:.3e}"
            )
        print(
            f"For {mirrors[-1].material_name}, max of reflectivity@E {r.max():.3e} and min reflectivity@3E {r3.min():.3e}"
            )
        print(f"will try with reflectivity = {reflectivity*reduction_factor:.2f}")
    if reflectivity < 0.01:
        # start new search with higher max_3E_reflectivity
        return find_mirror(
            E,
            mirrors=mirrors,
            reflectivity=1,
            angles=angles,
            max_3E_reflectivity=max_3E_reflectivity*10,
            verbose=verbose,
        )
    else:
        return find_mirror(
            E,
            mirrors=mirrors,
            reflectivity=reflectivity * reduction_factor,
            angles=angles,
            max_3E_reflectivity=max_3E_reflectivity,
            verbose=verbose,
        )
