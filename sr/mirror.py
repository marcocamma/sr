import numpy as np
import xrt.backends.raycing.materials as rm

_bulk_densities = dict( Pt = 21.45, Pd = 12.02, Si=2.329 )
_density_factor = dict( Pt = 0.9,   Pd = 0.9,   Si=1     )


def coddington_meridional(p,q,theta):
    """ return radius of curvature """
    f = p*q/(p+q)
    R = 2*f/np.sin(theta)
    return R

def coddington_sagittal(p,q,theta):
    """ return radius of curvature """
    f = p*q/(p+q)
    R = 2*f*np.sin(theta)
    return R

def get_material(atom="Pt",rho=None):
    if rho is None: rho=_bulk_densities[atom]*_density_factor[atom]
    return rm.Material(atom,rho=rho)

def mirror_reflectivity(E,material="Si",angle=2e-3,rho=None):
    m = get_material(material,rho=rho)
    field_reflectivity = m.get_amplitude(E*1e3, np.sin(angle))[0]
    intensity_reflectivity = np.abs(field_reflectivity)**2
    return intensity_reflectivity

class Mirror:
    def __init__(self,material="Si",rho=None):
        self.material = material
        self.rho = rho

    def reflectivity(self,E,angle):
        return mirror_reflectivity(E,material=self.material,rho=self.rho,angle=angle)

    def reflectivity2(self,E,angle):
        """ reflectivity for double bounce """
        return self.reflectivity(E,angle)**2

def find_mirror(E,reflectivity=0.9,verbose=True):
    angles = np.linspace(2e-3,5e-3,301)
    mirrors = [Mirror(material="Si"),
               Mirror(material="Pd"),
               Mirror(material="Pt")
               ]
    for mirror in mirrors:
        r = [mirror.reflectivity2(E,a) for a in angles]
        r = np.asarray(r)
        idx = np.argwhere(r>reflectivity).ravel()
        if len(idx) > 0:
            return mirror.material,angles[idx[-1]]

