import numpy as np
from matplotlib import pyplot as plt

import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes

class Monochromator:
    def __init__(self,hkl=[1,1,1],tempC=-140):
        if isinstance(hkl,str): hkl = [int(i) for i in hkl]
        cryst_name = f"Si{hkl[0]}{hkl[1]}{hkl[2]}"
        self.name = cryst_name
        self.temperature = 273.15+tempC
        self.cryst = rmats.CrystalSi(hkl=hkl,name=cryst_name,tK=self.temperature)
        self.dcm = roes.DCM(
            name=r"DCM",
            material=self.cryst,
            material2=self.cryst,
            cryst2perpTransl=4)

    def bragg_angle_deg(self,energy_keV=10,order=1):
        energy_eV = energy_keV*1e3
        bragg_angle = self.cryst.get_Bragg_angle(energy_eV,order=order)
        return np.rad2deg(bragg_angle)

    def darwin_width(self,energy_keV=10,polarization="s"):
        energy_eV = energy_keV*1e3
        dw = self.cryst.get_Darwin_width(energy_eV,polarization=polarization)
        return dw

    def resolution(self,energy_keV=10,order=1,beam_div_urad=10,polarization="s"):
        energy_eV = energy_keV*1e3
        bragg_angle = self.cryst.get_Bragg_angle(energy_eV,order=order)
        beam_div = beam_div_urad*1e-6
        dispersion = order/(2*self.cryst.d*np.cos(bragg_angle))
        
        dw = self.cryst.get_Darwin_width(energy_eV,polarization=polarization)


        wavelength = 12.398/energy_keV

        resolution = wavelength*dispersion/(dw+beam_div)
        return resolution

    def reflectivity(self,energy_keV=10,angle='auto',polarization="s",two_bounces=True):
        # one would have to search because of refraction it will not be at nominal bragg angle
        print("TODO")



    def coherence_length(self,energy_keV=10,order=1,beam_div_urad=10,polarization="s"):
        resolution = self.resolution(energy_keV=energy_keV,order=order,
                beam_div_urad=beam_div_urad,polarization=polarization)
        wavelength = 12.398/energy_keV*1e-10
        coherence_length = wavelength*resolution
        return coherence_length


def compare_monos(hkls=[ [1,1,1],[3,1,1],[3,3,3]],beam_div_urad=10,tempC=-170,polarization="s"):
    energy = np.arange(6,35,0.01)

    fig,ax = plt.subplots(3,1,sharex=True)
    for hkl in hkls:
        m = Monochromator(hkl=hkl,tempC=tempC)
        ax[0].plot(energy,m.bragg_angle_deg(energy))
        ax[1].plot(energy,m.darwin_width(energy,polarization=polarization)*1e6)
        ax[2].plot(energy,m.coherence_length(energy,beam_div_urad=beam_div_urad,polarization=polarization)*1e6,label=m.name)
    ax[0].set_ylabel("bragg angle (deg)")
    ax[1].set_ylabel("darwin width (urad)")
    ax[2].set_ylabel("coherence length (um)")
    ax[2].legend()
    for a in ax.ravel(): a.grid()
    title = f"Monochromator comparison, T={tempC+273}\nbeam divergence {beam_div_urad} urad, polarization {polarization}"
    ax[0].set_title(title)
