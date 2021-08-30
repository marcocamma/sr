import numpy as np
from matplotlib import pyplot as plt

import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes

from xrt.backends.raycing.materials import CH

from datastorage import DataStorage as ds

class Monochromator:
    def __init__(self,hkl=[1,1,1],tempC=-140,gap=5):
        if isinstance(hkl,str): hkl = [int(i) for i in hkl]
        cryst_name = f"Si{hkl[0]}{hkl[1]}{hkl[2]}"
        self.name = cryst_name
        self.temperature = 273.15+tempC
        self.gap=gap
        self.cryst = rmats.CrystalSi(hkl=hkl,name=cryst_name,tK=self.temperature)
        self.dcm = roes.DCM(
            name=r"DCM",
            material=self.cryst,
            material2=self.cryst,
            cryst2perpTransl=gap)

    def bragg_angle_deg(self,energy_keV=10,order=1):
        energy_eV = energy_keV*1e3
        bragg_angle = self.cryst.get_Bragg_angle(energy_eV,order=order)
        return np.rad2deg(bragg_angle)

    def vertical_offset(self,energy_keV=10,order=1):
        angle_deg = self.bragg_angle_deg(energy_keV=energy_keV,order=order)
        angle = np.deg2rad(angle_deg)
        offset = 2*self.gap*np.cos(angle)
        return offset

    def diffracted_energy_kev(self,angle_deg=10,order=1):
        angle = np.deg2rad(angle_deg)
        sin_angle = np.sin(angle)
        E = order*CH/2/self.cryst.d/sin_angle 
        return E/1e3

    def darwin_width(self,energy_keV=10,polarization="s"):
        energy_eV = energy_keV*1e3
        dw = self.cryst.get_Darwin_width(energy_eV,polarization=polarization)
        return dw

    def intrinsic_resolution(self,energy_keV=10,polarization="s"):
        # see https://www.esrf.fr/computing/scientific/people/srio/publications/SPIE04_MONO.pdf
        dw = self.darwin_width(energy_keV=energy_keV,polarization=polarization)
        bragg_angle = np.deg2rad(self.bragg_angle_deg(energy_keV=energy_keV))
        res = dw/np.arctan(bragg_angle)
        return res

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


def beamsize_mono_vibrations(focus_dist=200,mono_dist=40,lens_dist=60,rms_vibration=0.1e-6,source_rms_size=5e-6,source_rms_div=5e-6,lens_rms_aperture=300e-6,z=None):
    """  based on ttp://dx.doi.org/10.1107/S16005775160111881 
         Parameters
         ----------
         mono_dist : float
             distance from mono to source
    """
    # eq 10
    zi = focus_dist-lens_dist
    if z is None: z = np.linspace(zi-10,zi+10,1001)
    zo = lens_dist
    zm = mono_dist
    beam_size = zi/zo*np.sqrt(source_rms_size**2+(2*zm*rms_vibration)**2)
    beam_size_no_vibration  = zi/zo*source_rms_size

    A = lens_rms_aperture
    B = lens_dist*source_rms_div
    # 
    r = zi/zo
    r2 = (zi/zo)**2
    position_focus_from_lens = zi-zi*(2*rms_vibration*zm)**2*(1/A**2*r2+1/B**2*(r2+r-zi/zm))
    position_focus = position_focus_from_lens+lens_dist
    # eq 4
    c_z = z/zo-(zo-zm)*(z-zi)/ (zm*zi*(1+B**2/A**2))
    s_0_z_squared = source_rms_size**2*r2+(z/zi-1)**2/(1/A**2+1/B**2)
    s_z_squared = s_0_z_squared + (2*zm*rms_vibration*c_z)**2/(1+(2*rms_vibration*(zo-zm))**2/(A**2+B**2))
    s_z = np.sqrt(s_z_squared)

    ret = ds(
            rms_beamsize_at_dist=beam_size,
            fwhm_beamsize_at_dist=beam_size*2.35,
            fractional_change_of_beamsize=(beam_size-beam_size_no_vibration)/beam_size_no_vibration,
            focus_position=position_focus,
            fraction_change_of_focus_position=(position_focus-focus_dist)/focus_dist,
            z = z+lens_dist,
            rms_beamsize = s_z
            )
    return ret
