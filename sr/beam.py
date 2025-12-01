# -*- coding: utf-8 -*-
""" Module for storage ring calculations; all sizes and divergences are in RMS """

import numpy as np
from datastorage import DataStorage as ds
from scipy.special import erf
from .utils.conversion import energy_to_wavelength, wavelength_to_energy

def ebs_minibeta(beta_h=6.8,beta_v=2.7):
    eps_h = 130e-12
    eps_v = 10e-12
    return ds(
        sh = np.sqrt(eps_h*beta_h),
        divh = np.sqrt(eps_h/beta_h),
        sv = np.sqrt(eps_v*beta_v),
        divv = np.sqrt(eps_v/beta_v),
        ebeam_energy=6,
        sr_cur=0.2,
        rms_energy_spread=0.00094,
        name="minibeta",
        betav=beta_v,
        betah=beta_h,
    )


_e_lattices = ds(
    esrf_highb=ds(
        sh=387.80e-6,
        divh=10.31e-6,
        sv=5.43e-6,
        divv=1.84e-6,
        ebeam_energy=6.04,
        sr_cur=0.2,
        rms_energy_spread=0.001,
    ),
    ebs=ebs_minibeta(beta_h=6.8,beta_v=2.8),
    super_source=ds(
        sh=3e-6,
        divh=1e-6,
        sv=1.0e-6,
        divv=1e-6,
        ebeam_energy=6,
        sr_cur=0.2,
        rms_energy_spread=0.00094,
    ),
    jsr2019=ds(
        sh=4.47e-6,
        divh=2.24e-6,
        sv=4.47e-6,
        divv=2.24e-6,
        ebeam_energy=6,
        sr_cur=0.1,
        rms_energy_spread=0.001,
    ),
    jsr2019es0=ds(
        sh=4.47e-6,
        divh=2.24e-6,
        sv=4.47e-6,
        divv=2.24e-6,
        ebeam_energy=6,
        sr_cur=0.1,
        rms_energy_spread=0.000,
    ),

)



# utility functions
def _sqrt_squares(*K):
    """ return sqrt(a1**2+a2**2+...) """
    s = 0
    for k in K:
        s += k ** 2
    return np.sqrt(s)


def e_beam(lattice="EBS"):
    if isinstance(lattice,str):
        lattice = lattice.casefold()
        if not lattice in _e_lattices:
            raise KeyError(
                f"Can't find {lattice} in database; available lattices are (not case sensitive): {str(list(_e_lattices.keys()))}"
            )
        e = ds(_e_lattices[lattice].copy())
    else:
        e = ds(lattice.copy())
        lattice = "source"
    e.emitth = e.sh * e.divh
    e.emittv = e.sv * e.divv
    e.betah = e.sh/e.divh
    e.betav = e.sv/e.divv
    e.name = lattice
    return e


def srw_ebeam(ebeam_dict):
    from srwpy import srwlib

    # ***********Electron Beam
    eBeam = srwlib.SRWLPartBeam()
    eBeam.Iavg = ebeam_dict["sr_cur"]  # average current [A]
    eBeam.partStatMom1.x = 0.0  # initial transverse positions [m]
    eBeam.partStatMom1.y = 0.0
    eBeam.partStatMom1.z = (
        0.0
    )  # initial longitudinal positions (set in the middle of undulator)
    eBeam.partStatMom1.xp = 0  # initial relative transverse velocities
    eBeam.partStatMom1.yp = 0
    eBeam.partStatMom1.gamma = (
        ebeam_dict["ebeam_energy"] / 0.51099890221e-03
    )  # relative energy
    sigEperE = ebeam_dict["rms_energy_spread"]  # relative RMS energy spread
    sigX = ebeam_dict["sh"]  # horizontal RMS size of e-beam [m]
    sigXp = ebeam_dict["divh"]  # horizontal RMS angular divergence [rad]
    sigY = ebeam_dict["sv"]  # vertical RMS size of e-beam [m]
    sigYp = ebeam_dict["divv"]  # vertical RMS angular divergence [rad]
    # 2nd order stat. moments:
    eBeam.arStatMom2[0] = sigX * sigX  # <(x-<x>)^2>
    eBeam.arStatMom2[1] = 0  # <(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp * sigXp  # <(x'-<x'>)^2>
    eBeam.arStatMom2[3] = sigY * sigY  # <(y-<y>)^2>
    eBeam.arStatMom2[4] = 0  # <(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp * sigYp  # <(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE * sigEperE  # <(E-<E>)^2>/<E>^2
    return eBeam


def Qa(x):
    """
    Tanaka et al 2009 JSR eq 17
    doi:10.1107/S0909049509009479
    Expression as Vartanyans JSR 2019
    """
    if isinstance(x, (float, int)) and x < 1e-5:
        x = 1e-5
    elif isinstance(x, np.ndarray):
        x = x.copy()
        x[x < 1e-5] = 1e-5
    down = -1 + np.exp(-2 * x ** 2) + np.sqrt(2 * np.pi) * x * erf(np.sqrt(2) * x)
    up = 2 * x ** 2
    ratio = up/down
    qa = np.sqrt(ratio)
    return qa


def Qs(x):
    """
    Tanaka et al 2009 JSR eq 24
    doi:10.1107/S0909049509009479

    Expression as Vartanyans JSR 2019
    """
    return 2 * Qa(x / 4) ** (2 / 3)

def KKN(nu):
    """
    Tanaka et al 2009 JSR eq 17
    doi:10.1107/S0909049509009479
    Expression as Vartanyans JSR 2019
    """
    if isinstance(nu, (float, int)) and nu < 1e-5:
        nu = 1e-5
    elif isinstance(nu, np.ndarray):
        nu = nu.copy()
        nu[nu < 1e-5] = 1e-5
    pi = np.pi
    pi2 = pi*pi
    up = 8*pi2*nu**2
    down = np.sqrt(2*pi)*2*pi*nu*erf(np.sqrt(2)*2*pi*nu)+np.exp(-8*pi2*nu**2)-1
    ratio = up/down
    return ratio



class Photon_Beam:
    def __init__(
        self,
        wavelength=1,
        undulator_L=1,
        undulator_period=20,
        harmonic=1,
        lattice="EBS",
    ):
        if isinstance(lattice,str):
            self.ebeam = e_beam(lattice=lattice)
            self.lattice = lattice
        else:
            self.ebeam = lattice
            self.lattice = lattice.name
        self.harmonic = harmonic
        self.undulator_period = undulator_period
        self.undulator_N = undulator_L * 1e3 / undulator_period
        self.wavelength = wavelength
        self.undulator_L = undulator_L
        self._calculate()
        self._calculate_gsm_with_DelRio()

    def rms_size_at_dist(self, D=100):
        sh = _sqrt_squares(self.sh, self.divh * D)
        sv = _sqrt_squares(self.sv, self.divv * D)
        divh = self.divh
        divv = self.divv
        return ds(sh=sh, sv=sv, divh=divh, divv=divv)

    def rms_cl_at_dist(self, dist=100):
        sh, dh = self.gsm_sclh, self.gsm_cldivh
        sv, dv = self.gsm_sclv, self.gsm_cldivv
        clh, clv = _sqrt_squares(sh, dh * dist), _sqrt_squares(sv, dv * dist)
        return ds(clh=clh, clv=clv, cldivh=dh, cldivv=dv)

    def get_ebeam(self):
        return ds(self.ebeam.copy())

    def _calculate(self):
        """ emission from single electron from 
        Tanaka et al 2009 JSR eq 17
        doi:10.1107/S0909049509009479
        """
        e = self.ebeam

        wavelength = self.wavelength / 1e10  # convert to m

        # these are the classical expressions for single electron emission
        # see Kim, K. J. (1989). AIP Conf. Proc. 184, 565–632.
        xsize = 1 / 4 / np.pi * np.sqrt(2 * wavelength * self.undulator_L)
        xdiv = np.sqrt(wavelength / 2 / self.undulator_L)

        self.single_electron_emission_size_classic = xsize
        self.single_electron_emission_div_classic = xdiv
        self.single_electron_emission_emittance_classic = xsize * xdiv

        self.sh_old = _sqrt_squares(e.sh, xsize)
        self.sv_old = _sqrt_squares(e.sv, xsize)
        self.divh_old = _sqrt_squares(e.divh, xdiv)
        self.divv_old = _sqrt_squares(e.divv, xdiv)
        self.emitth_old = self.sh_old*self.divh_old
        self.emittv_old = self.sv_old*self.divv_old
        self.emitt_old = self.emitth_old*self.emittv_old

        # Tanaka & Kitamura doi:10.1107/S0909049509009479 eq 13
        radiation_relative_bandwidth = 1/(self.undulator_N*self.harmonic)
        self.normalized_energy_spread = e.rms_energy_spread/radiation_relative_bandwidth

        self.single_electron_emission_size = xsize * 2  # Qs(0)=2
        self.single_electron_emission_div = xdiv * 1  # Qa(0)=1

        # the single_electron_emittance is equal to wavelength/(2pi) 
        # and _NOT_ wavelength/(4pi) : see Vartanyans JSR 2019
        self.single_electron_emission_emittance = wavelength/2/np.pi

        # shorter notation for following formulas
        nu = self.normalized_energy_spread
        xsizeKK = xsize * 2*KKN(nu/4)**(1/3)
        xdivKK = xdiv * KKN(nu)**(1/2)

        self.sh = _sqrt_squares(e.sh, xsizeKK)
        self.sv = _sqrt_squares(e.sv, xsizeKK)
        self.divh = _sqrt_squares(e.divh, xdivKK)
        self.divv = _sqrt_squares(e.divv, xdivKK)

        self.emitth = self.sh * self.divh
        self.emittv = self.sv * self.divv
        self.emitt = self.emitth*self.emittv

        see = self.single_electron_emission_emittance
        self.cofh = see / self.emitth
        self.cofv = see / self.emittv
        self.cof = self.cofh * self.cofv

    def _calculate_gsm_with_2pi(self):
        """
        based on Vartanyans and Singer N J. of Physics 2010
        doi: 10.1088/1367-2630/12/3/035004
        but using eps_coh of w/2/pi
        """
        print("sr.beam will use GSM parameters definition")
        print("that uses λ/2π")
        w = self.wavelength*1e-10
        # in 2019 paper Vartanyans and coworkers discuss that the
        # minimum emittance is closer to lambda/2/pi, this is the reason
        # for chaning the original eq33 from the 2010 paper
        emitt_coh = w/(2*np.pi)
        k = 2*np.pi/w
        # sclh = source coherence length horizontal
        self.gsm_sclh = (
            2 * self.sh / np.sqrt( (self.emitth/emitt_coh) ** 2 - 1)
        )  # eq 33
        self.gsm_sclv = (
            2 * self.sv / np.sqrt( (self.emittv/emitt_coh) ** 2 - 1)
        )  # eq 33
        self.gsm_qh = self.gsm_sclh / self.sh
        self.gsm_qv = self.gsm_sclv / self.sv
        self.gsm_cofh = self.gsm_qh / np.sqrt(4 + self.gsm_qh ** 2)
        self.gsm_cofv = self.gsm_qv / np.sqrt(4 + self.gsm_qv ** 2)
        self.gsm_cldivh = 1 / (2 * k * self.sh) * np.sqrt(4 + self.gsm_qh ** 2)
        self.gsm_cldivv = 1 / (2 * k * self.sv) * np.sqrt(4 + self.gsm_qv ** 2)

    def _calculate_gsm_with_4pi(self):
        """
        based on Vartanyans and Singer N J. of Physics 2010
        doi: 10.1088/1367-2630/12/3/035004
        """
        #print("sr.beam will use GSM parameters definition")
        #print("that uses λ/4π, else the divergence is too small")
        w = self.wavelength*1e-10
        # although in 2019 paper Vartanyans and coworkers discuss that the
        # minimum emittance is closer to lambda/2/pi, the agreement with
        # the experimental data shown in the 2010 paper was calculated with
        # lambda/4/pi. This is the reason of using it here
        emitt_coh = w/(4*np.pi)
        k = 2*np.pi/w
        # sclh = source coherence length horizontal
        self.gsm_sclh = (
            2 * self.sh / np.sqrt( (self.emitth/emitt_coh) ** 2 - 1)
        )  # eq 33
        self.gsm_sclv = (
            2 * self.sv / np.sqrt( (self.emittv/emitt_coh) ** 2 - 1)
        )  # eq 33
        self.gsm_qh = self.gsm_sclh / self.sh
        self.gsm_qv = self.gsm_sclv / self.sv
        self.gsm_cofh = self.gsm_qh / np.sqrt(4 + self.gsm_qh ** 2)
        self.gsm_cofv = self.gsm_qv / np.sqrt(4 + self.gsm_qv ** 2)
        self.gsm_cldivh = 1 / (2 * k * self.sh) * np.sqrt(4 + self.gsm_qh ** 2)
        self.gsm_cldivv = 1 / (2 * k * self.sv) * np.sqrt(4 + self.gsm_qv ** 2)

    def _calculate_gsm_with_DelRio(self):
        """
        Calculate GSM parameters based on Manuel Del Rio
        matching of the CF = (λ/2π) / ε_tanaka 
        to the CF of the GSM; it results in
        ξ = σ*CF/(1-CF)
        """
        w = self.wavelength*1e-10
        emitt_coh = w/(2*np.pi)

        k = 2*np.pi/w
        # sclh = source coherence length horizontal
        cfh = self.cofh
        cfv = self.cofv
        self.gsm_sclh = self.sh*cfh/(1-cfh)
        self.gsm_cldivh = self.divh*cfh/(1-cfh)
        self.gsm_sclv = self.sv*cfv/(1-cfv)
        self.gsm_cldivv = self.divv*cfv/(1-cfv)


    def __str__(self):
        e = self.ebeam
        s = "Photon Beam instance\n"
        s += f"Harmonic Number : {self.harmonic}\n"
        s += f"E-Beam Lattice  : {self.lattice}\n"
        s += f"Wavelength      : {self.wavelength:.3f} Å\n"
        s += f"Photon Energy   : {12.398/self.wavelength:.3f} keV\n"
        s += f"Undulator L     : {self.undulator_L} m\n"
        title = "Electron Beam"
        s += title + " " + "-"*(50-len(title)) + "\n"
        s += f"RMS size  (h,v) : ({e.sh*1e6:.2f},{e.sv*1e6:.2f}) μm\n"
        s += f"RMS div   (h,v) : ({e.divh*1e6:.2f},{e.divv*1e6:.2f}) μrad\n"
        s += f"RMS relative energy spread : {e.rms_energy_spread:.4f}\n"
        s += f"emittance (h,v) : ({e.emitth*1e12:.2f},{e.emittv*1e12:.2f}) pm*rad\n"
        title = "Single Electron Radiation"
        s += title + " " + "-"*(50-len(title)) + "\n"
        s += f"RMS size  : {self.single_electron_emission_size*1e6:.2f} μm\n"
        s += f"RMS div   : {self.single_electron_emission_div*1e6:.2f} μrad\n"
        s += f"RMS emitt : {self.single_electron_emission_emittance*1e12:.2f} pm*rad\n"
        title = "X-ray Radiation"
        s += title + " " + "-"*(50-len(title)) + "\n"

        s += f"RMS size (h,v)  : ({self.sh*1e6:.2f},{self.sv*1e6:.2f}) μm\n"
        s += f"RMS div  (h,v)  : ({self.divh*1e6:.2f},{self.divv*1e6:.2f}) μrad\n"
        s += f"emittance (h,v) : ({self.emitth*1e12:.2f},{self.emittv*1e12:.2f}) pm*rad\n"
        s += f"Coherent fraction (h,v,tot) = ({self.cofh:.2f},{self.cofv:.2f},{self.cof:.2f})\n"
        s += f"Coherent lengths RMS size at source (h,v) = ({self.gsm_sclh*1e6:.2f},{self.gsm_sclv*1e6:.2f}) μm\n"
        s += f"Coherent lengths RMS divergence (h,v) = ({self.gsm_cldivh*1e6:.2f},{self.gsm_cldivv*1e6:.2f}) μrad"
        return s

    def __repr__(self):
        return self.__str__()


jsr2019 = Photon_Beam(
    lattice="jsr2019",
    undulator_L=5,
    harmonic=3,
    undulator_period=29,
    wavelength=12.4 / 12,
)
