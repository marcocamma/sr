import numpy as np
from scipy import special
from scipy.constants import m_e, c, eV
from scipy.integrate import trapezoid, cumulative_trapezoid
import functools
from datastorage import DataStorage as ds
import copy
from . import beam
from .utils.conversion import energy_to_wavelength

try:
    from .abcd.optics import GSM_Numeric
except ImportError:
    pass
mc2 = m_e * c**2 / eV / 1e9


def _integrate2d(h, v, i):
    # dh = h[1]-h[0]
    # dv = v[1]-v[0]
    # return i.sum(axis=(1,2))*dv*dh
    return trapezoid(trapezoid(i, x=h, axis=-1), x=v, axis=-1)


def _integrate1d(e, i):
    return trapezoid(i, x=e, axis=0)
    # de = e[1]-e[0]
    # return i.sum(axis=0)*de


def b_field(halbach_coeff=3, b_coeff=1, gap=6, period=18):
    return halbach_coeff * np.exp(-b_coeff * np.pi * gap / period)


def b_field_roomT(gap=6, period=18):
    halbach_coeff = 2.083
    b_coeff = 1.0054  # from J. Chavanne 2018 Document
    # halbach_coeff = 2.2  ; b_coeff = 0.99   # tweaked to match G. Le Bec examples
    return b_field(halbach_coeff=halbach_coeff, b_coeff=b_coeff, gap=gap, period=period)


def b_field_cryoT(gap=6, period=18):
    halbach_coeff = 3.29
    b_coeff = 1.095  # from J. Chavanne 2018 Document
    return b_field(halbach_coeff=halbach_coeff, b_coeff=b_coeff, gap=gap, period=period)


def gamma(E):
    """relativisitc approxiamtion"""
    return E / mc2


def k_value(B=0.5, period=18):
    """k = e/2/pi/m/c*period*B
    B is peak field of undulator [T]
    period is period in mm"""
    return 0.09337 * B * period


def b_field_from_k(k=1, period=18):
    return k / 0.09337 / period


def E_undulator(harmonic=1, ebeam_energy=6, k=1.5, period=18, theta=0):
    g = gamma(ebeam_energy)
    period_cm = period / 10
    num = harmonic * 0.95 * ebeam_energy**2
    den = period_cm * (1 + k**2 / 2 + (g * theta) ** 2)
    return num / den


def _Q(harmonic=1, k=1.5):
    x = harmonic * k**2 / 2 / (k**2 + 2)
    o2 = int((harmonic + 1) / 2)
    o1 = int((harmonic - 1) / 2)
    q = (
        harmonic
        * k**2
        / (1 + (k**2 / 2))
        * (special.jn(o1, x) - special.jn(o2, x)) ** 2
    )
    return q


def photon_flux(harmonic=1, nperiods=100, sr_current=0.2, k=1.5):
    """returns flux in photons/sec/1e-3 bandwidth
    from hercules book n. page 50
    """
    q = _Q(harmonic=harmonic, k=k)
    return 1.43e14 * nperiods * sr_current * q


def total_power(ebeam_energy=6, peak_field=1, undulator_L=1, sr_current=0.2):
    """return power in W"""
    return 0.633 * 1e3 * ebeam_energy**2 * peak_field**2 * undulator_L * sr_current


def power_density(
    ebeam_energy=6, peak_field=1, undulator_n_period=100, k=1.5, sr_current=0.2
):
    """return power density (W/mrad**2); G(K) is approaximated !!"""
    G = 2 * np.arctan(k * np.pi) / np.pi
    return 10.84 * ebeam_energy**4 * peak_field * undulator_n_period * sr_current * G


def _photon_flux_density_helper(data):
    units_spectral_photon_flux_density = "ph/sec/0.1%BW/mm^2"
    units_spectral_photon_flux = "ph/sec/0.1%BW"
    keV_to_J = eV * 1e3
    energy_one_photon = data.energy * keV_to_J
    bandwidth_factor = 1 / (data.energy * 1e-3)
    _power_factor = energy_one_photon * bandwidth_factor
    spectral_power = data.spectral_photon_flux * _power_factor
    units_spectral_power = "W/keV"
    _power_factor = _power_factor[:, np.newaxis, np.newaxis]
    spectral_power_density = data.spectral_photon_flux_density * _power_factor
    units_spectral_power_density = "W/mm2/keV"

    if len(data.energy) == 1:
        power_density = spectral_power_density
        power_cumulative = None
        power_total = np.nan
    else:
        power_density = _integrate1d(data.energy, spectral_power_density)
        power_cumulative = cumulative_trapezoid(spectral_power, x=data.energy, initial=0)
        power_total = power_cumulative[-1]
    units_power_density = "W/mm2"
    units_power_cumulative = "W"
    units_power_total = "W"

    to_add = dict(
        spectral_power_density=spectral_power_density,
        power_density=power_density,
        power_cumulative=power_cumulative,
        power_total=power_total,
        spectral_photon_flux_max=data.spectral_photon_flux.max(),
        spectral_photon_flux_at_max=data.energy[np.argmax(data.spectral_photon_flux)],
        units_spectral_photon_flux_max="ph",
        units_spectral_photon_flux_at_max="keV",
        units_spectral_photon_flux_density=units_spectral_photon_flux_density,
        units_spectral_photon_flux=units_spectral_photon_flux,
        units_power_cumulative=units_power_cumulative,
        units_power_total=units_power_total,
        units_spectral_power_density=units_spectral_power_density,
        units_power_density=units_power_density,
    )

    data.update(**to_add)
    return data


class Undulator:
    def __init__(
        self,
        length=2,
        gap_to_b="cryoT",
        period=18,
        elattice="EBS",
        name="undulator",
        min_gap=6,
        max_gap=40,
        k=None,
        ebeam_pars={},
    ):
        """
        Creates undulator instance

        Parameters
        ----------
        gap_to_b : func, str or None
            is the function that converts gap (in mm) and period (in mm)
            to B in tesla. If None k value has to be given. If string, it will
            look for corresponding function

        k : None or float

        """
        self.elattice = elattice
        ebeam = beam.e_beam(lattice=elattice)
        # create local copy
        ebeam = copy.deepcopy(ebeam)
        ebeam.update(ebeam_pars)
        self.ebeam = ebeam

        self.period = period
        self.sr_current = ebeam.sr_cur
        self.ebeam_energy = ebeam.ebeam_energy
        if isinstance(gap_to_b, str):
            names = ["roomT", "cryoT"]
            if not gap_to_b in names:
                raise ValueError("gap_to_b string has to be one of ", str(names))
            gap_to_b = globals()["b_field_%s" % gap_to_b]
        self._gap_to_b = gap_to_b
        if k is not None:
            b = b_field_from_k(k, period)
            self._gap_to_b = b
        self.length = length
        self.name = name
        self.N = int(length / (period / 1e3))
        self.min_gap = min_gap
        self.max_gap = max_gap

    def calc_sizes_and_coherence_lengths(
        self, dist, gap="min", energy=None, harmonic=1, return_fwhm=False, **kwargs
    ):
        """by default returns RMS size and divergences"""
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]
        energy = self.photon_energy(gap=gap, harmonic=harmonic)
        wavelength = energy_to_wavelength(energy)  # *1e-10
        b = beam.Photon_Beam(
            wavelength=wavelength,
            undulator_L=self.length,
            harmonic=harmonic,
            lattice=self.elattice,
        )
        cl = b.rms_cl_at_dist(dist)
        bs = b.rms_size_at_dist(dist)
        cl.update(bs)
        if return_fwhm:
            for k, v in cl.items():
                cl[k] = v * 2.35
        return cl

    def as_srw(self, gap="min", energy=None, harmonic=1, **kwargs):
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        from srwpy import srwlib

        ebeam = beam.srw_ebeam(self.ebeam)
        harmB = srwlib.SRWLMagFldH()  # magnetic field harmonic
        harmB.n = harmonic  # harmonic number
        harmB.h_or_v = "v"  # magnetic field plane: horzontal ('h') or vertical ('v')
        harmB.B = self.field(gap=gap)  # magnetic field amplitude [T]
        und = srwlib.SRWLMagFldU([harmB])
        und.per = self.period / 1e3  # period length [m]
        und.nPer = self.N  # number of periods (will be rounded to integer)
        magFldCnt = srwlib.SRWLMagFldC(
            [und],
            srwlib.array("d", [0]),
            srwlib.array("d", [0]),
            srwlib.array("d", [0]),
        )  # Container of all magnetic field elements
        return ds(ebeam=ebeam, und=und, field=magFldCnt)

    def as_gsm(self, gap="min", energy=None, harmonic=1, distance=None, **kwargs):
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]
        b = self.photon_beam_characteristics(gap=gap, harmonic=harmonic, **kwargs)
        gsmh = GSM_Numeric(
            rms_size=b.sh, rms_cl=b.gsm_sclh, wavelen=b.wavelength * 1e-10
        )
        gsmv = GSM_Numeric(
            rms_size=b.sv, rms_cl=b.gsm_sclv, wavelen=b.wavelength * 1e-10
        )
        if distance is not None:
            gsmh = gsmh.propagate(distance)
            gsmv = gsmv.propagate(distance)
        return ds(h=gsmh, v=gsmv)

    def as_wolfry(
        self, gap="min", energy=None, harmonic=1, npoints=400, k="auto", **kwargs
    ):

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap

        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=False
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]
        else:
            energy = self.photon_energy(gap=gap, harmonic=harmonic)

        return WolfryUndulator(
            energy=energy,
            npoints=npoints,
            undulator=self,
            k=k,
        )

    def as_gsm(self, gap="min", energy=None, harmonic=1, distance=None, **kwargs):
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]
        b = self.photon_beam_characteristics(gap=gap, harmonic=harmonic, **kwargs)
        gsmh = GSM_Numeric(
            rms_size=b.sh, rms_cl=b.gsm_sclh, wavelen=b.wavelength * 1e-10
        )
        gsmv = GSM_Numeric(
            rms_size=b.sv, rms_cl=b.gsm_sclv, wavelen=b.wavelength * 1e-10
        )
        if distance is not None:
            gsmh = gsmh.propagate(distance)
            gsmv = gsmv.propagate(distance)
        return ds(h=gsmh, v=gsmv)

    def srw_power_density(
        self,
        gap="min",
        dist=30,
        energy=None,
        h=[-3, 3],
        nh=151,
        v=[-2, 2],
        nv=101,
        **kwargs,
    ):
        """
        Calculate power density (W/mm^2)
        Parameters
        ----------
        dist : float [m]
            distance from source
        h : [min,max] or gap [mm]
            start/end of horizontal slit. If a single value the range -gap/2,gap/2 is used
        nh: int
            number of point in horizontal direction
        v : [min,max] or gap [mm]
            start/end of vertical slit. If a single value the range -gap/2,gap/2 is used
        nv: int
            number of point in vertical direction
        """
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=True
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]
        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        from srwpy import srwlib
        from srwpy import srwlpy

        if isinstance(v, (float, int)):
            v = [-v / 2, v / 2]
        if isinstance(h, (float, int)):
            h = [-h / 2, h / 2]

        data = self.as_srw(gap=gap)

        arPrecP = [0] * 5  # for power density
        arPrecP[0] = 1.5  # precision factor
        arPrecP[
            1
        ] = 2  # power density computation method (1- "near field", 2- "far field")
        arPrecP[2] = (
            -self.length / 2
        )  # initial longitudinal position (effective if arPrecP[2] < arPrecP[3])
        arPrecP[3] = (
            self.length / 2
        )  # final longitudinal position (effective if arPrecP[2] < arPrecP[3])
        arPrecP[4] = 20000  # number of points for (intermediate) trajectory calculation

        stkP = srwlib.SRWLStokes()  # for power density
        shape = (1, nh, nv)
        stkP.allocate(
            *shape
        )  # numbers of points vs horizontal and vertical positions (photon energy is not taken into account)
        stkP.mesh.zStart = dist  # longitudinal position [m] at which power density has to be calculated
        stkP.mesh.xStart = h[0] / 1e3  # initial horizontal position [m]
        stkP.mesh.xFin = h[1] / 1e3  # final horizontal position [m]
        stkP.mesh.yStart = v[0] / 1e3  # initial vertical position [m]
        stkP.mesh.yFin = v[1] / 1e3  # final vertical position [m]
        #        stkP.mesh.eStart =
        #        stkP.mesh.eFin = 8000
        srwlpy.CalcPowDenSR(stkP, data["ebeam"], 0, data["field"], arPrecP)
        pd = np.asarray(stkP.arS[: nh * nv]).reshape((nv, nh))
        h = np.linspace(h[0], h[1], nh)
        v = np.linspace(v[0], v[1], nv)
        power_total = _integrate2d(h, v, pd)
        units_power_density = "W/mm2"
        units_power_total = "W"
        units_h = "mm"
        units_v = "mm"
        return ds(
            h=h,
            v=v,
            power_density=pd,
            power_density_max=pd.max(),
            power_total=power_total,
            units_power_density=units_power_density,
            units_power_density_max=units_power_density,
            units_power_total=units_power_total,
            units_h=units_h,
            units_v=units_v,
        )

    def srw_photon_flux_density(
        self,
        gap="min",
        energy=None,
        dist=30,
        h=[-0.6, 0.6],
        nh=13,
        v=[-0.5, 0.5],
        nv=11,
        e=[1, 30],
        ne=200,
        harmonic="auto",
        abs_filter=None,
        **kwargs,
    ):
        """
        Calculate intensity spectral density (ph/sec)
        Parameters
        ----------
        dist : float [m]
            distance from source
        h : (min,max) or value [mm]
            start/end of horizontal slit. If a single value the range -value/2,value/2 is used
        nh: int
            number of point in horizontal direction
        v : (min,max) or value [mm]
            start/end of vertical slit. If a single value the range -value/2,value/2 is used

        nv: int
            number of point in vertical direction
        e : [min,max] or value [keV]
            start/end of energy spectral_photon_flux.
        ne: int
            number of point in energy spectral_photon_flux. If e is single value, forced to 1
        harmonic : int|(min,max)|"auto"
            harmonic to consider, if auto it is autodetected based on energy range
            if int, only that harmonic is used
        abs_filter : None or object with calc_transmission(energy_kev)
                     method
        """
        from srwpy import srwlib
        from srwpy import srwlpy

        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=True
            )
            if len(pars) == 0:
                return None
            else:
                pars = pars[0]
            gap = pars["gap"]
            # harmonic will be determined below

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        if isinstance(v, (float, int)):
            v = [-v / 2, v / 2]
        if isinstance(h, (float, int)):
            h = [-h / 2, h / 2]

        data = self.as_srw(gap=gap)

        if harmonic == "auto":
            harmonic_m = int(max(np.floor(e[0] / self.fundamental(gap=gap)), 1))
            harmonic_M = int(max(np.ceil(e[1] / self.fundamental(gap=gap)), 1))
        elif isinstance(harmonic, int):
            harmonic_m = harmonic
            harmonic_M = harmonic
        else:
            harmonic_m, harmonic_M = harmonic

        if isinstance(e, (float, int)):
            if e == 0:
                e = self.photon_energy(gap=gap, harmonic=harmonic[0])
            e = [e, e]
            ne = 1
        elif isinstance(e, (tuple, list)) and e[0] < 0:
            e0 = self.photon_energy(gap=gap, harmonic=harmonic_m)
            e = [e0 + e[0], e0 + e[1]]

        calc_flux = nh == 1 and nv == 1

        arPrecF = [0] * 5  # for spectral flux vs photon energy
        arPrecF[0] = harmonic_m  # initial UR harmonic to take into account
        arPrecF[1] = harmonic_M  # final UR harmonic to take into account
        arPrecF[2] = 1.5  # longitudinal integration precision parameter
        arPrecF[3] = 1.5  # azimuthal integration precision parameter
        if calc_flux:
            arPrecF[4] = 1  # calculate flux (1) or flux per unit surface (2)
        else:
            arPrecF[4] = 2  # calculate flux (1) or flux per unit surface (2)

        stk = srwlib.SRWLStokes()  # for spectral_photon_flux
        shape = (ne, nh, nv)
        stk.allocate(
            *shape
        )  # numbers of points vs horizontal and vertical positions (photon energy is not taken into account)
        stk.mesh.zStart = dist  # longitudinal position [m] at which power density has to be calculated
        stk.mesh.xStart = h[0] / 1e3  # initial horizontal position [m]
        stk.mesh.xFin = h[1] / 1e3  # final horizontal position [m]
        stk.mesh.yStart = v[0] / 1e3  # initial vertical position [m]
        stk.mesh.yFin = v[1] / 1e3  # final vertical position [m]
        stk.mesh.eStart = e[0] * 1e3
        stk.mesh.eFin = e[1] * 1e3

        srwlpy.CalcStokesUR(stk, data.ebeam, data.und, arPrecF)
        e = np.linspace(e[0], e[1], ne)

        dh = h[1] - h[0]
        dv = v[1] - v[0]

        h = np.linspace(h[0], h[1], nh)
        v = np.linspace(v[0], v[1], nv)

        if calc_flux:
            spectral_photon_flux = np.asarray(stk.arS[:ne])
            spectral_photon_flux_density = (
                spectral_photon_flux[:, np.newaxis, np.newaxis] / dh / dv
            )
        else:
            spectral_photon_flux_density = np.reshape(
                stk.arS[: ne * nh * nv], (nv, nh, ne)
            )
            spectral_photon_flux_density = np.moveaxis(
                spectral_photon_flux_density, 2, 0
            )
            spectral_photon_flux = _integrate2d(h, v, spectral_photon_flux_density)

        if abs_filter is not None:
            t = abs_filter.calc_transmission(e)
            spectral_photon_flux_density *= t[:, np.newaxis, np.newaxis]
            spectral_photon_flux *= t
            abs_info = "absorption filter : " + str(abs_filter)
        else:
            abs_info = "no absorption filter"

        data = ds(
            energy=e,
            h=h,
            v=v,
            spectral_photon_flux_density=spectral_photon_flux_density,
            spectral_photon_flux=spectral_photon_flux,
            undulator_info=str(self),
            info=f"harmonic = {str(harmonic)}, " + abs_info,
            ebeam=self.ebeam,
        )
        data = _photon_flux_density_helper(data)
        # calculates more things ...

        return data

    def srw_photon_flux(
        self,
        gap="min",
        energy=None,
        dist=30,
        h=[-0.6, 0.6],
        v=[-0.5, 0.5],
        e=[1, 30],
        ne=2901,
        harmonic="auto",
        **kwargs,
    ):
        """
        Calculate intensity spectral_photon_flux (ph/sec/0.1%BW)
        Parameters
        ----------
        See srw_spectral_irradiance docstring
        """
        ret = self.srw_photon_flux_density(
            gap=gap,
            energy=energy,
            dist=dist,
            h=h,
            nh=1,
            v=v,
            nv=1,
            e=e,
            ne=ne,
            harmonic=harmonic,
            **kwargs,
        )
        return ret

    def srw_total_flux(
        self,
        gap="min",
        energy=None,
        harmonic=1,
        **kwargs,
    ):
        """Calculate photon flux at resonance; note: SWR find maximum
        intensity a bit below resonance (even with very small slits).
        The value might be a bit underestimated
        """
        r = self.srw_photon_flux(
            gap=gap,
            energy=energy,
            dist=30,
            h=10,
            v=10,
            e=self.photon_energy(gap=gap, harmonic=harmonic),
            ne=1,
            harmonic=harmonic,
        )
        return r.spectral_photon_flux_max

    def as_xrt(
        self,
        gap="min",
        energy=None,
        harmonic=1,
        distE="BW",
        max_angle_rad=10e-3 / 30,
        nrays=1000,
        **kwargs,
    ):
        """use distE = 'eV' for
        - XRT ray tracing (for example important when using fluxkind='power')
        use distE = 'BW' for spectral calculations of sr"""
        import xrt.backends.raycing.sources as rs

        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap

        if "flux" in kwargs:
            kwargs.pop("flux")

        try:
            max_angle_rad[0]
        except TypeError:
            max_angle_rad = (max_angle_rad, max_angle_rad)

        ebeam = self.ebeam
        kw = dict(
            eE=ebeam.ebeam_energy,
            eI=ebeam.sr_cur,
            eSigmaX=ebeam.sh * 1e6,
            eSigmaZ=ebeam.sv * 1e6,
            eEpsilonX=ebeam.emitth * 1e9,
            eEpsilonZ=ebeam.emittv * 1e9,
            eEspread=ebeam.rms_energy_spread,
            period=self.period,
            n=int(self.length * 1e3 / self.period),
            K=self.k(gap="min"),
            xPrimeMax=max_angle_rad[0] * 1e3,  # wants in mrad
            zPrimeMax=max_angle_rad[1] * 1e3,  # wants in mrad
            nrays=nrays,
            targetE=[self.photon_energy(gap=gap, harmonic=harmonic) * 1e3, harmonic],
            distE=distE,
        )
        kw.update(**kwargs)
        print(kw)
        u = rs.Undulator(**kw)
        return u

    def xrt_photon_flux_density(
        self,
        gap="min",
        energy=None,
        dist=30,
        h=[-0.6, 0.6],
        nh=13,
        v=[-0.5, 0.5],
        nv=11,
        e=[1, 30],
        ne=200,
        harmonic="auto",
        abs_filter=None,
        **kwargs,
    ):
        """
        Calculate intensity spectral density (ph/sec)
        Parameters
        ----------
        dist : float [m]
            distance from source
        h : (min,max) or value [mm]
            start/end of horizontal slit. If a single value the range -value/2,value/2 is used
        nh: int
            number of point in horizontal direction
        v : (min,max) or value [mm]
            start/end of vertical slit. If a single value the range -value/2,value/2 is used

        nv: int
            number of point in vertical direction
        e : [min,max] or value [keV]
            start/end of energy spectral_photon_flux.
            if emin is neg it is interpreted as around harmonic[0]
            if e is float and == 0, the extact harmonic is used.
        ne: int
            number of point in energy spectral_photon_flux. If e is single value, forced to 1
        harmonic : int|list|None
            harmonic to consider, if auto it is autodetected based on energy range
            if int, only that harmonic is used
        """
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(e, (int, float)) and e != 0:
            e = [e, e]

        if harmonic == "auto":
            harmonic_m = int(max(np.floor(e[0] / self.fundamental(gap=gap)), 1))
            harmonic_M = int(max(np.ceil(e[1] / self.fundamental(gap=gap)), 1))
            harmonic = range(harmonic_m, harmonic_M + 1)
        if isinstance(harmonic, int):
            harmonic = (harmonic,)
        else:
            pass

        if isinstance(v, (float, int)):
            v = [-v / 2, v / 2]
        if isinstance(h, (float, int)):
            h = [-h / 2, h / 2]
        if isinstance(e, (float, int)):
            if e == 0:
                e = self.photon_energy(gap=gap, harmonic=harmonic[0])
            e = [e, e]
            ne = 1
        elif isinstance(e, (tuple, list)) and e[0] < 0:
            e0 = self.photon_energy(gap=gap, harmonic=harmonic[0])
            print("Working around", e0)
            e = [e0 + e[0], e0 + e[1]]

        if nh == 1:
            nh = 2
        if nv == 1:
            nv = 2

        dh = h[1] - h[0]
        dv = v[1] - v[0]

        h = np.linspace(h[0], h[1], nh)
        v = np.linspace(v[0], v[1], nv)
        e = np.linspace(e[0], e[1], ne)

        theta = h * 1e-3 / dist
        psi = v * 1e-3 / dist

        dtheta = dh * 1e-3 / dist
        dpsi = dv * 1e-3 / dist

        u = self.as_xrt(gap=gap)

        spectral_photon_flux_density, *_ = u.intensities_on_mesh(
            energy=e * 1e3, theta=theta, psi=psi, harmonic=harmonic
        )
        # sum over harmonics
        spectral_photon_flux_density = spectral_photon_flux_density.sum(-1)

        # convert from flux per unit solid angle into flux per mm^2
        spectral_photon_flux_density *= (dtheta * dpsi) / (dh * dv)
        spectral_photon_flux_density = np.swapaxes(spectral_photon_flux_density, 1, 2)

        if abs_filter is not None:
            t = abs_filter.calc_transmission(e)
            spectral_photon_flux_density *= t[:, np.newaxis, np.newaxis]
            abs_info = "absorption filter : " + str(abs_filter)
        else:
            abs_info = "no absorption filter"

        # integrate2d works only if len(axis) > 1
        if nh == 1 or nv == 1:
            spectral_photon_flux = spectral_photon_flux_density * (dh * dv)
        else:
            spectral_photon_flux = _integrate2d(h, v, spectral_photon_flux_density)

        data = ds(
            energy=e,
            h=h,
            v=v,
            spectral_photon_flux_density=spectral_photon_flux_density,
            spectral_photon_flux=spectral_photon_flux,
            undulator_info=str(self),
            info=f"harmonic = {str(harmonic)}, " + abs_info,
            ebeam=self.ebeam,
        )
        data = _photon_flux_density_helper(data)
        # calculates more things ...
        return data

    def field(self, gap="min", **kwargs):
        if isinstance(self._gap_to_b, (float, int)):
            return self._gap_to_b
        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        return self._gap_to_b(gap=gap, period=self.period)

    def k(self, gap="min", energy=None, **kwargs):
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        return k_value(B=self.field(gap), period=self.period)

    def photon_energy(self, gap="min", harmonic=1, theta=0, **k_value):
        """
        Returns X-ray photon energy (in keV)
        """

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        k = self.k(gap)
        return E_undulator(
            harmonic=harmonic,
            ebeam_energy=self.ebeam_energy,
            k=k,
            period=self.period,
            theta=theta,
        )

    def fundamental(self, gap="min", **kwargs):
        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        return self.photon_energy(gap=gap, harmonic=1)

    def flux(self, gap="min", energy=None, harmonic=1, **kwargs):
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        return photon_flux(
            harmonic=harmonic,
            nperiods=self.N,
            sr_current=self.sr_current,
            k=self.k(gap),
        )

    def photon_beam_characteristics(
        self, gap="min", energy=None, harmonic=1, theta=0, **kwargs
    ):
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        f = self.photon_energy(gap, harmonic=harmonic)
        wavelength = energy_to_wavelength(f)
        return beam.Photon_Beam(
            wavelength, undulator_L=self.length, lattice=self.ebeam, harmonic=harmonic
        )

    def photon_beam_emittance(self, gap="min", energy=None, harmonic=1, **kwargs):
        """in mrad^2 mm^2"""
        use_srw = kwargs.get("use_srw", False)
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        beam = self.photon_beam_characteristics(gap=gap, harmonic=harmonic)
        emitt = beam.divh * beam.divv * beam.sh * beam.sv
        emitt = emitt * 1e12  # from m2 rad2 to mm2 mrad2
        return emitt

    def brilliance(self, gap="min", energy=None, harmonic=1, use_srw=False, **kwargs):
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        if use_srw:
            f = self.srw_total_flux(gap=gap, harmonic=harmonic)
        else:
            f = self.flux(gap=gap, harmonic=harmonic)
        e = self.photon_beam_emittance(gap=gap, harmonic=harmonic)
        return f / e / (4 * np.pi**2)

    def coherent_flux(
        self, gap="min", energy=None, harmonic=1, use_srw=False, bw=1.4e-4, **kwargs
    ):
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        b = self.brilliance(gap=gap, harmonic=harmonic, use_srw=use_srw)

        e = self.photon_energy(gap=gap, harmonic=harmonic)
        lam = 12.398 / e
        coherent_flux = 1e-8 * b * (lam / 2) ** 2
        bw_factor = bw * 1e3  # brilliance is for 0.1% BW
        coherent_flux = coherent_flux * bw_factor
        return coherent_flux

    def degeneracy_number(self, gap="min", energy=None, harmonic=1, **kwargs):
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        b = self.brilliance(gap=gap, harmonic=harmonic)
        e = self.photon_energy(gap, harmonic=harmonic)
        l = beam._keV_ang / e
        D = 8.3e-25 * b * l**3  # from J. Als Nielnsen book, eq. 2.28
        return D

    def total_power(self, gap="min", energy=None, **kwargs):
        """kwargs is used to throw at it other parameters from gap scan"""
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        if isinstance(gap, str) and gap == "min":
            gap = self.min_gap
        B0 = self.field(gap=gap)
        return total_power(
            ebeam_energy=self.ebeam_energy,
            peak_field=B0,
            undulator_L=self.length,
            sr_current=self.sr_current,
        )

    def power_density(self, gap="min", energy=None, **kwargs):
        """return power density (W/mrad**2); appraoximated for low K is approaximated !!"""
        if energy is not None:
            pars = self.find_harmonic_and_gap(
                energy, sort_harmonics=True, use_srw=use_srw
            )[0]
            gap = pars["gap"]
            harmonic = pars["harmonic"]

        B0 = self.field(gap=gap)
        K = self.k(gap)
        return power_density(
            ebeam_energy=self.ebeam_energy,
            peak_field=B0,
            undulator_n_period=self.N,
            k=K,
            sr_current=self.sr_current,
        )

    def _gap_scan(
        self,
        gapmin="min",
        gapmax=40,
        deltagap=0.001,
        calc="brilliance",
        harmonic=range(1, 13, 2),
    ):
        if gapmin == "min":
            gapmin = self.min_gap
        can_be_calculated = [
            "brilliance",
            "flux",
            "photon_beam_emittance",
            "degeneracy_number",
            "divh",
            "divv",
            "sh",
            "sv",
            "cof",
        ]
        if calc not in can_be_calculated:
            raise ValueError(f"calc has to be one of {can_be_calculated}")
        else:
            try:
                f = getattr(self, calc)
            except AttributeError:

                def func(gap=6, harmonic=1):
                    ph = self.photon_beam_characteristics(harmonic=harmonic, gap=gap)
                    return getattr(ph, calc)

                f = func
        energies = []
        values = []
        ret = {}
        for h in harmonic:
            ret[h] = {}
            for gap in np.arange(gapmin, gapmax + deltagap, deltagap):
                e = self.photon_energy(gap=gap, harmonic=h)
                ret[h][gap] = dict(energy=e, value=f(gap=gap, harmonic=h))
        return ret

    def gap_scan(
        self,
        gapmin="min",
        gapmax=20,
        deltagap=0.001,
        calc="brilliance",
        photon_energy_max=100,
        n_energy=1000,
        harmonic=range(1, 13, 2),
        merge_results=True,
        **kwargs,
    ):
        if gapmin == "min":
            gapmin = self.min_gap
        ret = self._gap_scan(
            gapmin=gapmin,
            gapmax=gapmax,
            deltagap=deltagap,
            calc=calc,
            harmonic=harmonic,
        )
        if not merge_results:
            return ret
        energies = []
        values = []
        for harmonic in ret.values():
            for gap in harmonic.values():
                if gap["energy"] < photon_energy_max:
                    energies.append(gap["energy"])
                    values.append(gap["value"])
        energies = np.asarray(energies)
        values = np.asarray(values)
        bins = np.logspace(
            np.log10(energies.min()),
            np.log10(max(energies.max(), photon_energy_max)),
            n_energy,
        )

        idx = np.digitize(energies, bins)
        best = np.ones(len(bins) - 1) * np.nan
        for i in range(len(bins) - 1):
            idx_temp = idx == i
            if np.sum(idx_temp) > 1:
                best[i] = np.max(values[idx_temp])
        bins_cen = (bins[:-1] + bins[1:]) / 2
        best = np.asarray(best)
        return bins_cen, best

    def find_harmonic_and_gap(
        self,
        photon_energy,
        photon_energy_relative_acceptance=0.01,
        sort_harmonics=False,
        use_srw=False,
        **kwargs,
    ):
        """if sort_harmonics is False it returns a tuple with index the (odd)
        harmonic (i.e. index 0 is harm 1, index 1 is harm 3, ...)
        if sort_harmonics is False the different harmonics are sorted by
        increasing flux
        """

        # find right harmonic
        found = dict()
        for harmonic in range(1, 21, 2):
            emin = self.photon_energy(gap="min", harmonic=harmonic)
            emax = self.photon_energy(gap=self.max_gap, harmonic=harmonic)
            if emin < photon_energy < emax:
                found[harmonic] = True
        max_flux = 0
        best = []  # for each harmonic
        for harmonic in found.keys():
            # prepare function
            def ediff(gap):
                return self.photon_energy(gap=gap, harmonic=harmonic) - photon_energy

            gaps = np.arange(self.min_gap, self.max_gap + 0.01, 0.01)
            chi2 = ediff(gaps) ** 2
            gap_guess = gaps[np.argmin(chi2)]
            from scipy import optimize

            try:
                gap = optimize.root_scalar(
                    ediff,
                    method="ridder",
                    bracket=(gap_guess - 0.03, gap_guess + 0.03),
                    xtol=1e-3,
                )
                gap = gap.root
                flux = self.flux(gap=gap, harmonic=harmonic)
                brilliance = self.brilliance(
                    gap=gap, harmonic=harmonic, use_srw=use_srw
                )
                best.append(
                    dict(harmonic=harmonic, gap=gap, flux=flux, brilliance=brilliance)
                )
            except Exception as e:
                print("Could not find gap, error was",e)
                pass
        if sort_harmonics:
            best = sorted(best, reverse=True, key=lambda x: x.get("brilliance"))
        return best

    def __str__(self):
        s = f"Undulator @ {self.elattice} lattice, period {self.period:.1f}mm, length {self.length}m, max K {self.k(gap=self.min_gap):.2f}"
        return s

    def __repr__(self):
        s = f"Undulator @ {self.elattice} lattice\nPeriod {self.period:.1f}mm, length {self.length}m, max K {self.k(gap=self.min_gap):.2f}"
        return s


class WolfryUndulator:
    def __init__(
        self, energy=7, npoints=400, undulator=Undulator(), ebeam=None, k="auto"
    ):

        try:
            from wofryimpl.propagator.util.undulator_coherent_mode_decomposition_1d import (
                UndulatorCoherentModeDecomposition1D,
            )
        except ImportError:
            print(
                "could not import wofryimpl, did you start within right conda environment ?"
            )
            return None
        if ebeam is None:
            ebeam = undulator.ebeam

        if isinstance(k, str) and k == "auto":
            K = undulator.k(energy=energy)
        else:
            K = k
        pars = dict(
            electron_energy=ebeam.ebeam_energy,
            electron_current=ebeam.sr_cur,
            undulator_period=undulator.period / 1e3,
            undulator_nperiods=undulator.N,
            K=K,
            photon_energy=energy * 1e3,
            abscissas_interval=0.00025,
            number_of_points=npoints,
            distance_to_screen=100,
            useGSMapproximation=False,
        )
        print(pars)
        # cmd = coherent mode decomposition
        self.cmd_h = UndulatorCoherentModeDecomposition1D(
            **pars,
            scan_direction="H",
            sigmaxx=ebeam.sh,
            sigmaxpxp=ebeam.divh,
        )

        self.cmd_v = UndulatorCoherentModeDecomposition1D(
            **pars, scan_direction="V", sigmaxx=ebeam.sv, sigmaxpxp=ebeam.divv
        )
        # make calculation
        self.cmd_h_res = None
        self.cmd_v_res = None

    def get_h(self, mode=0):
        if self.cmd_h_res is None:
            self.cmd_h_res = self.cmd_h.calculate()
        if isinstance(mode, int):
            return self.cmd_h.get_eigenvector_wavefront(mode)
        else:
            return [self.cmd_h.get_eigenvector_wavefront(i) for i in mode]

    def get_v(self, mode=0):
        if self.cmd_v_res is None:
            self.cmd_v_res = self.cmd_v.calculate()
        if isinstance(mode, int):
            return self.cmd_v.get_eigenvector_wavefront(mode)
        else:
            return [self.cmd_v.get_eigenvector_wavefront(i) for i in mode]


cpmu15_gael = Undulator(
    length=4,
    gap_to_b=b_field_cryoT,
    period=15,
    elattice="EBS",
    name="cpmu15",
    min_gap=5,
)

cpmu19 = Undulator(
    length=2,
    gap_to_b=b_field_cryoT,
    period=19,
    elattice="EBS",
    name="cpmu19",
    min_gap=6,
)
cpmu20 = Undulator(
    length=2,
    gap_to_b=b_field_cryoT,
    period=20,
    elattice="EBS",
    name="cpmu20",
    min_gap=6,
)
cpmu21 = Undulator(
    length=2,
    gap_to_b=b_field_cryoT,
    period=21,
    elattice="EBS",
    name="cpmu21",
    min_gap=6,
)


def get_cpmu(period=18, length=2, min_gap=6, minibeta=False, beta_v=1, beta_h=1):
    name = f"cpmu{period}"
    if minibeta:
        ebeam = beam.ebs_minibeta(beta_v=beta_v, beta_h=beta_h)
    else:
        ebeam = beam.e_beam("EBS")
    if min_gap == "auto":
        min_gap = max(3.5, 2.75 * np.sqrt(ebeam.betav + length**2 / 4 / ebeam.betav))
    return Undulator(
        length=length,
        gap_to_b=b_field_cryoT,
        period=period,
        elattice=ebeam,
        name=name,
        min_gap=min_gap,
    )


u17 = Undulator(
    length=2,
    gap_to_b=b_field_roomT,
    period=17,
    elattice="EBS",
    name="u17",
    min_gap=6,
    #    k=0.83,
)


_id10_u35a = functools.partial(
    b_field, halbach_coeff=2.0028, b_coeff=1.0015, period=34.984
)
id10_u35a = Undulator(
    length=1.6, gap_to_b=_id10_u35a, period=34.984, min_gap=10, name="u35a"
)

_id10_u35b = functools.partial(
    b_field, halbach_coeff=2.0786, b_coeff=1.0276, period=35.225
)
id10_u35b = Undulator(
    length=1.6, gap_to_b=_id10_u35b, period=35.225, min_gap=10, name="u35b"
)

_id10_u27b = functools.partial(
    b_field, halbach_coeff=2.0411, b_coeff=1.0237, period=27.203
)
id10_u27b = Undulator(
    length=1.6, gap_to_b=_id10_u27b, period=27.203, min_gap=10, name="u27b"
)

id10_u27b_highb = Undulator(
    elattice = "esrf_highb",
    length=1.6, gap_to_b=_id10_u27b, period=27.203, min_gap=10, name="u27b"
)



_id10_u27c = functools.partial(
    b_field, halbach_coeff=1.9084, b_coeff=1.0061, period=27.219
)
id10_u27c = Undulator(
    length=1.6, gap_to_b=_id10_u27c, period=27.219, min_gap=10, name="u27c"
)
