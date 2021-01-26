import itertools
import copy
import sympy
from datastorage import DataStorage as ds
from .materials import get_delta_beta, attenuation_length
from .utils.conversion import energy_to_wavelength, wavelength_to_energy
from .utils.unicode import times as TIMES
from .utils.unicode import mu as MU
from .abcd.optics import GSM
from .abcd.optics import GSM_Numeric
import numpy as np
from numpy import real
import datetime
import os


DEFAULT_GSM = GSM_Numeric(wavelen=1e-10, rms_size=15e-6, rms_cl=2e-6).propagate(50)


def _calc_sum_inverse(args):
    s = sum([1 / a for a in args])
    if isinstance(s,sympy.Float): s = float(s)
    return 1 / s


def _calc_sum_square_inverse(args):
    s = sum([1 / a ** 2 for a in args])
    if isinstance(s,sympy.Float): s = float(s)
    return np.sqrt(1 / s)


def focal_length_single_lens(E, radius, material="Be", density=None):
    """ returns the focal length for a single lens f=r/2/delta """
    delta, beta = get_delta_beta(material, density=density, energy=E)
    f = radius / 2.0 / delta
    return f


class LensBlock:
    def __init__(
        self,
        n=1,
        radius=250e-6,
        material="Be",
        density=None,
        thickness=1e-3,
        web_thickness=30e-6,
    ):
        """ a LensBlock is a number of identical lenses """
        self.n = n
        self.radius = radius
        self.material = material
        self.density = density
        self.thickness = thickness
        self.web_thickness = web_thickness
        # simple geometry y=a*x**2 (R=1/(2a)) gives geometrical opening
        aperture = 2 * np.sqrt((thickness - web_thickness) * radius)
        self._aperture = aperture

    def focal_length(self, energy=10):
        m, d = self.material, self.density
        delta, _ = get_delta_beta(m, density=d, energy=energy)
        return self.radius / (2 * delta * self.n)

    def transmission_central_ray(self, energy=10):
        if self.web_thickness == 0:
            if isinstance(energy, np.ndarray):
                return np.ones_like(energy)
            else:
                return 1
        l = attenuation_length(self.material, density=self.density, energy=energy)
        total_l = self.n * self.web_thickness
        return np.exp(-total_l / l)

    def transmission_gaussian_beam(self, energy=10, gauss_beam_fwhm=100e-6):
        """
        considering finite opening (integral of 2*pi*r*I_{transmitted}
        from 0 to R0) calculated with sympy (file CRL_transmission.py)
        """
        R = self.radius
        l0 = attenuation_length(self.material, density=self.density, energy=energy)
        d = self.web_thickness
        t = self.thickness
        sig = gauss_beam_fwhm / 2.35
        N = self.n
        exp = np.exp
        transmission = (
            R
            * l0
            * (1 - exp((d - t) * (2 * N * sig ** 2 + R * l0) / (2 * l0 * sig ** 2)))
            * exp(-N * d / l0)
            / (2 * N * sig ** 2 + R * l0)
        )
        return transmission

    def absorption_opening(self, energy=10):
        m, d = self.material, self.density
        _, beta = get_delta_beta(m, density=d, energy=energy)
        w = energy_to_wavelength(energy) * 1e-10
        op2 = self.radius * w / (8 * np.pi * self.n * beta)
        return np.sqrt(op2)

    def aperture(self):
        return self._aperture

    def __str__(self):

        return "%d%s%s lenses, r=%.1f μm, D=%.1f mm (2R_0=%.0f μm)" % (
            self.n,
            TIMES,
            self.material,
            self.radius * 1e6,
            self.thickness * 1e3,
            self.aperture() * 1e6,
        )

    def __repr__(self):
        return self.__str__()


class LensSet:
    def __init__(self, lens_set, material="Be",pinhole=None):
        """
        lens_set = ( LensBlock1, LensBlock2, ....)
        OR
        lens_set = ( (n1,radius1,thick1),
                      n2,radius2,thick2),...) 
        thick is the thickness of each lens (1mm if not given)

        material is NOT used if lens_set is defined as list of LensBlock

        pinhole is an hard aperture
        """
        if isinstance(lens_set, (int, float)):
            lens_set = ((1, lens_set, 0),)

        if len(lens_set) > 0 and not isinstance(lens_set[0], LensBlock):
            temp = []
            for lensblock in lens_set:
                n = lensblock[0]
                r = lensblock[1]
                t = lensblock[2] if len(lensblock) > 2 else 1e-3
                lb = LensBlock(n=n, radius=r, thickness=t, material=material)
                temp.append(lb)
            lens_set = temp
        self.lens_set = lens_set
        self.pinhole = pinhole

    def transmission_central_ray(self, energy=10):
        if isinstance(energy, np.ndarray):
            T = np.ones_like(energy)
        else:
            T = 1
        for lb in self.lens_set:
            T *= lb.transmission_central_ray(energy=energy)
        return T

    def transmission_gaussian_beam(self, energy=10, gauss_beam_fwhm=100e-6):
        if isinstance(energy, np.ndarray):
            T = np.ones_like(energy)
        else:
            T = 1
        for lb in self.lens_set:
            T *= lb.transmission_gaussian_beam(
                energy=energy, gauss_beam_fwhm=gauss_beam_fwhm
            )
        return T

    def aperture(self):
        if len(self.lens_set) == 0:
            a = 1
        else:
            a = min([lb.aperture() for lb in self.lens_set])
        if self.pinhole is not None:
            a = min(a,self.pinhole)
        return a

    def gaussian_aperture(self):
        """ see singer&vartanyans JSR 2014 appendix A"""
        return self.aperture() / 4.55

    def focal_length(self, energy=10):
        if len(self.lens_set) == 0:
            return np.inf
        else:
            return _calc_sum_inverse([lb.focal_length(energy) for lb in self.lens_set])

    def absorption_opening(self, energy=10):
        if len(self.lens_set) == 0:
            return 1
        else:
            return _calc_sum_square_inverse(
                [lb.absorption_opening(energy) for lb in self.lens_set]
            )

    def calc_focusing(
        self,
        energy,
        distance=4,
        source_distance=150,
        fwhm_beam=500e-6,
        slit_opening=4e-3,
        verbose=True,
    ):
        # calculate effective beamsize
        input_rms_beam = fwhm_beam / 2.35
        gauss_slit_opening = slit_opening / 4.55  # see single&vartanyans JSR 2014
        aperture = min(self.gaussian_aperture(), gauss_slit_opening)
        absorption_ap = self.absorption_opening(energy)
        lens_effective_aperture = _calc_sum_square_inverse([aperture, absorption_ap])
        rms_beam = _calc_sum_square_inverse([input_rms_beam, lens_effective_aperture])

        fl = self.focal_length(energy)

        # distance_at which it will focus (1/a+1/b=1/f)
        focus_distance = source_distance * fl / (source_distance - fl)

        lam = energy_to_wavelength(energy) * 1e-10

        w_unfocused = rms_beam * 2
        # assuming gaussian beam divergence =
        # = w_unfocused/focus_distance we can obtain
        waist = lam / np.pi * focus_distance / w_unfocused
        waist_fwhm = waist * 2.35 / 2.0
        rayleigh_range = np.pi * waist ** 2 / lam
        size = waist * np.sqrt(
            1.0 + (distance - focus_distance) ** 2.0 / rayleigh_range ** 2
        )
        fwhm_at_dist = size / 2 * 2.35
        t = self.transmission_central_ray(energy)
        res = ds(
            focal_length=fl,
            focus_distance=focus_distance,
            distance=distance,
            fwhm_at_dist=fwhm_at_dist,
            fwhm_unfocused=fwhm_beam,
            fwhm_at_waist=waist_fwhm,
            rayleigh_range=rayleigh_range,
            energy=energy,
            lens_set=self.lens_set,
            transmission_central_ray=t,
        )

        if verbose:
            print("beam FWHM @ lens     : %.3e" % (input_rms_beam * 2.35))
            print("Lens set opening     : %.3e" % lens_effective_aperture)
            print("beam FWHM after lens : %.3e" % (rms_beam * 2.35))
            print("------------------------")
            print("focus distance       : %.3e" % focus_distance)
            print("focal length         : %.3e" % fl)
            print("------------------------")
            print("waist                : %.3e" % waist)
            print("waist FWHM           : %.3e" % waist_fwhm)
            print("rayleigh_range       : %.3e" % rayleigh_range)
            print("------------------------")
            print("size @ dist          : %.3e" % size)
            print("size FWHM @ dist     : %.3e" % fwhm_at_dist)
        return res

    def calc_focusing_GSM(
        self, gsm=DEFAULT_GSM, source_distance=None, slit_opening=4e-3, verbose=True,
    ):
        """ Based on Singer&Vartanyans JSR 2014 """
        if source_distance is None:
            gsm_at_lens = gsm
        else:
            gsm_at_lens = gsm.propagate(source_distance)
        if isinstance(gsm_at_lens,GSM): gsm_at_lens.evalf()

        input_rms_beam = float(gsm_at_lens.rms_size)
        input_R = float(gsm_at_lens.radius)
        input_cl = float(gsm_at_lens.rms_cl)

        energy = wavelength_to_energy(gsm_at_lens.wavelen * 1e10)
        k = 2 * np.pi / gsm_at_lens.wavelen

        # calculate lens opening (called Ω in paper)
        gauss_slit_opening = slit_opening / 4.55
        aperture = min(self.gaussian_aperture(), gauss_slit_opening)
        abs_opening = self.absorption_opening(energy)
        lens_effective_aperture = _calc_sum_square_inverse([aperture, abs_opening])

        fl = self.focal_length(energy)
        # tilde_ are values at lens
        tilde_Sigma = _calc_sum_square_inverse(
            [input_rms_beam, lens_effective_aperture]
        )
        tilde_R = _calc_sum_inverse([input_R, -fl])
        tilde_cl = input_cl  # Coherence length is not modified by the lens
        # gdc = global_degree_of_coherence
        tilde_gdc = 1 / np.sqrt(1 + (2 * tilde_Sigma / tilde_cl) ** 2)

        Z_L = 2 * k * tilde_Sigma ** 2 * tilde_gdc

        # distance_at which it will focus (1/a+1/b=1/f)
        focus_distance = -tilde_R / (1 + (tilde_R / Z_L) ** 2)
        focus_rms_size = tilde_Sigma / np.sqrt(1 + (Z_L / tilde_R) ** 2)
        focus_cl = tilde_cl / np.sqrt(1 + (Z_L / tilde_R) ** 2)
        rayleigh_range = 4 * k * focus_rms_size ** 2 * tilde_gdc

        t = self.transmission_central_ray(energy)

        if isinstance(gsm_at_lens,GSM):
            beam_at_focus = GSM(wavelen=gsm_at_lens.wavelen,
                rms_size=focus_rms_size,
                rms_cl=focus_cl,
                auto_apply_evalf=gsm_at_lens.auto_apply_evalf,
                )
        else:
            beam_at_focus = GSM_Numeric(wavelen=gsm_at_lens.wavelen,
                rms_size=focus_rms_size,
                rms_cl=focus_cl,
                )

        gsm_at_lens = beam_at_focus.propagate(-focus_distance)


        res = ds(
            focal_length=fl,
            focus_distance=focus_distance,
            fwhm_unfocused=input_rms_beam * 2.35,
            fwhm_at_waist=focus_rms_size * 2.35,
            cl_fwhm_at_waist=focus_cl * 2.35,
            rayleigh_range=rayleigh_range,
            energy=energy,
            lens_set=self.lens_set,
            transmission_central_ray=t,
            gsm_at_focus=beam_at_focus,
            gsm_at_lens=gsm_at_lens
        )

        if verbose:
            print("FWHM   @ lens        : %.3e" % (input_rms_beam * 2.35))
            print("RMS    @ lens        : %.3e" % input_rms_beam)
            print("RMS CL @ lens        : %.3e" % input_cl)
            print(
                "GDC    @ lens        : %.3e"
                % float(gsm_at_lens.global_degree_of_coherence)
            )
            print("--------------------------------")
            print("focus distance       : %.3e" % focus_distance)
            print("focal length         : %.3e" % fl)
            print("Lens set abs opening : %.3e" % abs_opening)
            print("Lens set opening     : %.3e" % lens_effective_aperture)
            print("beam FWHM after lens : %.3e" % (tilde_Sigma * 2.35))
            print("--------------------------------")
            print("FWHM   @ waist       : %.3e" % (focus_rms_size * 2.35))
            print("RMS    @ waist       : %.3e" % focus_rms_size)
            print("RMS CL @ waist       : %.3e" % focus_cl)
            print("GDC    @ waist       : %.3e" % tilde_gdc)
            print("rayleigh_range       : %.3e" % rayleigh_range)
        return res

    def find_energy(
        self, distance=4.0, source_distance=150, slit_opening=4e-3, fwhm_beam=500e-6
    ):

        """ 
        finds the energy that would focus at a given distance (default = 4m)
        """

        def calc(E):
            return self.calc_focusing(
                E,
                distance=distance,
                source_distance=source_distance,
                verbose=False,
                fwhm_beam=fwhm_beam,
            ).focus_distance

        Emin = 1.0
        Emax = 24.0
        E = (Emax + Emin) / 2.0
        absdiff = 100
        while absdiff > 0.001:
            dmin = calc(Emin)
            dmax = calc(Emax)
            E = (Emax + Emin) / 2.0
            # rint(E)
            d = calc(E)
            if (distance < dmax) and (distance > d):
                Emin = E
            elif (distance > dmin) and (distance < d):
                Emax = E
            else:
                print("somehow failed ...")
                break
            absdiff = abs(distance - d)
        print("Energy that would focus at a distance of %.3f is %.3f" % (distance, E))
        b = self.calc_focusing(E, distance=distance, source_distance=source_distance)
        return b

    def __repr__(self):
        if len(self.lens_set) == 0:
            return "No lenses"
        else:
            return "\n".join([lb.__str__() for lb in self.lens_set])


def dec2TrueFalse(n, npos=None):
    c = bin(n)[2:]  # remove 'header' 0b
    ret = list(map(bool, map(int, c)))
    if npos is not None and npos > len(c):
        ret = [False,] * (npos - len(c)) + ret
    return ret


class Transfocator:
    def __init__(self, lens_set):
        if not isinstance(lens_set, LensSet):
            lens_set = LensSet(lens_set)
        blocks = lens_set.lens_set
        self.lens_set = blocks
        nblocks = len(blocks)
        all_confs = []
        all_sets = []
        for i in range(0, 1 << nblocks):
            conf = dec2TrueFalse(i, npos=nblocks)
            all_confs.append(conf)
            lens = [b for b, c in zip(blocks, conf) if c]
            all_sets.append(LensSet(lens))
        self.all_sets = np.asarray(all_sets)
        self.all_confs = np.asarray(all_confs)
        n_tot = sum([s.n for s in self.lens_set])
        self.n_lenses_tot = n_tot

    def find_best_set_for_focal_length(
        self, energy=8, focal_length=10, accuracy_needed=0.1, verbose=False,
        beam_fwhm = None
    ):
        """
        find lensset that has a certain focal_length (within accuracy_needed)
        and transmission if beam_fwhm is provided (and sort_by_transmission is
        """
        fl = [s.focal_length(energy) for s in self.all_sets]
        fl = np.asarray(fl)
        delta_fl = np.abs(fl - focal_length)
        if fl[np.isfinite(fl)].max() < focal_length * 5:
            idx_best = 0
            idx_good = np.ones_like(fl, dtype=bool)
        else:
            idx_best = np.argmin(delta_fl)
            idx_good = delta_fl < accuracy_needed
        if idx_good.sum() == 0:
            if verbose:
                print(
                    f"Could not find good set within required accuracy ({accuracy_needed:.3f}); will try with factor 2 bigger"
                )
            return self.find_best_set_for_focal_length(
                energy=energy,
                focal_length=focal_length,
                accuracy_needed=2 * accuracy_needed,
                verbose=verbose
            )

        good_lensets = self.all_sets[idx_good]
        transmission = [g.transmission_central_ray(energy) for g in good_lensets]
        if beam_fwhm is not None:
            transmission_gauss_beam = [g.transmission_gaussian_beam(energy,gauss_beam_fwhm=beam_fwhm) for g in good_lensets]
            idx_best = np.argmax(transmission_gauss)
        else:
            idx_best = np.argmin(delta_fl[idx_good])
            transmission_gauss_beam = None
        # deep copying best_lens_set in case it is modified as return value
        ret = ds(
            in_out=self.all_confs[idx_good][idx_best],
            focal_length=self.all_sets[idx_good][idx_best].focal_length(energy),
            best_lens_set=copy.deepcopy(self.all_sets[idx_good][idx_best]),
            all_delta_fl=delta_fl,
            good_lensets=self.all_sets[idx_good],
            transmission_central_ray=transmission[idx_best],
        )
        return ret

    def __repr__(self):
        lenses = self.lens_set
        s = f"Transfocator with {len(lenses)} axis"
        for l in lenses:
            s += f"\n - {str(l)}"
        return s


#        lens_sets = [LensSet(s) for s in possible_lens_set]
#        self.lens_sets = lens_sets


def findEnergy(lens_set, distance=4.0, source_distance=150, material="Be"):
    """ usage findEnergy( (2,200e-6,4,500e-6) ,distance =4 )
      finds the neergy that would focus at a given distance (default = 4m)
  """
    lens_set = LensSet(lens_set, material=material)

    def calc(E):
        return lens_set.calc_focusing(
            E, distance=distance, source_distance=source_distance, verbose=False
        )

    Emin = 1.0
    Emax = 24.0
    E = (Emax + Emin) / 2.0
    absdiff = 100
    while absdiff > 0.001:
        dmin = calc(Emin).focus_distance
        dmax = calc(Emax).focus_distance
        E = (Emax + Emin) / 2.0
        print(E)
        d = calc(E).focus_distance
        if (distance < dmax) and (distance > d):
            Emin = E
        elif (distance > dmin) and (distance < d):
            Emax = E
        else:
            print("somehow failed ...")
            break
        absdiff = abs(distance - d)
    print("Energy that would focus at a distance of %.3f is %.3f" % (distance, E))
    b = lens_set.calc_focusing(E, distance=distance, source_distance=source_distance)
    return b


def test_p10():
    from .abcd import useful_beams

    h = useful_beams.p10h
    v = useful_beams.p10v
    ls = LensSet([[1, 200e-6], [2, 50e-6]])

    # In Singer&Vartanyans JSR 2014 article, the opening is given as
    # 'Gaussian opening' this is the reason of the 4.55 factor below
    for s in [25, 100]:
        print("\nP10 Horizontal focusing %s %sm slit" % (s, MU))
        ls.calc_focusing_GSM(h, source_distance=85, slit_opening=s * 1e-6 * 4.55)

    for s in [50, 150]:
        print("\nP10 Vertical focusing %s %sm slit" % (s, MU))
        ls.calc_focusing_GSM(v, source_distance=85, slit_opening=s * 1e-6 * 4.55)


l_1x5mm = LensBlock(1, radius=5000e-6)
l_1x2mm = LensBlock(1, radius=2000e-6)
l_1x1mm = LensBlock(1, radius=1000e-6)
l_1x500um = LensBlock(1, radius=500e-6)
l_2x500um = LensBlock(2, radius=250e-6)
l_4x500um = LensBlock(4, radius=250e-6)
l_8x500um = LensBlock(8, radius=250e-6)
l_16x500um = LensBlock(16, radius=250e-6)
l_Al_1x500um = LensBlock(1, radius=250e-6, material="Al")
l_Al_2x500um = LensBlock(2, radius=250e-6, material="Al")
l_Al_4x500um = LensBlock(4, radius=250e-6, material="Al")

lens_set = LensSet(
    (
        l_1x5mm,
        l_1x2mm,
        l_1x1mm,
        l_1x500um,
        l_2x500um,
        l_4x500um,
        l_8x500um,
        l_16x500um,
        l_Al_1x500um,
        l_Al_2x500um,
        l_Al_4x500um,
    )
)
t = Transfocator(
    (
        l_1x5mm,
        l_1x2mm,
        l_1x1mm,
        l_1x500um,
        l_2x500um,
        l_4x500um,
        l_8x500um,
        l_16x500um,
        l_Al_1x500um,
        l_Al_2x500um,
        l_Al_4x500um,
    )
)
