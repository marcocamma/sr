import numpy as np
from datastorage import DataStorage as ds

from . import beam as srbeam


def _sqrt_squares(x1, x2):
    return np.sqrt(x1 ** 2 + x2 ** 2)


class GSM:
    """
    based on Vartanyans and Singer N J. of Physics 2010
    doi: 10.1088/1367-2630/12/3/035004
    """

    def __init__(
        self, photon_beam=None, sh=None, sv=None, divh=None, divv=None, wavelength=None
    ):
        if photon_beam is None:
            photon_beam = srbeam.Photon_Beam()
        if sh is None:
            sh = photon_beam.sh
        if sv is None:
            sv = photon_beam.sv
        if divh is None:
            divh = photon_beam.divh
        if divv is None:
            divv = photon_beam.divv
        emitth = sh * divh
        emittv = sv * divv
        if wavelength is None:
            wavelength = photon_beam.wavelength
        if wavelength > 1e-6:
            wavelength = wavelength * 1e-10

        beam = ds(
            sh=sh,
            sv=sv,
            divh=divh,
            divv=divv,
            emitth=emitth,
            emittv=emittv,
            wavelength=wavelength,
        )
        self.beam = beam

        k = 2 * np.pi / wavelength
        # source coherence length ξ_{Sx,y}
        self.sclh = 2 * sh / np.sqrt(4 * k ** 2 * emitth ** 2 - 1)  # eq 33
        self.sclv = 2 * sv / np.sqrt(4 * k ** 2 * emittv ** 2 - 1)  # eq 33
        self.qh = self.sclh / sh
        self.qv = self.sclv / sv
        self.cofh = self.qh / np.sqrt(4 + self.qh ** 2)
        self.cofv = self.qv / np.sqrt(4 + self.qv ** 2)

        # cldiv = coherence length divergence
        self.cldivh = 1 / (2 * k * sh) * np.sqrt(4 + self.qh ** 2)
        self.cldivv = 1 / (2 * k * sv) * np.sqrt(4 + self.qv ** 2)

    def rms_size_at_dist(self, dist):
        b = self.beam
        k = 2 * np.pi / (b.wavelength)
        sh, dh = b.sh, b.divh
        sv, dv = b.sv, b.divv
        print(sh, b.divh * dist)
        return _sqrt_squares(sh, dh * dist), _sqrt_squares(sv, dv * dist)

    def rms_cl_at_dist(self, dist):
        b = self.beam
        k = 2 * np.pi / (b.wavelength)
        sh, dh = self.sclh, self.cldivh
        sv, dv = self.sclv, self.cldivv
        return _sqrt_squares(sh, dh * dist), _sqrt_squares(sv, dv * dist)


def speckle_size(
    wavelength=1e-10, det_dist=1, size_at_sample_or_sample_size=1e-6
):
    return det_dist * wavelength / size_at_sample_or_sample_size


def transversal_coherence_length(distance=100, wavelength=1e-10, source_size=30e-6):
    return distance * wavelength / source_size


def longitudinal_coherence_length(
    wavelength=1e-10, fractional_delta_wavelength=0.01, verbose=False
):
    lcl = wavelength / fractional_delta_wavelength / 2
    if verbose:
        msg = f"Calculating longitudinal coherence length, wavelength {wavelength*1e10} Ang and relative bandwidth {fractional_delta_wavelength:.3g}: {lcl:.4g} m"
        print(msg)
    return lcl


def fresnel_number(size_at_sample_or_sample_size=1e-6, det_dist=2, wavelength=1e-10):
    return size_at_sample_or_sample_size ** 2 / (det_dist * wavelength)


def is_in_far_field(
    size_at_sample_or_sample_size=10e-6,
    det_dist=2,
    wavelength=1e-10,
    verbose=False,
    fn_threshold=0.1,
):
    fn = fresnel_number(
        size_at_sample_or_sample_size=size_at_sample_or_sample_size,
        det_dist=det_dist,
        wavelength=wavelength,
    )
    if verbose:
        msg = f"Calculating Fresnel number for {size_at_sample_or_sample_size*1e6} um beam, det distance of {det_dist} m, wavelength {wavelength*1e10} Ang: {fn:.2f}"
        print(msg)
    return fn < fn_threshold


def max_optical_path_difference(
    size_at_sample_or_sample_size=10e-6,
    det_half_size=0.1,
    det_dist=1,
    angle_deg=0,
    verbose=False,
):
    # opl_saxs is due to finite size of detector
    detector_aperture = np.arctan(det_half_size / det_dist)
    # opl_waxs
    angle_rad = np.deg2rad(angle_deg)
    opl  = size_at_sample_or_sample_size*np.abs(np.sin(angle_rad))
    opl += size_at_sample_or_sample_size*np.abs(np.sin(detector_aperture))

    if verbose:
        msg = f"Calculating max optical path difference for {size_at_sample_or_sample_size*1e6} um beam, det half size {det_half_size} m at a distance of {det_dist} m: {opl:.3g} m"
        print(msg)
    return opl


def is_longitudinal_coherence_length_ok(
    wavelength=1e-10,
    fractional_delta_wavelength=0.01,
    size_at_sample_or_sample_size=10e-6,
    det_dist=1,
    det_half_size=0.1,
    angle_deg=0,
    verbose=False,
):
    op = max_optical_path_difference(
        size_at_sample_or_sample_size=size_at_sample_or_sample_size,
        det_half_size=det_half_size,
        det_dist=det_dist,
        verbose=verbose,
        angle_deg=angle_deg,
    )
    lcl = longitudinal_coherence_length(
        wavelength=wavelength,
        fractional_delta_wavelength=fractional_delta_wavelength,
        verbose=verbose,
    )
    return lcl > op


def cdi_resolution(wavelength=1e-10, det_half_size=0.1, det_dist=1, verbose=False):
    res = wavelength /2/ np.arctan(det_half_size / det_dist)
    if verbose:
        msg = f"Calculating cdi resolution for wavelength {wavelength*1e10} Ang, det half size {det_half_size:.2f} @ distance of {det_dist} m: {res:.3g} m"
        print(msg)
    return res


def is_cdi_oversampling_ok(
    wavelength=1e-10,
    det_dist=1,
    size_at_sample_or_sample_size=1e-6,
    pixel_size=100e-6,
    oversampling=3,
    verbose=False,
):
    speckle_size = det_dist * wavelength / size_at_sample_or_sample_size
    if verbose:
        os = speckle_size / pixel_size
        msg = f"Calculating cdi oversampling for wavelength {wavelength*1e10} Ang, det distance of {det_dist} m, beam sizeat sample of {size_at_sample_or_sample_size*1e6:.2f} um and pixel size of {pixel_size*1e6} um: {os:.2g}"
        print(msg)
    return oversampling * pixel_size < speckle_size

def is_xpcs_oversampling_ok(
    wavelength=1e-10,
    det_dist=1,
    size_at_sample_or_sample_size=1e-6,
    pixel_size=100e-6,
    oversampling=1,
    verbose=False,
):
    return is_cdi_oversampling_ok(
            wavelength=wavelength,
            det_dist=det_dist,
            size_at_sample_or_sample_size=size_at_sample_or_sample_size,
            pixel_size=pixel_size,
            oversampling=oversampling,
            verbose=verbose)



# source 1,0.02 (0)
# PS 0.075x0.25 (27 m)
# SS 0.1x0.25 (34 m)
# TS 0.15x0.25 (56 m)
id10_slits_h = ((0, 1000), (27, 75), (34, 100), (56, 150))
id10_slits_v = ((0, 20), (27, 250), (34, 250), (56, 250))


id18_slits_h = ((0, 141), (56, 150))
id18_slits_v = ((0, 20), (56, 250))


def slits_divergence(slits=((0, 30), (40, 30)), return_all=False):
    """ 
    list of list ( (pos1_m,size_1_um), (pos2_m,size_2_um), ...)
    For gaussian beams best seems to put 2*FWHM as size
    """
    nslits = len(slits)
    slit_pos = [slit[0] for slit in slits]
    slit_half_size = [slit[1] / 2 for slit in slits]
    divergence = dict()
    best = 1e10
    best_slits = None
    for i0 in range(nslits - 1):
        for i1 in range(i0 + 1, nslits):
            div = (slit_half_size[i1] + slit_half_size[i0]) / abs(
                slit_pos[i1] - slit_pos[i0]
            )
            divergence["%d-%d" % (i0, i1)] = div
            if div < best:
                best = div
                best_slits = "%d-%d" % (i0, i1)
    if return_all:
        return divergence
    else:
        return best_slits, best


def test_slit_position(required_div_urad=(1, 2, 3, 4)):
    import beam
    from matplotlib import pylab as plt

    if isinstance(required_div_urad, (float, int)):
        required_div_urad = (required_div_urad,)
    required_div = [r * 1e-6 for r in required_div_urad]
    fig, axes = plt.subplots(
        2, len(required_div), sharex=True, sharey="row", squeeze=False
    )
    b = beam.Photon_Beam()
    D = np.arange(0, 200, 0.1)  # in m
    beam_size_h = b.rms_size_at_dist(D).sh * 2.35
    beam_size_v = b.rms_size_at_dist(D).sv * 2.35
    source_size_h = (
        b.sh * 2.35 * 2
    )  # the factor 2 is needed to have the FWHM divergence
    source_size_v = b.sv * 2.35 * 2
    for i, d in enumerate(required_div):
        slit_h = d * D - source_size_h
        slit_v = d * D - source_size_v
        slit_h[slit_h < 0] = np.nan
        slit_v[slit_v < 0] = np.nan
        fraction_h = slit_h / beam_size_h
        fraction_v = slit_v / beam_size_v
        axes[0, i].set_title(f"Resulting beam\ndivergence {d*1e6} (μrad)")
        axes[0, i].plot(D, slit_h * 1e6, label="slit size h")
        axes[0, i].plot(D, slit_v * 1e6, label="slit size v")
        axes[0, i].plot(D, beam_size_h * 1e6, "--", label="beam size h")
        axes[0, i].plot(D, beam_size_v * 1e6, "--", label="beam size v")
        axes[1, i].plot(D, fraction_h, label="fraction h")
        axes[1, i].plot(D, fraction_v, label="fraction v")
    axes[0, 0].set_ylabel("Slit size &\nBeam size (μm)")
    axes[1, 0].set_ylabel("Fraction transmitted\nthrough slit")
    for a in axes.ravel():
        a.grid()
    for a in axes[1, :]:
        a.set_xlabel("Position along\ethe beamline (m)")
    axes[0, -1].legend()
    axes[1, -1].legend()

def diffraction_limited_spot(wavelength=1e-10,focal_length=10,acceptance=100e-6):
    return 0.88*wavelength*focal_length/acceptance

def diffraction_limited_spot_gaussian(wavelength=1e-10,focal_length=10,fwhm=100e-6):
    """
    returns FWHM of focused beam
    https://www.rp-photonics.com/focal_length.html
    formula is w_focus = λ*f/(π*w)
    w = 2*σ = 2/2.35*FWHM = 0.85*FWHM
    so the formula becomes
    0.85*FWHM_focus = λ*f/(π*0.85*FWHM) →
    FWHM_focus = λ*f/FWHM * 1/(π*0.85*0.85) = 0.44*λ*f/FWHM
    """
    return 0.44*wavelength*focal_length/fwhm


def total_fraction_scattered_intensity(R=1e-6,rho=0.4,wavelength=1e-10):
    """
    calculated for sphere with constant e-density
    rho = electron density per Å³ (protein 0.43, water 0.33)
    R = radius in m
    """
    rho = rho*1e30 # convert to m3
    return 4.9*1e-25*wavelength**2*rho**2*R**4
