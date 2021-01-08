"""
Trying to reproduce calculations from 
Singera & Vartanyants
Coherence properties of focused X-ray beams athigh-brilliance synchrotron sources
J.SynchrotronRad.(2014) 21, 5–15 doi:10.1107/S160057751302385
"""
import sys
sys.path.insert(0,"../../../../../")
import sympy
import numpy as np

sympy.init_printing(use_unicode=True, use_latex=True, pretty_printing=True)
from datastorage import DataStorage as ds
from sympy import Symbol
from sr.abcd.optics import GSM
from sr.abcd.optics import get_symbol, find_roots, lens, free_space, gaussian_aperture

from matplotlib import pyplot as plt
from functools import lru_cache


def source():
    gsmh = GSM(rms_size=36.2e-6, rms_cl=0.9e-6, wavelen=12.4 * 1e-10 / 8, auto_apply_evalf = True)
    gsmv = GSM(rms_size=6.3e-6, rms_cl=7.7e-6, wavelen=12.4 * 1e-10 / 8, auto_apply_evalf = True)
    return gsmh, gsmv


@lru_cache(maxsize=2048)
def propagate(b0, aperture=20e-6, as_numpy=True):

    # optics is at 85, from source
    zb = free_space(85)

    # equivalent FL = 2.13
    F = lens(2.13)

    a = gaussian_aperture(aperture)
    δ = sympy.Symbol("δ", real=True)
    za = free_space(2.13 + δ)

    b1 = b0.apply(zb)
    div = b0.divergence.evalf() * 1e6
    print(f"Divergence @ source: {div:.2f} (μrad)")
    b2 = b1.apply(a)
    b3 = b2.apply(F)
    s1 = b1.rms_size.evalf() * 1e6
    s2 = b2.rms_size.evalf() * 1e6
    print(f"RMS size before aperture: {s1:.2f} (μm)")
    print(f"RMS size after  aperture: {s2:.2f} (μm)")
    b4 = b3.apply(za)
    if as_numpy:
        size = b4.rms_size.evalf()
        size = sympy.lambdify(size.free_symbols, size)
        cl = b4.rms_cl.evalf()
        cl = sympy.lambdify(cl.free_symbols, cl)
    return ds(size=size, cl=cl)


def show_effect_pinhole(axis="h"):
    h, v = source()

    if axis == "v":
        b = v
        ylabel = "Vertical Direction\ny (μm)"
    else:
        b = h
        ylabel = "Horizontal Direction\nx (μm)"

    b25 = propagate(b, aperture=25e-6)
    b100 = propagate(b, aperture=100e-6)
    z = np.linspace(-0.75, 0.75, 376)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    y = np.linspace(-20, 20, 201)

    def igauss(s):
        return 1 / np.sqrt(2 * np.pi) / s * np.exp(-y ** 2 / 2 / s ** 2)

    # bf = beam around focus
    for i, bf in enumerate([b25, b100]):
        s = bf.size(z) * 1e6
        cl = bf.cl(z) * 1e6
        sm = s.min()
        clm = cl.min()
        title = f"Size and CL at focus [μm]: {sm:.2f},{clm:.2f}"
        I = np.asarray([igauss(_s) for _s in s]).T
        ax[i].pcolormesh(z, y, I, shading="gouraud")
        ax[i].plot(z, +cl / 2, "--", color="0.5")
        ax[i].plot(z, -cl / 2, "--", color="0.5")
        ax[i].set_title(title)
    ax[1].set_ylim(-16, 16)
    for a in ax:
        a.grid()
        a.set_ylabel(ylabel)
    ax[1].set_xlabel("Distance (m)")


def do_all():
    print("To compare with Singera & Vartanyants, JSR 2014, fig6, fig7 and table 4")
    show_effect_pinhole(axis="h")
    plt.savefig("p10_hor.pdf", transparent=True)
    show_effect_pinhole(axis="v")
    plt.savefig("p10_ver.pdf", transparent=True)


if __name__ == "__main__":
    do_all()
