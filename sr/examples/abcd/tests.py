"""
Notes: if some expressions do not simplify why they should they might 
have 'duplicates'. remove them by using remove_duplicates
"""
import numpy as np
import sympy.physics.optics as so
import sympy

from matplotlib import pyplot as plt
try:
    from sr import abcd
except ImportError:
    import sys
    sys.path.insert(0,"../../../")
    from sr import abcd


sympy.init_printing(use_unicode=True, use_latex=True, pretty_printing=True)
from datastorage import DataStorage as ds
from sympy import Symbol
from abcd import GSM
from abcd import GaussBeam
from abcd import get_symbol, find_roots, lens, free_space

from matplotlib import pyplot as plt


def test_collimate_gauss():

    g1 = GaussBeam(wavelen="λ", rms_size="σ_0")

    z0m = free_space("z_0")
    z0 = z0m.B
    print("Defining generic Gaussin beam", str(g1))

    l = lens("F", force_positive=True)
    print("Putting lens of focal length F")

    g2 = gs.apply_matrix(l * z0m)
    print(
        "Divergence of gauss beam after lens (F) @ dist z_0", g2.divergence.simplify()
    )
    sol = find_roots(g2.divergence, z0, use_derivative=True)
    print("Distance z_0 that minimizes divergence z_0=", sol[0])

    print("New divergence is", sol[1].simplify())


def test_image_gauss():
    g1 = GaussBeam(wavelen="λ", rms_size="σ_0")

    z0m = free_space("z_0")
    z0 = z0m.B
    l = lens("f")
    z1m = free_space("z_1")
    z1 = z1m.B

    g2 = gs.apply_matrix(z1m * l * z0m)

    magnification = (g2.rms_size / g1.rms_size).simplify()
    print("Magnification:", magnification.simplify())

    # find best focus
    sol = find_roots(magnification, z1, use_derivative=True)

    dist_to_focus = sol[0]
    print("Position of image", str(dist_to_focus.simplify()))
    print("1-z1/F=", (1 - dist_to_focus / l.C).simplify())

    return g1, g2, magnification, dist_to_focus


def test_collimate_GSM():
    print("Defining default Gaussian and Gaussian-Schell beams")
    gs = GSM()
    g = GaussBeam()

    z0m = free_space("z_0")
    z0 = z0m.B
    # F  = lens(z_0.B)
    F = lens("F", force_positive=True)
    # z1 = free_space('z1')

    print("Create ABCD matrix z_0 translation + lens F")
    M = F * z0m

    print("Applying matrix to both beams")
    gs1 = gs.apply_matrix(M)
    g1 = g.apply_matrix(M)

    print("Find best z_0 to minimize divergence of resulting Gaussian Beam")
    d1 = find_roots(g1.divergence, z0, use_derivative=True)
    print("Best z0 = ", str(d1[0]), "(divergence ", str(d1[1]), ")")

    print("Find best z_0 to minimize divergence of resulting Gaussian-Schell Beam")
    ds1 = find_roots(gs1.divergence, z0, use_derivative=True)
    print("Best z0 = ", str(d1[0]), "(divergence ", str(d1[1]), ")")
    return gs1, g1


def test_image_GSM():
    z_0 = sympy.Symbol("z_0", real=True, positive=True)
    z_1 = sympy.Symbol("z_1", real=True, positive=True)
    gs = GSM()
    g = GaussBeam()

    z0m = free_space(z_0, force_positive=True)
    z1m = free_space(z_1, force_positive=True)
    F = lens("F", force_positive=True)

    # z1 = free_space('z1')

    M = z1m * F * z0m

    gs1 = gs.apply_matrix(M)
    g1 = g.apply_matrix(M)

    # magnification
    m_gs = gs1.rms_size / gs.rms_size
    m_g = g1.rms_size / g.rms_size
    m_gs = m_gs.simplify()
    m_g = m_g.simplify()
    print("Gauss magnification", m_g)
    print("GSM   magnification", m_gs)
    print("Limit of GSM for ξ→0", m_gs.limit(gs.rms_cl, 0))
    print("Limit of GSM for ξ→inf", m_gs.limit(gs.rms_cl, sympy.oo))
    return m_gs, m_g


def test_image_GSM_numeric():
    gs = GSM(wavelen=1e-10, rms_waist_size=3e-6, rms_waist_cl=2e-6)

    z0m = free_space(60)
    F = lens(15)

    # z1 = free_space('z1')

    M = F * z0m

    gs1 = apply_matrix_GSM(M, gs)

    return gs1


def test_collimate_undulator():

    LPOS = 30
    DZ = 1
    gsh = GSM(wavelen=1.6e-10, rms_size=30e-6, rms_cl=3.1e-6)

    gsv = GSM(wavelen=1.6e-10, rms_size=5.4e-6, rms_cl=3.8e-6)

    clh = []
    sh = []
    clv = []
    sv = []
    z = []

    for zi in np.arange(0, LPOS, DZ):
        z.append(zi)
        d = free_space(zi)
        # cl.append(gs.rms_cl(zi).evalf()*1e6)
        gh = gsh.apply_matrix(d)
        sh.append(gh.rms_size.evalf() * 1e6)
        clh.append(gh.rms_cl.evalf() * 1e6)
        gv = gsv.apply_matrix(d)
        sv.append(gv.rms_size.evalf() * 1e6)
        clv.append(gv.rms_cl.evalf() * 1e6)

    z0m = free_space(LPOS)
    F = lens(LPOS)

    for zi in np.arange(LPOS, 200, DZ):
        z1m = free_space(zi - LPOS)
        M = z1m * F * z0m
        gh = gsh.apply_matrix(M)
        sh.append(gh.rms_size.evalf() * 1e6)
        clh.append(gh.rms_cl.evalf() * 1e6)
        gv = gsv.apply_matrix(M)
        sv.append(gv.rms_size.evalf() * 1e6)
        clv.append(gv.rms_cl.evalf() * 1e6)

    return z, sh, clh, sv, clv


def collimate_undulator():

    gsh = GSM(wavelen=1.6e-10, rms_size=30e-6, rms_cl=3.1e-6)

    gsv = GSM(wavelen=1.6e-10, rms_size=5.4e-6, rms_cl=3.8e-6)

    z0 = free_space("z0")
    F = lens("F")
    z1 = free_space("z1")

    # gsbh = Gauss-Schell Before Horizontal
    gsbh = gsh.apply_matrix(z0)
    gsbv = gsv.apply_matrix(z0)

    gsah = gsh.apply_matrix(z1 * F * z0)
    gsav = gsv.apply_matrix(z1 * F * z0)

    beams = dict(h=gsh, v=gsv)
    attrs = dict(size="rms_size", cl="rms_cl")

    funcs = dict()
    for bname, beam in beams.items():
        funcs[bname] = dict()
        for name, attr in attrs.items():
            funcs[bname][name] = dict()
            expr_b = getattr(beam.apply_matrix(z0), attr)
            expr_b = sympy.lambdify(expr_b.free_symbols, expr_b)
            expr_a = getattr(beam.apply_matrix(z1 * F * z0), attr)
            print(expr_a.free_symbols)
            symbols = (
                get_symbol(expr_a, "z0"),
                get_symbol(expr_a, "F"),
                get_symbol(expr_a, "z1"),
            )
            expr_a = sympy.lambdify(symbols, expr_a)
            funcs[bname][name]["before"] = expr_b
            funcs[bname][name]["after"] = expr_a

    def f(z, z0=30, F=30):
        if isinstance(z, (float, int)):
            z = np.asarray([z])
        res = dict()
        for bname in beams.keys():
            res[bname] = dict()
            for name in attrs.keys():
                res[bname][name] = dict()
                y = np.zeros_like(z)
                idx = z < z0
                y[idx] = funcs[bname][name]["before"](z[idx]) * 1e6
                y[~idx] = funcs[bname][name]["after"](z0, F, z[~idx] - z0) * 1e6
                res[bname][name] = y
        return ds(res)

    return f, funcs


def example_collimate_undulator():
    z = np.linspace(0, 230, 2301)

    f, funcs = collimate_undulator()

    h = f(z, z0=40, F=32)
    v = f(z, z0=42, F=33)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(z, h.h.cl, label="hor")
    ax[0].plot(z, v.v.cl, label="ver")
    ax[0].set_xlabel("distance (m)")
    ax[0].set_ylabel("RMS ξ (μm)")
    ax[0].legend()

    ax[1].plot(z, h.h.size)
    ax[1].plot(z, v.v.size)
    ax[1].set_xlabel("distance (m)")
    ax[1].set_ylabel("RMS size (μm)")

    for a in ax:
        a.grid()

    return funcs


gs = GSM()
gsn = GSM(wavelen=1e-10, rms_size=3e-6, rms_cl=2e-6)
