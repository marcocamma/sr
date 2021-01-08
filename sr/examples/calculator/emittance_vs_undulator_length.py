import numpy as np
from matplotlib import pyplot as plt

try:
    from sr import undulator
except ImportError:
    import sys

    sys.path.insert(0, "../../../")
    from sr import undulator

import sr

L = np.arange(0.1, 20, 0.1)


def emittance_vs_length(
    energy_keV=12, normalize=True, show_min_emitt=False, rms_energy_spread="ebs"
):

    if not isinstance(rms_energy_spread, (float, int)):
        rms_energy_spread = sr.beam.e_beam(lattice="ebs").rms_energy_spread

    emitt = np.zeros_like(L)
    for i, l in enumerate(L):
        u = undulator.Undulator(
            elattice="ebs",
            period=20,
            length=l,
            ebeam_pars=dict(rms_energy_spread=rms_energy_spread),
        )

        pars = u.find_harmonic_and_gap(energy_keV)[0]
        beam = u.photon_beam_characteristics(**pars)
        emitt[i] = beam.emitt * 1e18

    if normalize:
        emitt /= emitt.min()
        plt.ylabel("ε$_{h}$ ⨯ ε$_{v}$ / min(ε$_{h}$ ⨯ ε$_{v}$)")
    else:
        plt.ylabel("Photon emittance (epsh*epsv) [nm²rad²]")

    label = "%s keV" % energy_keV
    print(beam.harmonic)
    if beam.harmonic == 1:
        label += " 1st harm"
    elif beam.harmonic == 3:
        label += " 3rd harm"
    line, = plt.plot(L, emitt, label=label,lw=5,alpha=0.5)
    plt.title("RMS energy spread: " + str(rms_energy_spread))
    if not normalize and show_min_emitt:
        lam = 12.398 / energy_keV * 1e-10
        eps = lam / 2 / np.pi
        eps2 = (eps + beam.ebeam.emitth) * (eps + beam.ebeam.emittv)
        eps2 = eps2 * 1e18
        #        eps2 = eps*eps
        plt.axhline(eps2, color=line.get_color())
    plt.ion()
    plt.show()


def main(rms_energy_spread="ebs"):
    for energy in 8, 15, 20, 25:
        emittance_vs_length(energy, 
                normalize=True,
                rms_energy_spread=rms_energy_spread)
    plt.legend(title="Photon Energy")
    plt.grid()
    plt.xlim(0, 5)
    plt.ylim(0.95, 1.4)
    plt.xlabel("Undulator length (m)")


if __name__ == "__main__":
    plt.close("all")
    plt.figure("With EBS Energy spread")
    main(rms_energy_spread="ebs")
    plt.figure("With Zero Energy spread")
    main(rms_energy_spread=0)
    plt.xlim(0, 20)
    plt.ylim(0.95, 1.4)
    input("ok")
