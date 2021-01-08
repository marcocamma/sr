import os
import numpy as np
import sympy
from matplotlib import pyplot as plt
import scipy

import sr
from sr.undulator import Undulator
from sr.crl import Transfocator
from sr.crl import LensBlock

from sr.abcd import propagate
from sr.abcd.useful_beams import source
from sr.abcd import optics

transfocator = Transfocator([LensBlock(2 ** i, radius=5000e-6) for i in range(8)])


def ex1(energy=10,pinhole=0.5):
    beam = source(period=20, length=2.5, energy=energy)[0]
    return propagate(
        beam=beam,
        optics=[[40, f"x{pinhole}", None], [150, None, "focus@200"]],
        z=np.arange(0, 230, 0.5),
        use_transfocator=True,
        transfocator=transfocator,
    )


def ex2(energy=10,pinhole=0.5):
    beam = source(period=20, length=2.5, energy=energy)[0]
    return propagate(
        beam=beam,
        optics=[[100, f"x{pinhole}", None], [150, None, "focus@200"]],
        z=np.arange(0, 230, 0.5),
        use_transfocator=True,
        transfocator=transfocator,
    )


def ex3(energy=10,pinhole=0.5):
    beam = source(period=20, length=2.5, energy=energy)[0]
    return propagate(
        beam=beam,
        optics=[[150, f"x{pinhole}", "focus@200"],],
        z=np.arange(0, 230, 0.5),
        use_transfocator=True,
        transfocator=transfocator,
    )


def ex4(energy=10,pinhole=0.5):
    beam = source(period=20, length=2.5, energy=energy)[0]
    return propagate(
        beam=beam,
        optics=[[150, None, "focus@200"],],
        z=np.arange(0, 230, 0.5),
        use_transfocator=True,
        transfocator=transfocator,
    )


def ex5(energy=10,pinhole=0.5):
    print("This examples shows that numerically and analytically (Vartanyans 2013)")
    print("one gets the same result")

    beam = source(period=20, length=2.5, energy=energy)[0]  # [0] is hor

    # prepare beam to be focussed
    beam = beam.propagate(40).hard_aperture("x0.5").propagate(120)

    # pick lenses
    lenses = transfocator.find_best_set_for_focal_length(focal_length=25, energy=energy).best_lens_set

    # do analytical
    beam_at_focus = lenses.calc_focusing_GSM(beam)

    dist = beam_at_focus.focus_distance

    beam_analytical = beam_at_focus.gsm_at_focus

    dz = np.arange(-5,5,0.1)

    # do numerical
    beam_numerical = beam.apply(lenses)
    
    fwhm_analytical = [beam_analytical.propagate(dzi).rms_size*2.35*1e6 for dzi in dz]
    fwhm_numerical = [beam_numerical.propagate(dist+dzi).rms_size*2.35*1e6 for dzi in dz]
    cl_fwhm_analytical = [beam_analytical.propagate(dzi).rms_cl*2.35*1e6 for dzi in dz]
    cl_fwhm_numerical = [beam_numerical.propagate(dist+dzi).rms_cl*2.35*1e6 for dzi in dz]
    fig,ax=plt.subplots(2,1,sharex=True)
    ax[0].plot(dz,fwhm_analytical,label="FWHM analytical")
    ax[0].plot(dz,fwhm_numerical,".",label="FWHM numerical")
    ax[1].plot(dz,cl_fwhm_analytical,label="ξ FWHM analytical")
    ax[1].plot(dz,cl_fwhm_numerical,".",label="ξ FWHM numerical")
    ax[1].set_xlabel("Distance from analytical focus")
    ax[0].set_title(f"Focusing {dist:.3f}m from lens")
    for a in ax: a.grid()
    ax[0].set_ylabel("FWHM size")
    ax[1].set_ylabel("ξ FWHM")
    return beam_analytical


def ex6(energy=10,pinhole=0.5):
    print("ex6: showing that adding an aperture `outside` the CRL object")
    print("     gives equivalent output than the Vartanyans 2013 paper")

    beam = source(period=20, length=2.5, energy=energy)[0]  # [0] is hor

    # prepare beam to be focussed
    beam = beam.propagate(150)


    # pick lenses
    lenses = transfocator.find_best_set_for_focal_length(focal_length=35, energy=energy).best_lens_set
    
    cl = beam.rms_cl*2.35*pinhole

    beam1 = beam.hard_aperture(cl)

    # focus apertured beam
    beam_at_focus1 = lenses.calc_focusing_GSM(beam1)
    print("\n\n")
    print("\n\n")

    # focus beam
    lenses.pinhole = cl
    beam_at_focus2 = lenses.calc_focusing_GSM(beam)
    
    dist = beam_at_focus2.focus_distance


    beam_at_focus1=beam_at_focus1.gsm_at_focus
    beam_at_focus2=beam_at_focus2.gsm_at_focus


    dz = np.arange(-5,5,0.1)

    s1 = [beam_at_focus1.propagate(dzi).rms_size*2.35*1e6 for dzi in dz]
    s2 = [beam_at_focus2.propagate(dzi).rms_size*2.35*1e6 for dzi in dz]
    cl1 = [beam_at_focus1.propagate(dzi).rms_cl*2.35*1e6 for dzi in dz]
    cl2 = [beam_at_focus2.propagate(dzi).rms_cl*2.35*1e6 for dzi in dz]
    fig,ax=plt.subplots(2,1,sharex=True)
    ax[0].plot(dz,s1,label="FWHM aperturing outside CRL object")
    ax[0].plot(dz,s2,label="FWHM aperturing in CRL object")
    ax[1].plot(dz,cl1,label="ξ1")
    ax[1].plot(dz,cl2,label="ξ2")
    ax[1].set_xlabel(f"Distance from focus @ {dist:.2f}m from optics")
    #ax[0].set_title(f"Focusing {dist:.3f}m from lens")
    for a in ax: a.grid()
    ax[0].set_ylabel("FWHM size")
    ax[1].set_ylabel("ξ FWHM")

    return 

def ex7(energy=10,pinhole=0.5):
    print("ex7 showing that with too small of a pinhole it properly focus")
    print("The horizontal undulator beam is propagated for 150 m, then slitted")
    print("down to 0.1,0.2,0.4,0.8 FWHM ξ")
    print("For each lens set the beam value at 200 m is calculated and the")
    print("best set is used")

    beam = source(period=20, length=2.5, energy=energy)[0]  # [0] is hor

    # prepare beam to be focussed
    beam = beam.propagate(150)

    sizes = [0.1,0.2,0.4,0.8,1.6]
    x = np.arange(0,100,0.1)
    plt.figure("ex7")
    for pinhole in sizes:
        print(f"\n\n### Working on pinhole {pinhole} ###")
        cl = beam.rms_cl*2.35*pinhole
        beam1 = beam.hard_aperture(cl)
        s_at_200 = [beam1.apply(l).propagate(50).rms_size for l in transfocator.all_sets]
        s_at_200 = np.asarray(s_at_200)
        idx = np.argmin(s_at_200)
        best_lens = transfocator.all_sets[idx]
        print(f"Best size at 200m {min(s_at_200)*1e6:.2f}μm")
        print("Best set to have smallest spot @ 200m",str(best_lens))
        focus = best_lens.calc_focusing_GSM(beam1)
        gsm = beam1.apply(best_lens)
        plt.plot(150+x,gsm.propagate(x).rms_size*1e6,label=str(pinhole))
    return focus

def ex8(energy=10):
    print("ex8 similar conclusions as ex7")
    print("The horizontal undulator beam is propagated for 150 m, then slitted")
    print("The beamsize at 200m is calculated as funciton of the focal length")

    beam = source(period=20, length=2.5, energy=energy)[0]  # [0] is hor
    optics_dist = [0,5,10,20,40]
    sizes = [0.1,0.2,0.4,0.8,1.6]
    fig,ax=plt.subplots(len(optics_dist),1,num="ex8",sharex=True,sharey=True,figsize=[8.27,11.69])
    #fig.clf()
    f = np.arange(1,50,0.1)
    for iplot,d in enumerate(optics_dist):
        for pinhole in sizes:
            # b1 is beam at lens
            b1 = beam.propagate(150).hard_aperture(f"x{pinhole}").propagate(d)
            aperture = beam.propagate(150).rms_cl*2.35*pinhole*1e6
            s=[b1.lens(_f).propagate(50-d).rms_size*2.35*1e6 for _f in f]
            line,=ax[iplot].semilogy(f,s,label=f"aperture {pinhole}xξ={aperture:.1f}μm")
            dl_size = sr.coherent.diffraction_limited_spot_gaussian(
                wavelength=beam.wavelen,
                focal_length=50-d,
                fwhm=b1.rms_size*2.35,
            )
            ax[iplot].axhline(dl_size*1e6,ls="--",color=line.get_color(),alpha=0.5)

        ax[iplot].set_title(f"pinhole-optics distance {d}m")
        ax[iplot].set_ylabel("FWHM beam size (μm)")
        ax[iplot].grid()
    ax[-1].legend()
    ax[-1].set_xlabel("focal length (m)")
    fig.tight_layout()

def block1(energy=10,pinhole=0.5):
    plt.figure()
    print("\n\n")
    e1 = ex1(energy=energy,pinhole=pinhole)
    print("\n\n")
    e2 = ex2(energy=energy,pinhole=pinhole)
    print("\n\n")
    e3 = ex3(energy=energy,pinhole=pinhole)
    print("\n\n")
    e4 = ex4(energy=energy,pinhole=pinhole)
    plt.plot(e1.z, e1.fwhm_size)
    plt.plot(e2.z, e2.fwhm_size)
    plt.plot(e3.z, e3.fwhm_size)
    plt.plot(e4.z, e4.fwhm_size)



def main(energy=10,pinhole=0.5):

    block1(energy=energy,pinhole=pinhole)
    ex5(energy=energy,pinhole=pinhole)

    ex6(energy=energy,pinhole=pinhole)

    ex7(energy=energy,pinhole=pinhole)


if __name__ == "__main__":
    main(energy=10,pinhole=0.5)
    input("ok to exit")
