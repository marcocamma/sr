from matplotlib import pyplot as plt
try:
    from sr import undulator
except ImportError:
    import sys
    sys.path.insert(0,"../../")
    from sr import undulator

u=undulator.get_cpmu(20,2.5)



def compare(harmonic=3):
    plt.figure()
    pars_spectrum = dict(gap=10,h=0.3,v=0.3,nv=31,nh=31,dist=30,e=[-0.5,0.5],ne=51)
    i_xrt=u.xrt_photon_flux_density(harmonic=harmonic,**pars_spectrum)
    i_srw=u.srw_photon_flux_density(harmonic=harmonic,**pars_spectrum)
    plt.plot(i_srw.energy,i_srw.spectral_photon_flux,label="SRW")
    plt.plot(i_xrt.energy,i_xrt.spectral_photon_flux,label="XRT")
    plt.ylabel(i_xrt.units_spectral_photon_flux)
    plt.xlabel("energy (keV)")
    plt.grid()

    harmonic = 1
    energy_detuning = -0.2
    pars_spectrum = dict(
            gap=10,
            h=2,v=1.5,nv=61,nh=51,
            dist=30,
            e=u.photon_energy(gap=10,harmonic=harmonic)+energy_detuning)

    i_xrt=u.xrt_photon_flux_density(harmonic=harmonic,**pars_spectrum)
    i_srw=u.srw_photon_flux_density(harmonic=harmonic,**pars_spectrum)
 

    fig,ax=plt.subplots(2,1)
    idx = 0
    ax[0].set_title(f"SRW ({i_srw.units_spectral_photon_flux_density})")
    pcm = ax[0].pcolormesh(i_srw.h,i_srw.v,i_srw.spectral_photon_flux_density[idx])
    fig.colorbar(pcm,ax=ax[0])
    ax[1].set_title(f"XRT ({i_xrt.units_spectral_photon_flux_density})")
    pcm = ax[1].pcolormesh(i_xrt.h,i_xrt.v,i_xrt.spectral_photon_flux_density[idx])
    fig.colorbar(pcm,ax=ax[1])
    plt.tight_layout()

    return i_xrt



if __name__ == "__main__":
    compare()
