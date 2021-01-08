try:
    from sr import undulator
except ImportError:
    import sys
    sys.path.insert(0,"../../")
    from sr import undulator


u=undulator.Undulator(elattice="jsr2019",period=29,length=5,
        ebeam_pars=dict(rms_energy_spread=1e-3)
        )


energy_keV = 12

pars = u.find_harmonic_and_gap(energy_keV)[0]
uxrt = u.as_xrt(**pars)
beam = u.photon_beam_characteristics(**pars)

print(f"Photon Beam Size (KK SR)  {beam.sh*1e6:.2f},{beam.sv*1e6:.2f} um")
xrt_sigma = uxrt.get_SIGMA(energy_keV*1e3)
print(f"Photon Beam Size (KK XRT) {xrt_sigma[0]*1e3:.2f},{xrt_sigma[1]*1e3:.2f} um")

print(f"Photon Beam div (KK SR)  {beam.divh*1e6:.2f},{beam.divv*1e6:.2f} urad")
xrt_sigmap = uxrt.get_SIGMAP(energy_keV*1e3)
print(f"Photon Beam div (KK XRT) {xrt_sigmap[0]*1e6:.2f},{xrt_sigmap[1]*1e6:.2f} urad")

xrt_emitt = xrt_sigma[0]*xrt_sigmap[0]*1e9,xrt_sigma[1]*xrt_sigmap[1]*1e9
print(f"Photon Beam Emitt (KK SR)  {beam.emitth*1e12:.3f},{beam.emittv*1e12:.3f} pm*rad")
print(f"Photon Beam Emitt (KK XRT) {xrt_emitt[0]:.3f},{xrt_emitt[1]:.3f} pm*rad")
