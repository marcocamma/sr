import sys
from matplotlib import pyplot as plt
sys.path.append("../../../")
from sr import undulator

u = undulator.u17
e = u.fundamental()

colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

divhs = (0.5,1,2,4)
idxs = [10,20,30]
fig,axes=plt.subplots(len(divhs)+2,len(idxs)+1,figsize=[12,12])#,sharex="col",sharey="row")
gs = axes[0, 0].get_gridspec()
# remove the underlying axes
for ax in axes[:,0]: ax.remove()

axspec = fig.add_subplot(gs[:, 0])

for irow,divh in enumerate(divhs):
    u.ebeam.divh = divh*1e-6
    u.ebeam.emitth = u.ebeam.sh*divh*1e-6 # emitth is used
    s=u.xrt_photon_flux_density(h=1,v=1,dist=30,e=[e-0.3,e+0.3],ne=50,nv=62,nh=61)
    axspec.plot(s.energy,s.spectral_photon_flux,color=colors[irow],label=f"{divh} urad")
    for icol,idx in enumerate(idxs):
        axes[irow][icol+1].pcolormesh(s.h,s.v,s.spectral_photon_flux_density[idx],shading="gouraud")
        axes[irow][icol+1].set_aspect("equal")
        axes[0][icol+1].set_title(f"E = {s.energy[idx]:.3f} keV")
        axes[-2][icol+1].plot(s.h,s.spectral_photon_flux_density[idx].sum(axis=0),color=colors[irow])
        axes[-1][icol+1].plot(s.v,s.spectral_photon_flux_density[idx].sum(axis=1),color=colors[irow])
        axspec.axvline(s.energy[idx],ls="--",ymax=0.1)
        axes[-1][icol+1].set_xlabel("mm")
    axes[irow][1].set_ylabel(f"RMS divh {divh} urad",color=colors[irow])
    axes[-2][1].set_ylabel("horizontal profiles")
    axes[-1][1].set_ylabel("vertical profiles")
axspec.legend()
axspec.grid()
axspec.set_ylim(0,1.4e15)
axspec.set_ylabel("Spectrum in 1x1mm$^2$ @ 30m\n[ph/s/0.1%BW]")
axspec.set_xlabel("Energy (keV)")

plt.tight_layout()
plt.savefig("profile_vs_emittance_at_detuned_energy.pdf",transparent=True)
