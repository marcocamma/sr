from .optics import GSM_Numeric
from .. import undulator
p10h = GSM_Numeric(rms_size=36.2e-6, rms_cl=0.9e-6, wavelen=12.4 * 1e-10 / 8)
p10v = GSM_Numeric(rms_size= 6.3e-6, rms_cl=7.7e-6, wavelen=12.4 * 1e-10 / 8)

def source(period=20,length=2.5,energy=8):
    u = undulator.get_cpmu(period=period,length=length)
    pars = u.find_harmonic_and_gap(energy)[0]
    b = u.photon_beam_characteristics(**pars)
    gsmh = GSM_Numeric(rms_size=b.sh, rms_cl=b.gsm_sclh, wavelen=b.wavelength*1e-10)
    #gsmh = GSM()
    gsmv = GSM_Numeric(rms_size=b.sv, rms_cl=b.gsm_sclv, wavelen=b.wavelength*1e-10)
    #gsmv = GSM()
    return gsmh, gsmv,b

_temp = source()
id18h = _temp[0]
id18v = _temp[1]

