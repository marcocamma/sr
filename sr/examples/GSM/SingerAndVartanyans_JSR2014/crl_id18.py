import sys
sys.path.insert(0,"../../../../")

from sr.utils.unicode import mu as MU
from sr.abcd import useful_beams
from sr.crl import LensSet


def test_id18(frac_cl=1):
    h = useful_beams.id18h
    v = useful_beams.id18v
    ls = LensSet( [ [1,200e-6], [2,50e-6] ] )
    ls = LensSet( [ [1,350e-6], ] )


    h50 = h.propagate(50)
    h50.evalf()
    cl = float(h50.rms_cl*2.35)
    s = cl*frac_cl
    print("\nID18 Horizontal focusing %.1f %sm slit"%(s*1e6,MU))
    ls.calc_focusing_GSM(h,source_distance=40,slit_opening=s)

    v50 = v.propagate(50)
    v50.evalf()
    cl = float(v50.rms_cl*2.35)
    s = cl*frac_cl
    print("\nID18 Vertical focusing %.1f %sm slit"%(s*1e6,MU))
    ls.calc_focusing_GSM(h,source_distance=40,slit_opening=s)

if __name__ == "__main__":
    test_id18()
