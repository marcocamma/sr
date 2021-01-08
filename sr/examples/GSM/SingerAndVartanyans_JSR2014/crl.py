import sys
sys.path.insert(0,"../../../../")

from sr.utils.unicode import mu as MU
from sr.abcd import useful_beams
from sr.crl import LensSet


def test_p10():
    h = useful_beams.p10h
    v = useful_beams.p10v
    ls = LensSet( [ [1,200e-6], [2,50e-6] ] )


    # In Singer&Vartanyans JSR 2014 article, the opening is given as 
    # 'Gaussian opening' this is the reason of the 4.55 factor below
    print("Here slit opening means Gaussian Opening in line with")
    print("Singer&Vartanyans JSR 2014 paper")
    for s in [25,100]:
        print("\nP10 Horizontal focusing %s %sm slit"%(s,MU))
        ls.calc_focusing_GSM(h,source_distance=85,slit_opening=s*1e-6*4.55)

    for s in [50,150]:
        print("\nP10 Vertical focusing %s %sm slit"%(s,MU))
        ls.calc_focusing_GSM(v,source_distance=85,slit_opening=s*1e-6*4.55)

if __name__ == "__main__":
    test_p10()
