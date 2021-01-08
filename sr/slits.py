from scipy.special import erf
import numpy as np
from matplotlib import pyplot as plt
import glob
SQRT2 = np.sqrt(2)

def slit_transmission(gap,gap0=0,pos0=0,I0=1,fwhm=1):
    if isinstance(gap,(float,int)) and gap<gap0: return 0
    sig = np.abs(fwhm)/2.355
    x1 = -(gap-gap0)/2 + pos0
    x2 = +(gap-gap0)/2 + pos0
    integral1 = 0.5*(1-erf(-x1/SQRT2/sig))
    integral2 = 0.5*(1-erf(-x2/SQRT2/sig))
    res = I0*(integral2-integral1)
    if not isinstance(gap,(float,int)):
        idx = gap<gap0
        res[idx]=0
    return res#+bkg


def fit_gap_scan(x,y,retry=True,show_plot=True,**fit_kw):
    """ if retry is true, attempt second fit with bkg and pos0 as free pars """
    import lmfit

    # put closer gap at the beginning
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    m = lmfit.Model(slit_transmission)

    # estimate pars
    fwhm_estimate = fit_kw.get("fwhm",(x.max()-x.min())/6)

    yp = np.gradient(y)
    gap0_idx = np.argwhere(yp>2*np.mean(yp[:4])).ravel()[0]
    gap0_estimate = x[gap0_idx]

    pos0_estimate = 0.0
    i0_estimate= fit_kw.get("I0",y.max())

    bkg = fit_kw.get("bkg",np.percentile(y,5))

    params = lmfit.Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)

    params.add_many(('gap0', gap0_estimate, True , None   , None      , None, None),
                    ('pos0', 0            , False , None,  None      , None, None),
                    ('I0',   i0_estimate  , True, 0       , 5*y.max()  , None, None),
                    ('fwhm', fwhm_estimate, True, 0       , None       , None, None),
                    ('bkg',   bkg, False, None, None    , None, None),
                    )



    print("Initial parameters")
    params.pretty_print()

    # do fit with some fixed parameter first
    r = m.fit(y,gap=x,params=params)

    params = r.params
    if retry:
        params["bkg"].vary=True
        params["pos0"].vary=True
        r = m.fit(y,gap=x,params=params)
    if show_plot:
        r.plot_fit()
        s = ""
        keys = r.best_values.keys()
        for k,v in r.best_values.items():
            s += "%10s=%.3e\n"%(k,v)
        plt.title(s)
        plt.tight_layout()
    print("Final parameters")
    r.params.pretty_print()
    return r

