"""
Notes: if some expressions do not simplify why they should they might 
have 'duplicates'. remove them by using remove_duplicates
"""
import numpy as np
import sympy.physics.optics as so
import sympy

sympy.init_printing(use_unicode=True, use_latex=True, pretty_printing=True)
from sympy import Symbol

RMS_TO_FWHM = 2*sympy.sqrt(2*sympy.ln(2))

def gaussian_transmission_1D(sig,slit_opening):
    return sympy.erf(sympy.sqrt(2)*slit_opening/(4*sig))

def gaussian_transmission_2D(sig,pinhole_diameter):
    return 1 - sympy.exp(-pinhole_diameter**2/(8*sig**2))


def get_symbol(expr, name):
    symbols = list(expr.free_symbols)
    match = [s for s in symbols if s.name == name]
    if len(match) == 0:
        return None
    if len(match) > 1:
        raise ValueError(expr, "has multiple parameters with the same name!")
    else:
        return match[0]


def remove_duplicates(expr):
    symbols = list(expr.free_symbols)
    for symbol in symbols:
        match = [s for s in symbols if s.name == symbol.name]
        for m in match[1:]:
            expr = expr.subs(m, match[0])
    return expr


def find_roots(expr, var, use_derivative=False, squeeze=True):
    expr = expr.simplify()
    orig = expr.copy()
    if isinstance(var, str):
        var = get_symbol(expr, var)
    if use_derivative:
        expr = sympy.diff(expr, var)
    roots = sympy.solve(expr, var)
    res = [orig.subs(var, r) for r in roots]
    if squeeze and len(roots) == 1:
        roots = roots[0]
        res = res[0]
    return roots, res


def real_sympify(x):
    if isinstance(x, str):
        x = sympy.Symbol(x, real=True, positive=True)
    return x


def lens(focal_length="f", force_positive=False):
    if isinstance(focal_length, str):
        if force_positive:
            focal_length = sympy.Symbol(focal_length, real=True, positive=True)
        else:
            focal_length = sympy.Symbol(focal_length, real=True)
    return so.ThinLens(focal_length)


def free_space(dist="d", force_positive=False):
    if isinstance(dist, str):
        if force_positive:
            dist = sympy.Symbol(dist, real=True, positive=True)
        else:
            dist = sympy.Symbol(dist, real=True)
    if isinstance(dist,np.ndarray):
        D = np.empty( (len(dist),2,2) )
        for i,d in enumerate(dist):
            D[i,0,0] = 1
            D[i,0,1] = d
            D[i,1,0] = 0
            D[i,1,1] = 1
    else:
        D = so.FreeSpace(dist)
    return D


class GaussianAperture:
    def __init__(self, effective_aperture="Ω"):
        if isinstance(effective_aperture, str):
            effective_aperture = sympy.Symbol(
                effective_aperture, real=True, positive=True
            )
        self.effective_aperture = effective_aperture

class HardAperture:
    """
    As shown in Singer&Vartanyans JSR 2014, a pinhole can be approaximated
    by a gaussian_aperture with effective_aperture Ω = D/4.55
    """
    def __init__(self, aperture="D"):
        if isinstance(aperture, str):
            aperture = sympy.Symbol(
                aperture, real=True, positive=True
            )
        self.effective_aperture = aperture/4.55



def gaussian_aperture(effective_aperture="Ω"):
    return GaussianAperture(effective_aperture=effective_aperture)


def hard_aperture(aperture="D"):
    """
    As shown in Singer&Vartanyans JSR 2014, a pinhole can be approaximated
    by a gaussian_aperture with effective_aperture Ω = D/4.55
    """
    return HardAperture(aperture=aperture)


class GSM:
    def __init__(
        self,
        wavelen="λ",
        rms_size="σ_0",
        rms_cl="ξ_0",
        radius=sympy.oo,
        intensity = 1,
        simplify=True,
        auto_apply_evalf=False
    ):
        """ 
        auto_apply_evalf can be used to make make evalf inheritable
        """
        if isinstance(wavelen, str):
            wavelen = sympy.Symbol(wavelen, real=True, positive=True)
        if isinstance(rms_cl, str):
            rms_cl = sympy.Symbol(rms_cl, real=True, positive=True)
        if isinstance(rms_size, str):
            rms_size = sympy.Symbol(rms_size, real=True, positive=True)
        
        self._attrs_to_work_on = "alpha rms_size rms_cl beta b divergence w radius intensity global_degree_of_coherence".split()


        self.wavelen = wavelen
        self.w = 2 * rms_size
        self.radius = radius
        self.intensity = intensity


        if not isinstance(rms_cl,np.ndarray) and rms_cl == sympy.oo:
            self.beta = 1
            self.global_degree_of_coherence = 1
        else:
            self.beta = rms_cl / sympy.sqrt((rms_cl ** 2 + self.w ** 2))
            self.global_degree_of_coherence = 1/sympy.sqrt(1+(2*rms_size/rms_cl)**2)

        self.b = sympy.pi / wavelen * self.w ** 2 * self.beta  # eq 38

        self.rms_size = rms_size
        self.rms_cl = rms_cl
        self.q = 1 / (1 / self.radius - sympy.I / self.b)
        self.alpha = self.rms_cl / self.rms_size

        # note: the /2 in the expression of divergence
        # comes from the fact that for me it means
        # d(RMS size)/dz while usually (in the gaussian optics context)
        # it is meant to be d(w)/dz

        self.divergence = sympy.sqrt(
            self.wavelen / sympy.pi / sympy.Abs(sympy.im(self.q * self.beta))
        )/2

        self.global_degree_of_coherence = 1/sympy.sqrt(1+(2*rms_size/rms_cl)**2)

        self.auto_apply_evalf=auto_apply_evalf
        if self.auto_apply_evalf:
            self.evalf()
            simplify = False
        self.simplify = simplify
        if simplify: self.dosimplify()


    def dosimplify(self):
        for attr in self._attrs_to_work_on:
            try:
                setattr(self, attr, getattr(self, attr).simplify())
            except AttributeError:
                pass

    def evalf(self):
        for attr in self._attrs_to_work_on:
            try:
                setattr(self, attr, getattr(self, attr).evalf())
            except AttributeError:
                pass


    def propagate(self, pos="z",evalf_result="auto"):
        if isinstance(pos, str):
            pos = sympy.Symbol(pos, real=True)
        d = free_space(pos)
        return self.apply(d,evalf_result=evalf_result)

    def lens(self, focal_length="f",evalf_result="auto"):
        if isinstance(focal_length, str):
            focal_length = sympy.Symbol(focal_length, real=True)
        m = lens(focal_length)
        return self.apply(m,evalf_result=evalf_result)

    def apply_matrix(self, M,evalf_result="auto"):
        A, B, C, D = M
        R = self.radius
        b = self.b
        up = (A + B / R) ** 2 + (B / b) ** 2
        b2 = b * up / (A * D - B * C)
        R2 = up / ((A + B / R) * (C + D / R) + B * D / b ** 2)

        wavelen = self.wavelen
        radius = R2
        w = sympy.sqrt(b2 * wavelen / sympy.pi / self.beta)
        rms_size = w / 2
        rms_cl = rms_size * self.alpha
        if evalf_result == "auto": evalf_result=self.auto_apply_evalf

        return GSM(
            wavelen=wavelen,
            rms_size=rms_size,
            rms_cl=rms_cl,
            radius=radius,
            auto_apply_evalf=evalf_result,
            intensity=self.intensity,
            simplify=self.simplify
        )

    def gauss_aperture(self, aperture, evalf_result="auto"):
        if isinstance(aperture,(HardAperture,GaussianAperture)):
            pass
        else:
            aperture = gaussian_aperture(aperture)
        rms_cl = self.rms_cl
        s = self.rms_size ** 2
        a = aperture.effective_aperture ** 2
        rms_size = sympy.sqrt(a * s / (a + s))
        radius = self.radius
        intensity=self.intensity*gaussian_transmission_1D(self.rms_size,aperture.effective_aperture*4.55)
        if evalf_result == "auto": evalf_result=self.auto_apply_evalf
        return GSM(
            wavelen=self.wavelen,
            rms_size=rms_size,
            rms_cl=rms_cl,
            radius=radius,
            auto_apply_evalf=evalf_result,
            intensity=intensity,
            simplify=self.simplify
        )

    def hard_aperture(self, aperture, evalf_result="auto"):
        """
        aperture can be specified as 'x1.2' meaning 1.2 time the FWHM of the coherence length
        """
        if isinstance(aperture,str):
            aperture = float(aperture[1:])*self.rms_cl*RMS_TO_FWHM
        if not isinstance(aperture,HardAperture):
            aperture = hard_aperture(aperture)
        return self.gauss_aperture(aperture,evalf_result=evalf_result)


    def apply(self, optics, evalf_result="auto"):
        from .. import crl
        if isinstance(optics, so.RayTransferMatrix):
            return self.apply_matrix(optics,evalf_result=evalf_result)
        elif isinstance(optics, GaussianAperture):
            return self.gauss_aperture(optics,evalf_result=evalf_result)
        elif isinstance(optics, HardAperture):
            return self.hard_aperture(optics,evalf_result=evalf_result)
        elif isinstance(optics, (crl.LensBlock,crl.LensSet)):
            energy = 12.398/(self.wavelen*1e10)
            lens_hard_aperture = optics.aperture()
            lens_abs_gauss_aperture = optics.absorption_opening(energy)
            fl = optics.focal_length(energy)
            
            # convert into Classes
            fl = lens(fl)

            t1 = self.hard_aperture(lens_hard_aperture)
            t2 = t1.gauss_aperture(lens_abs_gauss_aperture)
            return t2.apply(fl)
        else:
            raise ValueError("optics has to be gaussian aperture or matrix or CRL lens")


    def coherent(self):
        return GSM(
                wavelen=self.wavelen,
                rms_size=self.rms_size,
                rms_cl=sympy.oo,
                radius = self.radius,
                intensity=self.intensity,
                auto_apply_evalf=self.auto_apply_evalf,
                simplify=self.simplify
                )

    def as_numeric(self):
        return GSM_Numeric(
            wavelen=np.float(self.wavelen),
            rms_size=np.float(self.rms_size),
            rms_cl=np.float(self.rms_cl),
            radius=np.float(self.radius),
            intensity=np.float(self.intensity),
        )


    def __repr__(self):
        s = "GSM beam\n"
        s += "wavelen : " + str(self.wavelen) + "\n"
        s += "rms size: " + str(self.rms_size) + "\n"
        s += "rms cl  : " + str(self.rms_cl) + "\n"
        s += "radius  : " + str(self.radius) + "\n"
        s += "div     : " + str(self.divergence) + "\n"
        s += "g.d.c   : " + str(self.global_degree_of_coherence)
        return s

    def __str__(self):
        return self.__repr__()


class GSM_Numeric:
    def __init__(
        self,
        wavelen=1e-10,
        rms_size=5e-6,
        rms_cl=10e-6,
        radius=np.inf,
        intensity = 1,
    ):
        

        self.wavelen = wavelen
        self.w = 2 * rms_size
        self.radius = radius
        self.intensity = intensity


        if not isinstance(rms_cl,np.ndarray) and rms_cl == np.inf:
            self.beta = 1
            self.global_degree_of_coherence = 1
        else:
            self.beta = rms_cl / np.sqrt((rms_cl ** 2 + self.w ** 2))
            self.global_degree_of_coherence = 1/np.sqrt(1+(2*rms_size/rms_cl)**2)

        self.b = np.pi / wavelen * self.w ** 2 * self.beta  # eq 38

        self.rms_size = rms_size
        self.rms_cl = rms_cl
        self.q = 1 / (1 / self.radius - 1j / self.b)
        self.alpha = self.rms_cl / self.rms_size

        # note: the /2 in the expression of divergence
        # comes from the fact that for me it means
        # d(RMS size)/dz while usually (in the gaussian optics context)
        # it is meant to be d(w)/dz
        self.divergence = np.sqrt(
            self.wavelen / np.pi / np.abs(np.imag(self.q * self.beta))
        )/2


        self.global_degree_of_coherence = 1/np.sqrt(1+(2*rms_size/rms_cl)**2)



    def propagate(self, pos=10):
        d = free_space(pos)
        return self.apply_matrix(d)

    def lens(self, focal_length=10):
        m = lens(focal_length)
        return self.apply(m)


    def apply_matrix(self, M):
        if not isinstance(M,np.ndarray):
            M = np.asarray(M).astype(float)
        if M.ndim == 2:
            M = M[np.newaxis,:]
        n = len(M)
        rms_size = np.empty(n)
        radius = np.empty(n)
        R = self.radius
        b = self.b
        beta = self.beta
        alpha = self.alpha
        wavelen = self.wavelen
        for i,m in enumerate(M):
           A, B, C, D = m.ravel()
           p = (A + B / R) ** 2 + (B / b) ** 2
           up = (A + B / R) ** 2 + (B / b) ** 2
           b2 = b * up / (A * D - B * C)
           R2 = up / ((A + B / R) * (C + D / R) + B * D / b ** 2)

           radius[i] = R2
           w = np.sqrt(b2 * wavelen / np.pi / beta)
           rms_size[i] = w / 2
        rms_cl = rms_size * alpha
        if M.shape[0] == 1:
            radius = float(radius)
            rms_size = float(rms_size)
            rms_cl = float(rms_cl)
        return GSM_Numeric(
            wavelen=self.wavelen,
            rms_size=rms_size,
            rms_cl=rms_cl,
            radius=radius,
            intensity = self.intensity)


    def gauss_aperture(self, aperture):
        if isinstance(aperture,(HardAperture,GaussianAperture)):
            pass
        else:
            aperture = gaussian_aperture(aperture)
        rms_cl = self.rms_cl
        s = self.rms_size ** 2
        a = float(aperture.effective_aperture) ** 2
        rms_size = np.sqrt(a * s / (a + s))
        radius = self.radius
        intensity=self.intensity*float(gaussian_transmission_1D(self.rms_size,aperture.effective_aperture*4.55))
        return GSM_Numeric(
            wavelen=self.wavelen,
            rms_size=rms_size,
            rms_cl=rms_cl,
            radius=radius,
            intensity=intensity,
        )

    def hard_aperture(self, aperture):
        """
        aperture can be specified as 'x1.2' meaning 1.2 time the FWHM of the coherence length
        """
        if isinstance(aperture,str):
            aperture = float(aperture[1:])*self.rms_cl*RMS_TO_FWHM
        if not isinstance(aperture,HardAperture):
            aperture = hard_aperture(aperture)
        return self.gauss_aperture(aperture)

    def apply(self, optics,):
        from .. import crl
        if isinstance(optics, so.RayTransferMatrix) or isinstance(optics,np.ndarray):
            return self.apply_matrix(optics)
        elif isinstance(optics, GaussianAperture):
            return self.gauss_aperture(optics)
        elif isinstance(optics, HardAperture):
            return self.hard_aperture(optics)
        elif isinstance(optics, (crl.LensBlock,crl.LensSet)):
            energy = 12.398/(self.wavelen*1e10)
            lens_hard_aperture = optics.aperture()
            lens_abs_gauss_aperture = optics.absorption_opening(energy)
            fl = optics.focal_length(energy)
            
            # convert into Classes
            fl = lens(fl)

            t1 = self.hard_aperture(lens_hard_aperture)
            t2 = t1.gauss_aperture(lens_abs_gauss_aperture)
            return t2.apply(fl)
        else:
            raise ValueError("optics has to be gaussian aperture or matrix or CRL lens")


    def coherent(self):
        return GSM_Numeric(
                wavelen=self.wavelen,
                rms_size=self.rms_size,
                rms_cl=np.inf,
                radius = self.radius,
                intensity=self.intensity,
                )

    def __repr__(self):
        s = "GSM Numeric beam\n"
        s += "wavelen : " + str(self.wavelen) + "\n"
        s += "rms size: " + str(self.rms_size) + "\n"
        s += "rms cl  : " + str(self.rms_cl) + "\n"
        s += "radius  : " + str(self.radius) + "\n"
        s += "div     : " + str(self.divergence) + "\n"
        s += "g.d.c   : " + str(self.global_degree_of_coherence)
        return s

    def __str__(self):
        return self.__repr__()



class GaussBeam:
    def __init__(self, wavelen="λ", rms_size="σ_0", radius=sympy.oo, q=None):
        wavelen, rms_size, radius = map(real_sympify, [wavelen, rms_size, radius])
        self.wavelen = wavelen
        if q is not None:
            self.q = q
            self.radius = sympy.Abs(q) ** 2 / sympy.re(q)
            self.w = sympy.sqrt(
                sympy.Abs(self.wavelen / sympy.pi * sympy.Abs(q) ** 2 / sympy.im(q))
            )
            self.rms_size = self.w / 2

        else:
            self.radius = radius
            self.rms_size = rms_size
            self.w = 2 * rms_size
            self.q = 1.0 / (
                1.0 / radius + sympy.I * self.wavelen / (sympy.pi * self.w ** 2)
            )
        # note: the /2 in the expression of divergence
        # comes from the fact that for me it means
        # d(RMS size)/dz while usually (in the gaussian optics context)
        # it is meant to be d(w)/dz
        self.divergence = sympy.sqrt(
            self.wavelen / sympy.pi / sympy.Abs(sympy.im(self.q))
        )/2
        # self.radius = self.z*(1+(self.z_r/self.z_0)**2)

    def q(self, pos="z"):
        if isinstance(pos, str):
            pos = sympy.Symbol(pos)
        self.q = (pos - self.z_0) + sympy.I * self.z_r

    # def w(self,pos="z"):
    #    if isinstance(pos,str): pos = sympy.Symbol(pos)
    #    diff = pos-self.z_0
    #    return self.w_0*sympy.sqrt(1+(diff/self.z_r)**2)

    # def rms_size(self,pos="z"):
    #    return self.w(pos=pos)/2

    def R(self, pos="z"):
        if isinstance(pos, str):
            pos = sympy.Symbol(pos)
        diff = pos - self.z_0
        return diff * (1 + (self.z_r / diff) ** 2)

    def apply_matrix(self, M):
        return apply_matrix_gauss(M, self)

    def propagate(self, pos="z"):
        if isinstance(pos, str):
            pos = sympy.Symbol(pos, real=True)
        d = free_space(pos)
        return self.apply_matrix(d)



    def __repr__(self):
        s = "Gaussian beam\n"
        s += "radius : " + str(self.radius) + "\n"
        s += "rms size : " + str(self.rms_size) + "\n"
        s += "Divergence: " + str(self.divergence)
        return s

    def __str__(self):
        s = f"Gbeam, radius={str(self.radius)}, rms {str(self.rms_size)}, div {str(self.divergence)}"
        return s


def apply_matrix_gauss(M, beam):
    """ return new BeamParameter object """
    qnew = (M[0] * beam.q + M[1]) / (M[2] * beam.q + M[3])
    g = GaussBeam(wavelen=beam.wavelen, q=qnew)
    return g



gs = GSM()
gsn = GSM_Numeric(wavelen=1e-10, rms_size=3e-6, rms_cl=2e-6)
