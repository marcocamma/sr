import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase

def arrays_to_wolfry(x,y,wavelength=1e-10):
    from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
    w=GenericWavefront1D(wavelength=wavelength)
    w=w.initialize_wavefront_from_arrays(x,y)
    return WolfryWaveFronts((w,))

def find_fwhm(wf):
    try:
        x, y = wf.get_sum()
        n = wf[0].size()
    except:
        x = wf.get_abscissas()
        y = wf.get_intensity()
        n = wf.size()
    idx = np.argwhere(y > y.max() / 2).ravel()
    m = idx[0]
    M = idx[-1]
    if m == 0 or M == n:
        return np.nan
    else:
        return x[M] - x[m]


class WolfryWaveFronts:
    def __init__(self, wavefronts, xpos=0):
        from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D

        self.wavefronts = wavefronts
        if isinstance(wavefronts, GenericWavefront1D):
            wavefronts = (wavefronts,)

        self.n = len(wavefronts)
        # for iteration
        self._i = 0
        self.xpos = xpos

    def __getitem__(self, n):
        return self.wavefronts[n]

    def __len__(self):
        return self.n

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.n:
            self._i += 1
            return self.wavefronts[self._i - 1]
        else:
            raise StopIteration

    def __repr__(self):
        temp = self.wavefronts[0].get_abscissas()
        win = temp.max() - temp.min()
        n = temp.shape[0]
        return f"WolfryWaveFronts: {self.n} wavefronts, {win:.3e} xrange, {n} points, {self.xpos}m from source"

    def __str__(self):
        return self.__repr__()

    def fresnel_max_dist(self):
        """ this is the limit for the 2 FFT version """
        w0 = self.wavefronts[0]
        w0x = w0.get_abscissas()
        input_spacing = w0x[1] - w0x[0]
        nx = w0.size()
        lam = w0.get_wavelength()
        # see Vincent notes
        max_dist = (input_spacing * nx) ** 2 / lam / nx
        return max_dist

    def is_zoom_propagator_ok(self,dist=10,magnification=1,verbose=False,use_fwhm=True):
        """ tested for magnification>1"""
        w0 = self.wavefronts[0]
        n = w0.size()
        lam = w0.get_wavelength()
        if use_fwhm:
            w = 3*find_fwhm(self)
        else:
            delta = w0.delta()
            w = n*delta
        m = magnification

        f = w**2/n/lam
        zmin1 = abs(m-1)*f
        zmin2 = abs(m-1)/m*f
        zmin = max( zmin1,zmin2 )
        zmax = m*f
        if verbose:
            if dist < zmin:
                print(f"propagation distance ({dist}) smaller than minimum distance ({zmin})")
            if dist > zmax:
                print(f"propagation distance ({dist}) larger than maximum distance ({zmax})")
        return zmin < dist < zmax

    def find_zoom_propagator_magnification(self,dist=10,use_fwhm=True):
        w0 = self.wavefronts[0]
        n = w0.size()
        lam = w0.get_wavelength()
        if use_fwhm:
            w = find_fwhm(self)
        else:
            delta = w0.delta()
            w = n*delta

        f = w**2/n/lam
        if dist < f:
            m = 1
        else:
            m = dist/f+0.5
        print(self.is_zoom_propagator_ok(dist=dist,magnification=m))
        return m



    def replica_position(self,dist=10):
        w0 = self.wavefronts[0]
        w0x = w0.get_abscissas()
        input_spacing = w0x[1] - w0x[0]
        lam = w0.get_wavelength()
        replica_pos = dist * lam / input_spacing
        return replica_pos
 
    def get_wavelength(self):
        return self[0].get_wavelength()

    def get_sum(self):
        x = self.wavefronts[0].get_abscissas()
        y = [w.get_intensity() for w in self]
        ys = np.sum(y, axis=0)
        return x, ys

    def plot(self, n=None, show_sum=True, ax=None):
        return self.show(n=n, show_sum=show_sum, ax=ax)

    def show(self, n=None, show_sum=True, ax=None):
        if ax is None:
            ax = plt.gca()
        x = self.wavefronts[0].get_abscissas()
        if show_sum:
            y = [w.get_intensity() for w in self]
            ys = np.sum(y, axis=0)
            ax.plot(x, ys, label="sum", color="0.7", lw=2)
        if n is None:
            n = range(0, 0)
        elif isinstance(n, str) and n == "all":
            n = range(self.n)
        elif isinstance(n, int):
            n = (n,)
        for i in n:
            y = self.wavefronts[i].get_intensity()
            ax.plot(x, y, label=str(i), lw=0.5)
            if len(n) > 1:
                plt.pause(0.01)
                input(f"ok to continue (mode {i}/{len(n)})")

    def propagate(
        self, dist=30, pos=None, magnification=1, auto_tune=True, verbose=False
    ):
        if pos is not None:
            dist = pos - self.xpos
        else:
            pos = self.xpos + dist
        return self.free_space(
            dist=dist, magnification=magnification, auto_tune=auto_tune, verbose=verbose
        )

    def free_space(
        self,
        dist=30,
        pos=None,
        magnification=1,
        auto_tune=True,
        verbose=False,
        magnification_N=1,
    ):
        """ magnification_N used only for free_space_integral """
        if pos is not None:
            dist = pos - self.xpos
        else:
            pos = self.xpos + dist
        # verify Shannon conditions
        w0 = self.wavefronts[0]
        w0x = w0.get_abscissas()
        input_spacing = w0x[1] - w0x[0]
        output_win = w0x[-1] * magnification
        lam = w0.get_wavelength()
        replica_pos = dist * lam / input_spacing
        if replica_pos < 2 * output_win:
            print("replica_pos", f"{replica_pos*1e3:.1f} mm")
            print("Warning, it might run into replica problems")

        mag, wf0 = free_space(
            self.wavefronts[0],
            dist=dist,
            magnification=magnification,
            auto_tune=auto_tune,
            verbose=verbose,
            magnification_N=magnification_N,
        )
        wfs = [
            wf0,
        ]
        for _wf in self.wavefronts[1:]:
            _, temp = free_space(
                _wf,
                dist=dist,
                magnification=mag,
                auto_tune=False,
                magnification_N=magnification_N,
            )
            wfs.append(temp)

        wfs = WolfryWaveFronts(wfs, xpos=pos)

        if verbose:
            x, y = wfs.get_sum()
            ym = y.max() / 100
            if y[0] > ym or y[-1] > ym:
                print("Range might be too small, try increasing magnification")
        return wfs

    def free_space_fresnel_zoom(self, dist=30, pos=None, magnification=1):
        if pos is not None:
            dist = pos - self.xpos
        else:
            pos = self.xpos + dist
        wfs = [
            free_space_fresnel_zoom(w, dist=dist, magnification=magnification)
            for w in self
        ]
        return WolfryWaveFronts(wfs, xpos=pos)

    def free_space_fresnel_1fft(self, dist=30, pos=None):
        if pos is not None:
            dist = pos - self.xpos
        else:
            pos = self.xpos + dist
        wfs = [
            free_space_fresnel_1fft(w, dist=dist)
            for w in self
        ]
        return WolfryWaveFronts(wfs, xpos=pos)


    def free_space_integral(
        self, dist=30, pos=None, magnification=1, magnification_N=1
    ):
        if pos is not None:
            dist = pos - self.xpos
        else:
            pos = self.xpos + dist
        wfs = [
            free_space_integral(
                w,
                dist=dist,
                magnification=magnification,
                magnification_N=magnification_N,
            )
            for w in self
        ]
        return WolfryWaveFronts(wfs, xpos=pos)

    def lens(self, n=1, radius=1e-3, material="Be", web_thickness=30e-6):
        wlens = wolfry_lens(
            n=n, radius=radius, material=material, web_thickness=web_thickness
        )
        wfs = [wlens.applyOpticalElement(w) for w in self]
        return WolfryWaveFronts(wfs, xpos=self.xpos)

    def slit(self, blade1=-1e-3, blade2=1e-3):
        wolfry_slit = slit(blade1=blade1, blade2=blade2)
        wfs = [wolfry_slit.applyOpticalElement(w) for w in self]
        return WolfryWaveFronts(wfs, xpos=self.xpos)

    def as_tally(self):
        from orangecontrib.esrf.wofry.util.tally import TallyCoherentModes

        tally = TallyCoherentModes()
        for w in self.wavefronts:
            tally.append(w)
        return tally


def free_space_fresnel_zoom(wf, dist=30, magnification=1):
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D
    from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
    from wofry.propagator.propagator import (
        PropagationManager,
        PropagationElements,
        PropagationParameters,
    )

    wf = wf.duplicate()
    optical_element = WOScreen1D()
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(
        optical_element=optical_element,
        coordinates=ElementCoordinates(p=dist, q=0, angle_radial=0, angle_azimuthal=0),
    )
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(
        wavefront=wf, propagation_elements=propagation_elements
    )
    # self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters(
        "magnification_x", float(magnification)
    )
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(
        propagation_parameters=propagation_parameters, handler_name="FRESNEL_ZOOM_1D"
    )
    return output_wavefront

def free_space_fresnel_1fft(wf, dist=30):
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D
    from wofryimpl.propagator.propagators1D.fresnel_1fft import Fresnel1D1FFT
    from wofry.propagator.propagator import (
        PropagationManager,
        PropagationElements,
        PropagationParameters,
    )

    wf = wf.duplicate()
    optical_element = WOScreen1D()
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(
        optical_element=optical_element,
        coordinates=ElementCoordinates(p=dist, q=0, angle_radial=0, angle_azimuthal=0),
    )
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(
        wavefront=wf, propagation_elements=propagation_elements
    )
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelD1FFT())
    except:
        pass
    output_wavefront = propagator.do_propagation(
        propagation_parameters=propagation_parameters, handler_name="FRESNEL_1FFT_1D"
    )
    return output_wavefront



def free_space_integral(wf, dist=30, magnification=1, magnification_N=1):
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D
    from wofryimpl.propagator.propagators1D.integral import Integral1D
    from wofry.propagator.propagator import (
        PropagationManager,
        PropagationElements,
        PropagationParameters,
    )

    wf = wf.duplicate()
    optical_element = WOScreen1D()
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(
        optical_element=optical_element,
        coordinates=ElementCoordinates(p=dist, q=0, angle_radial=0, angle_azimuthal=0),
    )
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(
        wavefront=wf, propagation_elements=propagation_elements
    )
    # self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters(
        "magnification_x", float(magnification)
    )
    propagation_parameters.set_additional_parameters(
        "magnification_N", int(magnification_N)
    )
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(Integral1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(
        propagation_parameters=propagation_parameters, handler_name="INTEGRAL_1D"
    )
    return output_wavefront


def use_fresnel(lam=1e-10, a=1e-3, l=10):
    # try to guess which propagator to use based on Fresnel (Fθ²/4<<1) see
    # https://en.wikipedia.org/wiki/Fresnel_diffraction
    F = a ** 2 / l / lam
    # theta = a/l
    # value = F*theta**2/4
    if F > 50:
        return True
    else:
        return False


def use_fresnel_wf(wf, l=10):
    a = find_fwhm(wf)
    lam = wf.get_wavelength()
    return use_fresnel(lam=lam, a=a, l=l)


def _free_space(wf, dist=30, magnification=1, magnification_N=1):
    # try to guess which propagator to use based on Fresnel (Fθ²/4<<1) see
    # https://en.wikipedia.org/wiki/Fresnel_diffraction

    # TO DO improve switching
    if dist > 5:
        return free_space_integral(
            wf, dist=dist, magnification=magnification, magnification_N=magnification_N
        )
    else:
        return free_space_fresnel_zoom(wf, dist=dist, magnification=magnification)


def free_space(
    wavefront,
    dist=30,
    magnification=1,
    magnification_N=1,
    auto_tune=True,
    verbose=False,
):
    wf = _free_space(
        wavefront,
        dist=dist,
        magnification=magnification,
        magnification_N=magnification_N,
    )
    fwhm = find_fwhm(wf)
    x = wf.get_abscissas()
    x_win = x.max() - x.min()
    if auto_tune:
        for i in range(10):
            if verbose:
                print(f"free_space auto_tune {i}: fwhm={fwhm:.3e}, x_win={x_win:.3e}")
            if not np.isfinite(fwhm) or fwhm > x_win / 3:
                magnification = magnification * 1.5
                if verbose:
                    print(f"free_space auto_tune {i}: increasing magnification")
            elif fwhm < x_win / 6:
                magnification = magnification * 0.75
                if verbose:
                    print(f"free_space auto_tune {i}: decreasing magnification")
            else:
                if verbose:
                    print(f"free_space auto_tune {i}: exiting")
                break
            wf = _free_space(
                wavefront,
                dist=dist,
                magnification=magnification,
                magnification_N=magnification_N,
            )
            fwhm = find_fwhm(wf)
            x = wf.get_abscissas()
            x_win = x.max() - x.min()
    return magnification, wf


def wolfry_lens(n=1, radius=1e-3, material="Be", web_thickness=30e-6):
    from orangecontrib.esrf.wofry.util.lens import WOLens1D

    # web_thickness = 30e-6
    thickness = 1e-3

    # simple geometry y=a*x**2 (R=1/(2a)) gives geometrical opening
    aperture = 2 * np.sqrt((thickness - web_thickness) * radius)
    # aperture = 2e-3

    optical_element = WOLens1D.create_from_keywords(
        name="",
        shape=1,
        radius=radius,
        lens_aperture=aperture,
        wall_thickness=web_thickness,
        material=material,
        number_of_curved_surfaces=2,
        n_lenses=n,
        error_flag=0,
        error_file="<none>",
        error_edge_management=0,
        mis_flag=0,
        xc=0,
        ang_rot=0,
        wt_offset_ffs=0,
        offset_ffs=0,
        tilt_ffs=0,
        wt_offset_bfs=0,
        offset_bfs=0,
        tilt_bfs=0,
        verbose=0,
    )

    return optical_element


def lens(wf, radius=1e-3, material="Be", web_thickness=30e-6):
    wlens = wolfry_lens(radius=radius, material=material, web_thickness=web_thickness)
    wf = wf.duplicate()
    output_wavefront = wlens.applyOpticalElement(wf)
    return output_wavefront


def lens_f(wf, f=10, material="Be"):
    l = wf.get_wavelength() * 1e10
    E = 12.398 / l
    radius = sr.crl.get_radius_from_focal_length(E, f, material=material)
    return lens(wf, radius=radius)


def slit(blade1=-1e-3, blade2=1e-3):
    from syned.beamline.shape import Rectangle

    boundary_shape = Rectangle(blade1, blade2, blade1, blade2)
    from wofryimpl.beamline.optical_elements.absorbers.slit import WOSlit1D

    optical_element = WOSlit1D(boundary_shape=boundary_shape)
    return optical_element


def apply_slit(wf, blade1=-1e-3, blade2=1e-3):
    wf = wf.duplicate()
    wolfry_slit = slit(blade1=blade1, blade2=blade2)
    output_wavefront = wolfry_slit.applyOpticalElement(wf)
    return output_wavefront
