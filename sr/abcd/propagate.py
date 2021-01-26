import os
import numpy as np
import sympy
from matplotlib import pyplot as plt
import scipy

from datastorage import DataStorage as ds
import datastorage

from .. import undulator
#from .abcd import 
from .optics import hard_aperture
from .optics import lens
from .useful_beams import id18h
from ..crl import LensBlock, Transfocator

transfocator = Transfocator([LensBlock(2 ** i, radius=500e-6) for i in range(8)])


def size_at_dist(beam, f, dist):
    # print(f"{f:12.2f}", end="\r")
    op = lens(f)
    b = beam.apply(op)
    b = b.propagate(dist)
    return float(b.rms_size * 2.35)


def find_fl_to_get_size(beam, dist, size, verbose=True, retry=False):
    def tominimize(f):
        return abs(size_at_dist(beam, f, dist) - size)

    if beam.radius > 0:
        bracket = float(beam.radius * 0.9), float(beam.radius * 1.1)
    else:
        bracket = (35, 45)

    res = scipy.optimize.minimize_scalar(
        tominimize, bracket=bracket, method="golden", tol=1e-3
    )

    size_with_best_fl = size_at_dist(beam, res.x, dist)
    size_without_optics = float(beam.propagate(dist).rms_size * 2.35)

    if abs(size_without_optics - size) < abs(size_with_best_fl - size):
        fl = np.inf
        if verbose:
            print(f"Looked for best FL, found best match without optics")
    else:
        fl = res.x

    if verbose:
        print(
            f"size@dist: found FL of {res.x:.1f}m after {res.nit} iteration, asked for {size*1e6:.1f}μm, got {size_at_dist(beam, res.x, dist)*1e6:.1f}μm"
        )
    return fl

    F = np.arange(5, 50.5, 0.5)
    s = [size_at_dist(beam, f, dist) for f in F]
    s = np.asarray(s)
    idx = np.argmin(np.abs(s - size))
    if verbose:
        for fi, si in zip(F, s):
            print(f"{fi:.1f} {abs(si-size)*1e6:.0f}")
        print(
            f"Will be using focus length of {F[idx]:.1f}; it gives a size of {s[idx]*1e6:.0f} um (objective was {size*1e6:.0f}"
        )
    return F[idx]


def find_fl(beam, dist, verbose=True):
    return find_fl_to_get_size(beam, dist, 0, verbose=verbose)


def propagate(
    beam=id18h,
    optics=[[40, "x1", "coll"], [150, "x1", "focus@200"]],
    z=np.arange(0, 230, 0.5),
    use_transfocator=True,
    transfocator=transfocator,
    fixed_f = None,
    fname=None,
    force=False,
):
    """ 
    beam is a GSM or a GSM_Numeric beam

    optics = [
        [ pos1, aperture1, coll|flat|focus@dist|sizeUM@dist,None ],
        [ pos2, aperture2, coll|flat|focus@dist|sizeUM@dism,None ],
        [ .................................... ],
        ]

    sizeUM@dist means that FL to get as close as possible to UM um  at a certain
    distance will be calculated (and used)


    if optics element starts with 'crl_', a CRL lens set will be used.
    The use of lenses can be 'forced' using use_transfocator=True
    transfocator is either an istance of crl.Transfocator or a dictionary of 
    transfocator instances (key is the distance from source)

    fixed_f is to take into account constrains in one direction (e.g. h)
    imposed by having fixed focal length or lenses in v direction
        It should be a dictionary of focal length or CRL lenset (key is distance)

    aperture can be an:
      - absolute value, "x1.2" is 1.2 times the FWHM CL, coherence length, None

    """
    print("WARNING: hard apertures are implemented as shown in Vartanyans 2013 JSR paper")
    print("         They are an approximations (valid in the far field?)")

    if not force and fname is not None and os.path.isfile(fname):
        data = datastorage.read(fname)
        return data

    info = ds()

    energy = 12.398 / (beam.wavelen * 1e10)

    positions = [0]
    apertures = [None]
    focal_lengths = [None]
    desired_focal_lengths = [None]
    lenses = [None]
    beams_before_aperture_before_optics = [beam]
    beams_after_aperture_before_optics = [beam]
    beams_after_aperture_after_optics = [beam]
    log = ["source"]

    for i, o in enumerate(optics, start=1):
        _log = []
        _log.append(f"Working on optics element {i}")
        _pos, _aperture, _element = o

        if isinstance(_element, (int, float)):
            fl = _element
            _element = ""  # to make if statements below happy
            _log.append(f"Explicitly asked to use {fl:.3f} focal_length")

        if _element is not None and _element.startswith("crl_"):
            _use_transfocator = True
            _element = _element[:4]
            _log.append(
                "Will use CRL for this element because element name starts with crl_"
            )
        else:
            _use_transfocator = False

        _use_transfocator = _use_transfocator or use_transfocator

        positions.append(_pos)

        dpos = _pos - positions[i - 1]

        # babo = before aperture, before optics
        babo = beams_after_aperture_after_optics[-1].propagate(dpos)

        ### WORK ON APERTURE ###
        # aabo = after aperture, before optics
        if _aperture is None:
            aabo = babo
            apertures.append(None)
            _log.append("no aperture for this element")
        else:
            if isinstance(_aperture, str):
                # "x1.2"
                aperture_as_fraction_of_cl = float(_aperture[1:])
                _aperture = aperture_as_fraction_of_cl * babo.rms_cl * 2.35
                _log.append(
                    f"aperture defined as {aperture_as_fraction_of_cl:.2f} of FWHM CL {babo.rms_cl * 2.35:.2e} m"
                )
            _log.append(f"aperture of {_aperture:.2e} m")
            apertures.append(_aperture)
            _aperture = hard_aperture(_aperture)
            aabo = babo.apply(_aperture)

        ### WORK ON FOCUSING OPTICS ###
        # aaao = after aperture, after optics
        if _element is None:
            aaao = aabo
            focal_lengths.append(None)
            desired_focal_lengths.append(None)
            _log.append(f"No focusing optics for this element")
        else:
            if fixed_f is not None and _pos in fixed_f:
                c = fixed_f[_pos]
                _log.append("Adding constrain from other direction:"+str(c))
                if isinstance(c,(float,int)):
                    c = abcd.optics.Lens(c)
                aabo = aabo.apply(c)
            if _element[:4].lower() == "coll":
                fl = aabo.radius
                _log.append(
                    f"Asked for collimating, will try to use focal length = radius of curvature {fl:.3e} m"
                )
            if _element[:5].lower() == "focus":
                where_to_focus = float(_element.split("@")[1])
                dist_to_focus = where_to_focus - _pos
                _log.append(
                    f"Asked for focusing at {where_to_focus:.3f} m from source (meaning {dist_to_focus:.3f} m from optical element)"
                )
                fl = find_fl(aabo, dist_to_focus)
                _log.append(f"Found the FL needed: {fl:.3f} m")
            if _element[:4].lower() == "size":
                size, where = _element[4:].split("@")
                size = float(size) * 1e-6
                dist = float(where) - _pos
                _log.append(
                    f"Asked for imaging beam to a size of {size:.3e} m at a distance of {float(where):.3f} m (meaning {float(dist):.3f} m from optical element)"
                )
                fl = find_fl_to_get_size(aabo, dist, size)
                _log.append(f"Found the FL needed: {fl:.3f} m")
            desired_focal_lengths.append(fl)
            if _use_transfocator:
                _log.append(
                    f"Using transfocator, finding best combination for FL {fl:.3f} m"
                )
                if not isinstance(transfocator,Transfocator):
                    _transfocator = transfocator[_pos]
                else:
                    _transfocator = transfocator
                ret_transf = _transfocator.find_best_set_for_focal_length(
                    energy=energy, focal_length=fl,
                    accuracy_needed=min(fl / 1000, 0.1),
                    beam_fwhm=None,
                )
                fl = ret_transf.focal_length
                _log.append(f"Using transfocator, found set with FL of {fl:.2f} m")
                _log.append(
                    f"Using transfocator, using set {str(ret_transf.best_lens_set)}"
                )
                fl_obj = ret_transf.best_lens_set
                _log.append(fl_obj)
                lenses.append(str(fl_obj))
            else:
                lenses.append(None)
                fl_obj = lens(fl)
            focal_lengths.append(fl)
            if fl == np.inf:
                aaao = aabo
            else:
                aaao = aabo.apply(fl_obj)
        beams_before_aperture_before_optics.append(babo)
        beams_after_aperture_before_optics.append(aabo)
        beams_after_aperture_after_optics.append(aaao)
        log.append(_log)
        print("\n".join([l for l in _log if isinstance(l, str)]))
    positions = np.asarray(positions)
    info = ds(
        log=log,
        inputs=optics,
        optics_positions=positions,
        apertures=apertures,
        focal_lengths=focal_lengths,
        desired_focal_lengths=desired_focal_lengths,
        beams_before_aperture_before_optics=beams_before_aperture_before_optics,
        beams_after_aperture_before_optics=beams_after_aperture_before_optics,
        beams_after_aperture_after_optics=beams_after_aperture_after_optics,
    )

    size = np.zeros_like(z)
    cl = np.zeros_like(z)

    for i, zi in enumerate(z):
        print(f"calculating {i}/{len(z)}", end="\r")
        # calc which beam to use
        div = np.floor_divide(positions, zi)
        temp_idx = np.ravel(np.argwhere(div == 0))
        if len(temp_idx) == 0:
            idx = 0
        else:
            idx = temp_idx[-1]
        beam = beams_after_aperture_after_optics[idx]
        dpos = zi - positions[idx]
        b = beam.propagate(dpos)
        size[i] = b.rms_size * 2.35 * 1e6
        cl[i] = b.rms_cl * 2.35 * 1e6
    divergence = np.gradient(size, z)
    ret = ds(
        z=z,
        divergence=divergence,
        fwhm_size=size,
        fwhm_cl=cl,
        info=info,
    )
    if fname is not None:
        ret.save(fname)
    return ret

