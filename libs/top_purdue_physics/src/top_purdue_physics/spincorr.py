import numpy as np
import awkward as ak
import vector
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class SpinObservables:
    b1k: Optional[Union[np.ndarray, ak.Array]] = None
    b1r: Optional[Union[np.ndarray, ak.Array]] = None
    b1n: Optional[Union[np.ndarray, ak.Array]] = None
    b2k: Optional[Union[np.ndarray, ak.Array]] = None
    b2r: Optional[Union[np.ndarray, ak.Array]] = None
    b2n: Optional[Union[np.ndarray, ak.Array]] = None
    ckk: Optional[Union[np.ndarray, ak.Array]] = None
    crr: Optional[Union[np.ndarray, ak.Array]] = None
    cnn: Optional[Union[np.ndarray, ak.Array]] = None
    ckr: Optional[Union[np.ndarray, ak.Array]] = None
    ckn: Optional[Union[np.ndarray, ak.Array]] = None
    crk: Optional[Union[np.ndarray, ak.Array]] = None
    crn: Optional[Union[np.ndarray, ak.Array]] = None
    cnk: Optional[Union[np.ndarray, ak.Array]] = None
    cnr: Optional[Union[np.ndarray, ak.Array]] = None
    cHel: Optional[Union[np.ndarray, ak.Array]] = None
    cHan: Optional[Union[np.ndarray, ak.Array]] = None

def compute_spin_polarizations(
    top: Union[vector.MomentumNumpy4D, ak.Array],
    tbar: Union[vector.MomentumNumpy4D, ak.Array],
    l: Union[vector.MomentumNumpy4D, ak.Array],
    lbar: Union[vector.MomentumNumpy4D, ak.Array],
    ttbar: Optional[Union[vector.MomentumNumpy4D, ak.Array]] = None
) -> SpinObservables:
    """
    Computes the spin polarization values for top and tbar, given vector arrays.
    Returns an instance of SpinObservables with the values.
    """
    if (ttbar is None):
        ttbar = top + tbar
    
    boosted_top = top.boostCM_of(ttbar)
    boosted_tbar = tbar.boostCM_of(ttbar)

    boosted_l = l.boostCM_of(ttbar)
    boosted_l = boosted_l.boostCM_of(boosted_tbar)
    boosted_lbar = lbar.boostCM_of(ttbar)
    boosted_lbar = boosted_lbar.boostCM_of(boosted_top)

    p_axis = vector.obj(x=0, y=0, z=1)
    k_axis = boosted_top.to_xyz().unit()

    cos_scat_angle = k_axis.costheta
    axis_sign_coeff = np.sign(cos_scat_angle) # Does this work for Awkward arrays?

    r_axis = (axis_sign_coeff * (p_axis - (cos_scat_angle * k_axis))).unit()
    n_axis = (axis_sign_coeff * p_axis.cross(k_axis)).unit()

    polarizations = {
        "b1k": np.cos(boosted_lbar.deltaangle(k_axis)),
        "b1r": np.cos(boosted_lbar.deltaangle(r_axis)),
        "b1n": np.cos(boosted_lbar.deltaangle(n_axis)),
        "b2k": np.cos(boosted_l.deltaangle(-1 * k_axis)),
        "b2r": np.cos(boosted_l.deltaangle(-1 * r_axis)),
        "b2n": np.cos(boosted_l.deltaangle(-1 * n_axis)),
    }
    return SpinObservables(**polarizations)

def compute_spin_correlations(
    top: Union[vector.MomentumNumpy4D, ak.Array],
    tbar: Union[vector.MomentumNumpy4D, ak.Array],
    l: Union[vector.MomentumNumpy4D, ak.Array],
    lbar: Union[vector.MomentumNumpy4D, ak.Array],
    ttbar: Optional[Union[vector.MomentumNumpy4D, ak.Array]] = None
) -> SpinObservables:
    """
    Computes spin polarizations and correlations.
    """
    correlations = compute_spin_polarizations(top, tbar, l, lbar, ttbar)

    correlations.ckk = correlations.b1k * correlations.b2k
    correlations.crr = correlations.b1r * correlations.b2r
    correlations.cnn = correlations.b1n * correlations.b2n
    correlations.ckr = correlations.b1k * correlations.b2r
    correlations.ckn = correlations.b1k * correlations.b2n
    correlations.crk = correlations.b1r * correlations.b2k
    correlations.crn = correlations.b1r * correlations.b2n
    correlations.cnk = correlations.b1n * correlations.b2k
    correlations.cnr = correlations.b1n * correlations.b2r

    return correlations

def compute_partial_spin_observables(
    top: Union[vector.MomentumNumpy4D, ak.Array],
    tbar: Union[vector.MomentumNumpy4D, ak.Array],
    l: Union[vector.MomentumNumpy4D, ak.Array],
    lbar: Union[vector.MomentumNumpy4D, ak.Array],
    ttbar: Optional[Union[vector.MomentumNumpy4D, ak.Array]] = None
) -> SpinObservables:
    """
    Computes spin polarizations, diagonal correlation elements, and cHel.
    """
    polarizations = compute_spin_polarizations(top, tbar, l, lbar, ttbar)

    polarizations.ckk = polarizations.b1k * polarizations.b2k
    polarizations.crr = polarizations.b1r * polarizations.b2r
    polarizations.cnn = polarizations.b1n * polarizations.b2n

    polarizations.cHel = -(polarizations.ckk + polarizations.crr + polarizations.cnn)
    # polarizations.cHan = polarizations.ckk - polarizations.crr - polarizations.cnn

    return polarizations

def compute_all_spin_observables(
    top: Union[vector.MomentumNumpy4D, ak.Array],
    tbar: Union[vector.MomentumNumpy4D, ak.Array],
    l: Union[vector.MomentumNumpy4D, ak.Array],
    lbar: Union[vector.MomentumNumpy4D, ak.Array],
    ttbar: Optional[Union[vector.MomentumNumpy4D, ak.Array]] = None
) -> SpinObservables:
    """
    Computes spin polarizations and correlations, and cHel and cHan.
    """
    correlations = compute_spin_correlations(top, tbar, l, lbar, ttbar)

    correlations.cHel = -(correlations.ckk + correlations.crr + correlations.cnn)
    correlations.cHan = correlations.ckk - correlations.crr - correlations.cnn

    return correlations
