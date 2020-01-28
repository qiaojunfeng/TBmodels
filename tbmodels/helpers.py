# -*- coding: utf-8 -*-
#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
This module contains a helper function to create a list of hoppings from a given matrix (:meth:`matrix_to_hop`).
"""

import typing as ty

import numpy as np
from fsc.export import export


@export
def matrix_to_hop(
    mat: ty.Collection[ty.Collection[complex]],
    orbitals: ty.Optional[ty.Sequence[int]] = None,
    R: ty.Collection[int] = (0, 0, 0),
    multiplier: float = 1.
) -> ty.List[ty.List[ty.Union[complex, int, np.ndarray]]]:
    r"""
    Turns a square matrix into a series of hopping terms.

    Parameters
    ----------
    mat :
        The matrix to be converted.
    orbitals :
        Indices of the orbitals that make up the basis w.r.t. which the
        matrix is defined. By default (``orbitals=None``), the first
        ``len(mat)`` orbitals are used.
    R :
        Lattice vector for all the hopping terms.
    multiplier :
        Multiplicative constant for the hopping strength.
    """
    if orbitals is None:
        orbitals = list(range(len(mat)))
    hop = []
    for i, row in enumerate(mat):
        for j, x in enumerate(row):
            hop.append([multiplier * x, orbitals[i], orbitals[j], np.array(R, dtype=int)])
    return hop


def utility_w0gauss(x: float, smr_index: int) -> float:
    """from wannier90/src/utility.F90

    the derivative of utility_wgauss:  an approximation to the delta function
    
    (n>=0) : derivative of the corresponding Methfessel-Paxton utility_wgauss
    
    (n=-1 ): derivative of cold smearing:
                1/sqrt(pi)*exp(-(x-1/sqrt(2))**2)*(2-sqrt(2)*x)
    
    (n=-99): derivative of Fermi-Dirac function: 0.5/(1.0+cosh(x))
    
    :param x: [description]
    :type x: float
    :param smr_index: the order of the smearing function
    :type smr_index: int
    :return: [description]
    :rtype: float
    """

    # Fermi-Dirac smearing
    sqrtpm1 = 1.0 / np.sqrt(np.pi)

    if smr_index == -99:
        if np.abs(x) < 36.0:
            w0gauss = 1.00 / (2.00 + np.exp(-x) + np.exp(+x))
            # in order to avoid problems for large values of x in the e
        else:
            w0gauss = 0.0
        return w0gauss

    # cold smearing  (Marzari-Vanderbilt)
    if smr_index == -1:
        arg = np.min(200.0, (x - 1.00 / np.sqrt(2.00))**2)
        w0gauss = sqrtpm1 * np.exp(-arg) * (2.00 - np.sqrt(2.00) * x)
        return w0gauss

    if (smr_index > 10) or (smr_index < 0):
        raise ValueError('higher order smearing is untested and unstable')

    # Methfessel-Paxton
    arg = min(200.0, x**2)
    w0gauss = np.exp(-arg) * sqrtpm1
    if smr_index == 0:
        return w0gauss

    hd = 0.00
    hp = np.exp(-arg)
    ni = 0
    a = sqrtpm1
    for i in range(1, smr_index + 1):
        hd = 2.00 * x * hp - 2.00 * ni * hd
        ni = ni + 1
        a = -a / (i * 4.00)
        hp = 2.00 * x * hd - 2.00 * ni * hp
        ni = ni + 1
        w0gauss = w0gauss + a * hp

    return w0gauss
