#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test the __repr__ of the Model class.
"""

import pytest
import numpy as np  # pylint: disable=unused-import

import tbmodels  # pylint: disable=unused-import
from tbmodels._sparse_matrix import csr  # pylint: disable=unused-import

from parameters import T_VALUES


@pytest.mark.parametrize('t', T_VALUES)
def test_repr_reload(t, get_model, models_equal):
    """Check that the repr() of a Model can be evaluated to get the same model."""
    model1 = get_model(*t)
    model2 = eval(repr(model1))  # pylint: disable=eval-used
    models_equal(model1, model2, ignore_sparsity=True)
