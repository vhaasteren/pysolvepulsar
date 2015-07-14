#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
units:  Tools to deal with the units of libstempo/PINT

"""
from __future__ import print_function
from __future__ import division
import astropy.units as units

# How to deal with units, if they are there. (incomplete)
par_units = {'RAJ': units.radian,
             'DECJ': units.radian,
             'F0': units.Hz,
             'F1': units.Hz/units.s}

def has_astropy_unit(x):
    """
    has_astropy_unit(x):

    Return True/False if x has an astropy unit type associated with it. This is
    useful, because different data types can still have units associated with
    them.
    """
    return hasattr(x,'unit') and isinstance(x.unit, units.core.UnitBase)

def un_unitize(key, x):
    """Return the unitless representation of a parameter"""
    if has_astropy_unit(x):
        if key in par_units:
            rv = x.to(par_units[key]).value
        else:
            NotImplementedError("Key {0} has unknown units".format(key))
    else:
        rv = x

    return rv
