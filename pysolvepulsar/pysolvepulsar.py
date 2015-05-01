#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
pysolvepulsar:  Algorithmic timing package

"""

from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.linalg as sl, scipy.stats as sst
import logging
import libstempo as lt
import os, glob, sys
import copy

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

tempo2_excludepars = ['START', 'FINISH', 'PEPOCH', 'POSEPOCH', 'DMEPOCH', 'EPHVER']

tempo2_nonmandatory = ['DM']

class PulsarSolver(object):
    """ This class provides a user interface to algorithmic timing methods

    This is the base algorithmic timing class. An instance of this class
    represents a working environment in which a single pulsar can be solved. It
    has to be initialized with a Tempo2 parfile and timfile through libstempo.


    """

    def __init__(self, parfile, timfile, priors=None, logfile=None,
            loglevel=logging.DEBUG):
        """
        Initialize the pulsar solver by loading a libstempo object from a
        parfile and a timfile, and by setting the priors from a prior dictionary

        :param parfile:
            Filename of a tempo2 parfile

        :param timfile:
            Filename of a tempo2 timfile

        :param priors:
            Dictionary describing the priors. priors[key] = (val, err)

        :param logfile:
            Name of the logfile

        :param loglevel:
            Level of logging
        """
        self._psr = lt.tempopulsar(parfile, timfile, dofit=False)
        self.init_sorting_map()

        # Set the logger
        if logfile is None:
            logfile = parfile + '.log'
        self.set_logger(logfile, loglevel)

        if priors is None:
            self._logger.info("No prior given: creating from the psr object.")
            priors = self.create_prior_from_psr()
        self.init_priors(priors)

    def set_logger(self, logfile, loglevel):
        """
        Initialize the logger

        :param logfile:
            Name of the logfile

        :param loglevel:
            Level of logging

        REMINDER, we have the following logging levels:
            self._logger.debug('debug message')
            self._logger.info('info message')
            self._logger.warn('warn message')
            self._logger.error('error message')
            self._logger.critical('critical message')
        """
        logname = os.path.basename(logfile.rstrip('/'))
        self._logger = logging.getLogger(logname)
        self._logger.setLevel(loglevel)

        # Create console handler and set level to debug
        #ch = logging.StreamHandler()
        #ch.setLevel(loglevel)

        # Create formatter
        #formatter = logging.Formatter(
        #        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to ch
        #ch.setFormatter(formatter)

        # Add ch to logger
        #self._logger.addHandler(ch)

    def init_sorting_map(self):
        """
        Provide the sorting map and its inversion

        In order to study the coherence length, we need the toas to be sorted.
        We therefore create an index map to and from the libstempo object. We
        use mergesort so that it keeps identical elements in order.
        """
        self._isort = np.argsort(self._psr.toas(), kind='mergesort')
        self._iisort = np.zeros_like(self._isort, dtype=np.int)
        for ii, p in enumerate(self._isort):
            self._iisort[p] = ii

    def create_prior_from_psr(self):
        """
        Create the prior dictionary from the pulsar object

        Create a prior dictionary from the information in the parfile/libstempo
        object
        """
        priors = OrderedDict()
        for key in self._psr.pars(which='set'):
            if not self._psr[key].fit and not key in tempo2_excludepars \
                    and not key in tempo2_nonmandatory:
                self._logger.info(
                    'Fitting for {0} automatically turned on.'.format(key))
                self._psr[key].fit = True

            if self._psr[key].err <= 0.0 and self._psr[key].fit:
                self._logger.error('Prior for {0} cannot have 0 width'.
                        format(key))
            
            priors[key] = (self._psr[key].val, self._psr[key].err)

        return priors

    def init_priors(self, priors):
        """
        Initialize the priors from a prior dictionary. This dictionary should be
        organized as follows:

        priors[key] = (parval, parsigma)

        This corresponds to a Gaussian prior, with mean parval, and standard
        deviation parsigma.

        The libstempo object will be initialized such that the constant 'Offset'
        is always fit for, with a prior sigma equal to twice the pulse period,
        and all other 'set' parameters are only fit for if the prior is set. Set
        parameters that do not have a prior are not fitted for, and a warning is
        displayed.
        """
        # First un-set all the parameters from the par-file
        un_fit = []
        un_set = []
        for key in self._psr.pars(which='set'):
            if self._psr[key].set:
                un_set.append(key)
            if self._psr[key].fit:
                un_fit.append(key)

            self._psr[key].fit = False
            self._psr[key].set = False

        self._pars = OrderedDict()
        self._prpars = OrderedDict()
        self._prerr = OrderedDict()

        # Start with the 'Offset' parameter: a general phase offset that is
        # always unknown and fit for
        self._pars['Offset'] = 0.0
        self._prpars['Offset'] = 0.0
        self._prerr['Offset'] = 2.0 / self._psr['F0'].val   # 2 * P0

        for key in priors:
            self._psr[key].set = True
            self._psr[key].fit = True
            # Do we set the initial value equal to the prior?
            #self._psr[key].val = priors[key][0]

            # We keep the value, so check whether it is somewhat consistent with
            # the prior. Give a warning otherwise
            self.check_value_against_prior(
                    self._psr[key].val, priors[key][0], priors[key][1])

            self._pars[key] = self._psr[key].val
            self._prpars[key] = priors[key][0]
            self._prerr[key] = priors[key][1]

        # Check whether we have any unset parameters we have not addressed
        for key in np.unique(np.append(un_fit, un_set)):
            if not key in priors:
                self._logger.info(
                        "Prior for parameter {0} was not given. Ignoring.")

    def check_value_against_prior(self, parval, prval, prerr, siglevel=0.0001):
        """
        Check whether parval is consistent with N(prval, prerr)

        Check whether parval is consistent with N(prval, prerr), with a
        significance level of siglevel
        """
        pval = sst.norm(np.float(prval), np.float(prerr)).cdf(np.float(parval))

        if pval < 0.5*siglevel or 1-pval < 0.5*siglevel:
            self._logger.warn("Parameter value of {0} in tail of the prior")


class CandidateSolution(object):
    """
    Class that represents a single candidate solution for a pulsar. Often this
    solution will need to be refined further.
    """

    def __init__(self, pars, parerrs, nobs):
        self._pars = pars
        self._parerrs = parerrs

    def init_separate_patches(self, nobs):
        """
        Initialize the individual coherence patches
        """
        self._patches = [[ind] for ind in range(nobs)]
        self._rpn = [[0] for ind in range(nobs)]
