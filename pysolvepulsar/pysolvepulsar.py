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

# Parameters that have tempo2 status as parameter, but are not model parameters
tempo2_excludepars = ['START', 'FINISH', 'PEPOCH', 'POSEPOCH', 'DMEPOCH', 'EPHVER']

# Parameters that are not always required to be fit for
tempo2_nonmandatory = ['DM']

# Parameter bounds (min, max):
tempo2_parbounds = {'F0': (0.0, np.inf),
                    'SINI': (0.0, 1.0),
                    'ECC': (0.0, 1.0),
                    'E': (0.0, 1.0)
                    }

tempo2_parbounds_front = {'JUMP': ('P0', -0.5, 0.5)}

class PulsarSolver(object):
    """ This class provides a user interface to algorithmic timing methods

    This is the base algorithmic timing class. An instance of this class
    represents a working environment in which a single pulsar can be solved. It
    has to be initialized with a Tempo2 parfile and timfile through libstempo.

    """

    def __init__(self, parfile, timfile, priors=None, logfile=None,
            loglevel=logging.DEBUG):
        """Initialize the pulsar solver

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
        self.load_pulsar(parfile, timfile)
        self.init_sorting_map()

        # Set the logger
        if logfile is None:
            logfile = parfile + '.log'
        self.set_logger(logfile, loglevel)

        # Initialize the priors
        if priors is None:
            self._logger.info("No prior given. Using from the psr object/parfile.")
            priors = self.create_prior_from_psr()
        else:
            self._logger.info("Prior given separately. Ignore those in the psr object.")
        self.init_priors(priors)

        # Start with one candidate solution (based on the prior)
        cand = CandidateSolution()
        cand.set_start_solution(self._prpars, self.nobs)
        self._candidates = [cand]

        # TODO: shift the PEPOCH
        self.set_pepoch_to_center()

    def load_pulsar(self, parfile, timfile):
        """Read the pulsar object from par/tim file

        Use libstempo to read in a tempo2 pulsar from a parfile and a timfile
        """
        self._psr = lt.tempopulsar(parfile, timfile, dofit=False)
        self.nobs = len(self._psr.toas())

    def set_logger(self, logfile, loglevel):
        """Initialize the logger

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
        """Provide the sorting map and its inversion (NOT USED)

        In order to study the coherence length, we need the toas to be sorted.
        We therefore create an index map to and from the libstempo object. We
        use mergesort so that it keeps identical elements in order.

        NOTE: These mappings are not used anymore
        """
        self._isort = np.argsort(self._psr.toas(), kind='mergesort')
        self._iisort = np.zeros_like(self._isort, dtype=np.int)
        for ii, p in enumerate(self._isort):
            self._iisort[p] = ii

    def get_prior_bounds(self, key):
        """Given the parameter key, obtain the prior bound

        Get the prior bound for the parameter key.
        """
        parmin, parmax = (-np.inf, np.inf) if not key in tempo2_parbounds \
                                            else tempo2_parbounds[key]
        if key[:4] in tempo2_parbounds_front:
            # We have a JUMP or something like that
            key_link = tempo2_parbounds_front[key[:4]][0]
            par_mult = 1.0 if key_link is None else self._psr[key_link].val
            parmin = par_mult * tempo2_parbounds_front[key[:4]][1]
            parmax = par_mult * tempo2_parbounds_front[key[:4]][2]

        return parmin, parmax

    def create_prior_from_psr(self):
        """Create the prior dictionary from the pulsar object

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
            elif self._psr[key].fit and (key in tempo2_excludepars or key in
                    tempo2_nonmandatory):
                self._psr[key].fit = False
                self._logger.info(
                    'Fitting for {0} automatically turned off.'.format(key))

            if self._psr[key].err <= 0.0 and self._psr[key].fit:
                self._logger.error('Prior for {0} cannot have 0 width'.
                        format(key))

            # TODO: tempo2_nonmandatory is not always excluded. Should not be
            #       the case
            if not key in tempo2_excludepars and not key in tempo2_nonmandatory:
                parmin, parmax = self.get_prior_bounds(key)
                priors[key] = (self._psr[key].val, self._psr[key].err, \
                        parmin, parmax)

        return priors

    def init_priors(self, priors):
        """Initialize the priors from a prior dictionary.

        Initialize the priors from a prior dictionary. This dictionary values should be
        organized as (parval, parsigma)

        :param priors:
            Dictionary with the priors for the paramters. Organized as
            priors[key] = (parval, parsigma), with distribution as a normal
            Gaussian distribution N(parval, parsigma)

        This corresponds to a Gaussian prior, with mean parval, and standard
        deviation parsigma.

        We use the libstempo routines without the 'Offset' parameter. We are
        including jumps for each coherency patch, so the 'Offset' is
        unnecessary. The jumps have a prior sigma equal to twice the pulse
        period, and all other 'set' parameters are only fit for if the prior is
        set. Set parameters that do not have a prior are not fitted for, and a
        warning is displayed.
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
        self._prmin = OrderedDict()
        self._prmax = OrderedDict()

        # Start with the 'Offset' parameter: a general phase offset that is
        # always unknown and fit for
        #self._pars['Offset'] = 0.0
        #self._prpars['Offset'] = 0.0
        #self._prerr['Offset'] = 2.0 / self._psr['F0'].val   # 2 * P0
        # NOTE: We should not have placed a prior on the phase offset anyway!

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
            self._prmin[key] = priors[key][2]
            self._prmax[key] = priors[key][3]

        # Check whether we have any unset parameters we have not addressed
        for key in np.unique(np.append(un_fit, un_set)):
            if not key in priors and not key in tempo2_excludepars and not \
                    key in tempo2_nonmandatory:
                self._logger.info(
                        "Prior for parameter {0} was not given. Ignoring.")

    def check_value_against_prior(self, parval, prval, prerr, siglevel=0.0001):
        """Check whether parval is consistent with N(prval, prerr)

        Check whether parval is consistent with N(prval, prerr), with a
        significance level of siglevel
        """
        pval = sst.norm(np.float(prval), np.float(prerr)).cdf(np.float(parval))

        if pval < 0.5*siglevel or 1-pval < 0.5*siglevel:
            self._logger.warn("Parameter value of {0} in tail of the prior")

    def make_pars_respect_constraints(self, cand, ass_cmin, ass_cmax):
        """Make the parameters in `cand` respect the linear constraints

        Make the parameters in `cand` respect the linear constraints, given by
        the min/max boundaries. Return the new (temporary) candidate object, and
        the mask of parameters that are being fixed.

        :param cand:
            The candidate solution from where to start

        :param ass_cmin:
            Assumed constraint minimum bounds to respect

        :param ass_cmax:
            Assumed constraint minimum bounds to respect
        """
        cmin = ass_cmin if ass_cmin is not None \
                        else np.zeros(cand.npars, dtype=np.bool)
        cmax = ass_cmax if ass_cmax is not None \
                        else np.zeros(cand.npars, dtype=np.bool)
        if np.any(np.logical_and(cmin, cmax)):
            raise ValueError("Cannot satisfy both min and max constraints.")

        prparsmin = np.array([self._prmin[key] for key in self._prmin])
        prparsmax = np.array([self._prmax[key] for key in self._prmax])

        newcand = CandidateSolution(cand)
        newcand.pars[cmin] = prparsmin[cmin]
        newcand.pars[cmax] = prparsmax[cmax]

        return newcand, np.logical_not(np.logical_or(cmin, cmax))

    def set_pepoch_to_center(self):
        """Set the PEPOCH value to the middle of the dataset, if necessary

        Set the tempo2 PEPOCH value to the middle of the dataset, if the value
        is not somewhere around the middle already. If changed, transform the
        value of F1 as well.
        """
        #tmid = 0.5 * (np.max(psr.toas()) - np.min(psr.toas()))
        #self._psr['PEPOCH'] = tmid
        #self._psr['PEPOCH'].set = True
        #self._psr['PEPOCH'].fit = False
        self._logger.warn("PEPOCH translation not done yet!")

    def get_mergable_patches(self, cand, offstd=100.0, mergetol=2.0):
        """Given a candidate, see which coherence patches can be merged

        Given a candidate solution, use prediction to see which coherence
        patches are within a coherence length, so that they can be merged.

        :param cand:
            CandidateSolution candidate

        :param offstd:
            When using `fitpatch`, this parameter sets the width of the prior on
            the jump/offset of all patches

        :param mergetol"
            How much larger the prediction uncertainty can be than the pulse
            period for us to still include a merge (potentially with a phase
            jump)

        :return:
            list of patch lists
        """
        mergables = []

        # Get the mergables for all patches
        for ii, patch in enumerate(cand.get_patches()): 
            mergables.append(self.get_mergable_patches_onepatch(cand, ii,
                    offstd=offstd, mergetol=mergetol))

        # For all non-zero mergables, see if there is not another one with
        # overlap. If so, only use the larges list.
        dd = self.get_linear_solution(cand)

    def get_mergable_patches_onepatch(self, cand, fitpatch, offstd=100.0,
            mergetol=2.0):
        """Given a candidate and a patch, return potential mergable patches

        Given a candidate solution, and a patch number, use prediction to see
        which coherence patches are within a coherence length, so that they can
        be merged.
        
        NOTE: the candidate solution is already assumed to be in a (local)
              maximum of the likelihood. Otherwise, the extrapolation does not
              make sense

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            Patch number we are fitting to

        :param offstd:
            When using `fitpatch`, this parameter sets the width of the prior on
            the jump/offset of all patches

        :param mergetol"
            How much larger the prediction uncertainty can be than the pulse
            period for us to still include a merge (potentially with a phase
            jump)

        :return:
            patch indices
        """
        dd = self.get_linear_solution(cand, fitpatch=fitpatch, offstd=offstd)
        P0 = 1.0 / self._psr['F0'].val
        obsns = np.where(dd['stdrp'] < mergetol*P0)[0]
        mergepatches = []

        so = set(obsns)
        for ii, patch in enumerate(cand.get_patches()):
            sp = set(patch)
            if ii != fitpatch and len(set.intersection(so, sp)) > 0:
                mergepatches.append(ii)

        return mergepatches

    def get_linear_solution(self, cand, fitpatch=None, offstd=100.0):
        """Return the quantities necessary for solving the linear system

        For a given, properly maximized, candidate solution, calculate the
        quantities that are necessary for maximum likelihood calculation and
        prediction.

        "param cand:
            CandidateSolution candidate

        :param fitpatch:
            Number of the patch we are fitting to. This adds this particular
            patch/jump into the timing model (like a tempo2 offset), with a
            Gaussian prior with standard deviation offstd*P0.
            Ideally we would use a flat prior on [0, P0], but we cannot do that
            analytically.

        :param offstd:
            When using `fitpatch`, this parameter sets the width of the prior on
            the jump/offset of all patches

        Use the G-matrix formalism
        """
        dd = dict()
        nobs = self.nobs

        # Create the full design matrix
        # If we use 'fitpatch', we do not include a jump for patch 'fitpatch' in
        # Mj. Instead, we include an offset for the entire dataset (Mo), because
        # we need the uncertainty of the offset when doing prediction.
        # TODO: change this when using without fitpatch??
        Mj = self.get_jump_designmatrix(cand, fitpatch=fitpatch)
        #Mo = 1.0-self.get_jump_designmatrix_onepatch(cand, fitpatch=fitpatch)
        Mo = np.ones((nobs, 0 if fitpatch is None else 1))
        #Mt = self._psr.designmatrix(updatebats=True,
        #        fixunits=True, fixsigns=True, incoffset=False)
        Mt = self.designmatrix(cand)
        Mtot = np.append(Mt, Mo, axis=1)
        Gj = self.get_jump_Gmatrix(cand, fitpatch=fitpatch)
        
        # The parameter identifiers/keys
        parlabelst = list(self._psr.pars(which='fit'))
        parlabelso = ['PatchOffset'] * Mo.shape[1]
        parlabels = parlabelst + parlabelso

        npart = Mtot.shape[1]
        nparj = Mj.shape[1]

        # Create the noise and prior matrices
        Nvec = self.toaerrs(cand)**2
        Phivect = np.array([self._prerr[key]**2 for key in self._prerr])
        Phiveco = np.array([offstd/self._psr['F0'].val]*Mo.shape[1])**2
        Phivec = np.append(Phivect, Phiveco)
        Phivec_inv = 1.0/Phivec
        prparst = np.array([self._prpars[key] for key in self._prpars])
        prparso = np.array([0.0]*Mo.shape[1])
        prpars = np.append(prparst, prparso)
        phipar = prpars * Phivec_inv

        # Set psr parameters, and remove the phase offsets of the patches
        #self._psr.vals(which='fit', values=cand.pars[:])
        #dt_lt = self._psr.residuals(removemean=False)
        dt_lt = self.residuals(cand)
        dt, jvals = self.subtract_jumps(dt_lt, Mj, Nvec)
        pars = cand.pars # np.append(cand.pars, jvals)

        if nparj == nobs:
            # We know nothing but the prior
            Sigma_inv = np.diag(Phivec_inv)
            Sigma = np.diag(Phivec)

            dpars = np.dot(Sigma, phipar)
            rp = 0.0 * dt               # Only true if pars chosen as ML prior
            rr = np.dot(Mtot, np.dot(Sigma, Mtot.T))
            rr = np.dot(Mtot, np.dot(np.diag(Phivec), Mtot.T))

            Np = np.zeros((0,0))
            MNM = np.zeros((0,0))
            MNt = np.zeros(0)
            Mp = np.zeros((0, Mtot.shape[1]))
            dtp = np.zeros(0)
            Np_cf = (np.zeros((0,0)), False)
        elif nparj < nobs:
            # Transform M, N, dt
            Mp = np.dot(Gj.T, Mtot)
            dtp = np.dot(Gj.T, dt)
            Np = np.dot(Gj.T * Nvec, Gj)
            Np_cf = sl.cho_factor(Np)
            MNM = np.dot(Mp.T, sl.cho_solve(Np_cf, Mp))
            Sigma_inv = MNM + np.diag(Phivec_inv)

            # TODO: Use regularized Mtot??
            Sigma_inv_cf = sl.cho_factor(Sigma_inv)
            Sigma = sl.cho_solve(Sigma_inv_cf, np.eye(len(MNM)))

            # Calculate the prediction quantities
            MNt = np.dot(Mp.T, sl.cho_solve(Np_cf, dtp))
            dpars = np.dot(Sigma, MNt + phipar)
            rp = np.dot(Mtot, np.dot(Sigma, MNt))   # Should be approx~0.0
            rr = np.dot(Mtot, np.dot(Sigma, Mtot.T))

        else:
            raise ValueError("# of patches is higher than # of observations")

        # Wrap up in a dictionary, and return
        dd['dt'] = dt
        dd['dtp'] = dtp
        dd['Mj'] = Mj
        dd['Mt'] = Mt
        dd['Mtot'] = Mtot
        dd['Mp'] = Mp
        dd['Gj'] = Gj
        dd['Nvec'] = Nvec
        dd['Np'] = Np
        dd['Np_cf'] = Np_cf
        dd['MNM'] = MNM
        dd['MNt'] = MNt
        dd['Phivec'] = Phivec
        dd['Phivec_inv'] = Phivec_inv
        dd['Sigma_inv'] = Sigma_inv
        dd['Sigma'] = Sigma
        dd['dpars'] = dpars
        dd['rp'] = rp
        dd['stdrp'] = np.sqrt(np.diag(rr))
        dd['parlabels'] = parlabels
        return dd

    def get_jump_designmatrix(self, cand, fitpatch=None):
        """Obtain the design matrix of inter-coherence-patch jumps

        Obtain the design matrix of jumps that disconnect all the coherence
        patches that are not phase connected.

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            If not None, exclude this patch from the designmatrix jumps

        """
        nobs = len(self._psr.toas())
        patches = cand.get_patches(fitpatch=fitpatch)

        npatches = len(patches)
        Mj = np.zeros((nobs, npatches))
        for pp, patch in enumerate(patches):
            Mj[patch,pp] = True

        return Mj

    def get_jump_designmatrix_onepatch(self, cand, fitpatch=None):
        """Obtain the design matrix of one inter-coherence-patch jump

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            Get the designmatrix of the jump for this patch number
        """
        nobs = len(self._psr.toas())
        if fitpatch is not None:
            patch = cand._patches[fitpatch]
            Mj = np.zeros((nobs, 1))
            Mj[patch,0] = True
        else:
            Mj = np.zeros((nobs, 0))
        return Mj

    def get_jump_Gmatrix(self, cand, fitpatch=None):
        """Obtain the complement/G-matrix of get_jump_designmatrix

        Obtain the G-matrix that is complementary to the design matrix of jumps
        that disconnect all the coherence patches that are not phase connected

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            If not None, exclude this patch from the designmatrix jumps

        """
        nobs = len(self._psr.toas())
    
        # If fitpatch is not None, we have one extra column. We therefore need
        # all patches (fitpatch = None in get_patches)
        patches = cand.get_patches(fitpatch=None)
        add_fitpatch_cols = 0 if fitpatch is None else 1

        npatches = len(patches)
        Gj = np.zeros((nobs, nobs-npatches+add_fitpatch_cols))
        ind = 0
        for pp, patch in enumerate(patches):
            ngsize = len(patch)
            if ngsize > 1 and not fitpatch == pp:
                patchGmat = self.get_Gmatrix_onepatch(ngsize)
                Gj[patch,ind:ind+ngsize-1] = patchGmat
                ind += ngsize-1
            elif fitpatch == pp:
                # The G-matrix for this particular patch is just the identity matrix
                for idind in patch:
                    Gj[idind,ind] = True
                    ind += 1

        return Gj

    def get_Gmatrix_onepatch_an(self, n):
        """For a single coherent patch, return the G-matrix

        For a single coherent patch, return the G-matrix that is orthogonal to
        the patch jump designmatrix

        The design matrix for a jump in one patch, looks like this:
        M = (1, 1, 1, ..., 1)^T

        The G-marix is an orthonormal n-by-(n-1) matrix, orthogonal to M. We can
        construct one as follows:
        G = (v2, v3, v4, ..., vn), where vi is the i-th column of the G-matrix,
        and n is the number of rows of M. Note that we start numbering with 2,
        so that in our notation vi is also the i-th column of U, where we use
        the SVD: M = U * Sigma * V^T

        We now define the G-matrix (not unique, for n>2) to be
        v2 = (-1, a, b, b, ... b)^T / sqrt(n)
        v3 = (-1, b, a, b, ... b)^T / sqrt(n)
        vn = (-1, b, b, b, ... a)^T / sqrt(n)

        The values a and b are calculated with the functions below. For n=2, we
        have G = (sqrt(2), -sqrt(2))^T / 2

        NOTE: Turns out that this is only (very!) marginally faster than the SVD
              version for n>15000
        """
        # Calculate auxiliary 'a' and 'b'
        def get_ab(n):
            bm = -2.0/(2.0*(1-n)) + np.sqrt(4.0-4.0*(1-n)) / (2.0*(1-n))
            return 1.0-(n-2)*bm, bm

        U = np.zeros((n, n))
        
        if n > 2:
            sq = 1.0/np.sqrt(n)
            ap, bp = get_ab(n)
            
            U[:,0] = sq
            U[0,1:] = -sq
            for ii in range(1, n):
                U[ii,1:] = bp*sq
                U[ii,ii] = ap*sq
            Gmat = U[:,1:]
        elif n == 2:
            U = np.array([[1.0, 1.0], [1.0, -1.0]]) * 0.5 * np.sqrt(2)
            Gmat = U[:,1:]
        elif n == 1:
            U = np.array([[1.0]])
            Gmat = np.zeros((1, 0))
        else:
            raise ValueError("Cannot get a G-matrix for n={0}".format(n))
        
        return Gmat

    def get_Gmatrix_onepatch(self, ngsize):
        """For a single coherent patch, return the G-matrix

        For a single coherent patch, return the G-matrix that is orthogonal to
        the patch jump designmatrix

        TODO: Do this analytical, not with an SVD
        """
        pvec = np.ones((ngsize, 1))
        U, s, Vt = sl.svd(pvec)
        return U[:,1:]

    def subtract_jumps(self, dt, Mj, Nvec):
        """Subtract the best-fit value of the patch jumps

        Because libstempo does not know about the inter-patch jumps, we need to
        remove the best-fit jumps from the residuals. Fortunately, we know that
        we have converged to a proper solution, so just fitting for the jumps
        should be sufficient.

        :param dt:
            The non-jump-subtracted residuals as given by libstempo

        :param Mj:
            The jump-only design matrix

        :param Nvec:
            The diagonal elements of the noise (co)variance matrix
        """
        MNM = np.dot(Mj.T / Nvec, Mj)
        Sigma_inv = MNM
        cf = sl.cho_factor(Sigma_inv)   # Cond. number of Mj = 1. Always works
        Sigma = sl.cho_solve(cf, np.eye(len(MNM)))
        jvals = np.dot(Sigma, np.dot(Mj.T, dt / Nvec))
        dtj = np.dot(Mj, jvals)

        return dt - dtj, jvals

    def residuals(self, cand, exclude_nonconnected=False):
        """Return the residuals, taking into account pulse number corrections

        Given a candidate solution, return the timing residuals, taking into
        account the relative pulse number corrections within coherent patches.
        When exclude_nonconnected is True, only residuals within coherent
        patches with more than one residual are being returned

        Note: exclude_nonconnected will/might mess up the ordering of the
              residuals

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual
        """
        # Obtain the residuals as tempo2 would calculate them
        self._psr.vals(which='fit', values=cand.pars)
        dt_lt = self._psr.residuals(updatebats=True,
                formresiduals=True, removemean=False)
        pn_lt = self._psr.pulsenumbers(updatebats=False,
                formresiduals=False, removemean=False)
        P0 = 1.0 / self._psr['F0'].val

        # Check against the recorded relative pulse numbers
        dt = dt_lt.copy()
        selection = np.array([], dtype=np.int) if exclude_nonconnected \
                                               else np.arange(len(dt))
        rpns = cand.get_rpns()
        patches = cand.get_patches()
        for pp, patch in enumerate(patches):
            # Relative pulse numbers of patch, and from tempo2
            rpn = np.array(rpns[pp])
            rpn_lt = pn_lt[patch] - pn_lt[patch[0]]

            dt[patch] += (rpn_lt - rpn) * P0

            if exclude_nonconnected and len(patch) > 1:
                selection = np.append(selection, patch)

        return dt[selection]

    def toas(self, cand, exclude_nonconnected=False):
        """Return the toas, possibly excluding non-connected toas

        Return the toas. If exclude_nonconnected is True, then only return the
        toas for which the patches in cand contain more than one toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              residuals

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual
        """
        self._psr.vals(which='fit', values=cand.pars)
        toas = self._psr.toas(updatebats=True)
        selection = np.array([], dtype=np.int) if exclude_nonconnected \
                                               else np.arange(len(toas))

        if exclude_nonconnected:
            patches = cand.get_patches()
            for pp, patch in enumerate(patches):
                # Only return if requested
                if len(patch) > 1:
                    selection = np.append(selection, patch)

        return toas[selection]

    def toaerrs(self, cand, exclude_nonconnected=False):
        """Return the toa uncertainties, possibly excluding non-connected toas

        Return the uncertainties. If exclude_nonconnected is True, then only
        return the errors for which the patches in cand contain more than one
        toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              toas

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return uncertainties within coherent patches with more
            than one residual
        """
        toaerrs = self._psr.toaerrs*1e-6
        selection = np.array([], dtype=np.int) if exclude_nonconnected \
                                               else np.arange(len(toaerrs))

        if exclude_nonconnected:
            patches = cand.get_patches()
            for pp, patch in enumerate(patches):
                # Only return if requested
                if len(patch) > 1:
                    selection = np.append(selection, patch)

        return toaerrs[selection]

    def designmatrix(self, cand, exclude_nonconnected=False):
        """Return the design matrix, possibly excluding non-connected epochs

        Return the design matrix. If exclude_nonconnected is True, then only
        return the rows for which the patches in cand contain more than one toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              residuals

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual
        """
        self._psr.vals(which='fit', values=cand.pars)
        M = self._psr.designmatrix(updatebats=True, fixunits=True,
                fixsigns=True, incoffset=False)

        selection = np.array([], dtype=np.int) if exclude_nonconnected \
                                               else np.arange(M.shape[0])

        if exclude_nonconnected:
            patches = cand.get_patches()
            for pp, patch in enumerate(patches):
                # Only return if requested
                if len(patch) > 1:
                    selection = np.append(selection, patch)

        return M[selection,:]


    def get_chi2_pvalue(self, cand):
        """Return the chi2 p-value for this candidate solution

        For the provided candidate solution, calculate the p-value of the chi^2
        distribution, properly correcting for the number of degrees of freedom.
        We are doing a one-sided test here.

        The chi2 we are calculating is from the likelihood:
        xi^2 =  (dt - M ksi)^T N^{-1} (dt - M ksi)

        Where dt, N, and M are projected with the G-matrix

        :param cand:
            The candidate solution object
        """
        dd = self.get_linear_solution(cand)
        dt = dd['dtp']
        M = dd['Mp']
        N_cf = dd['Np_cf']
        ksi = cand.pars

        # We calculate the p-value from chi2 and the degrees of freedom. If
        # dof=0, the p-value equals 1 (no discrepancy).
        dof = max(len(dt)-M.shape[1], 0)
        pval = 1.0

        if dof > 0:
            xi2 = np.dot(dt, sl.cho_solve(N_cf, dt))
            pval = sst.chi2.sf(xi2, dof)

        return pval

    def fit_constrained_iterative(self, cand, lltol=0.01, maxiter=10,
            offstd=100.0):
        """Perform a constrained, iterative, linear fit

        From a given candidate solution starting point, iterate to the
        maximum-likelihood solution using constrained linear least-squares fits.
        The constraints are only allowed to be bounds on the parameter values
        (no general linear constraints).

        :param cand:
            The candidate solution from where to start the fit
            
        :param lltol:
            The log-likelihood tolerance for accepting a solution

        :param maxiter:
            The maximum number of iterations before complaining something is
            wrong

        :param offstd:
            When using `fitpatch`, this parameter sets the width of the prior on
            the jump/offset of all patches
        """
        # Assume no constraints are necessary, so set all to zero
        ass_cmin = np.zeros(cand.npars, dtype=np.bool)
        ass_cmax = np.zeros(cand.npars, dtype=np.bool)

        dd = self.perform_linear_least_squares_fit(cand, offstd=100.0)

    def perform_linear_least_squares_fit(self, cand, ass_cmin=None,
            ass_cmax=None, fitpatch=None, offstd=100.0):
        """Perform a constrained, linear, least-squares fit

        Perform a constrained, linear, least-squares fit.

        WARNING: the constraints are respected by removing degrees of freedom of
                 the model. This is therefore not a true representation of the
                 actual covariance in the model.

        :param cand:
            The candidate solution from where to start

        :param ass_cmin:
            Assumed constraint minimum bounds to respect

        :param ass_cmax:
            Assumed constraint minimum bounds to respect

        :param fitpatch:
            Number of the patch we are fitting to. This adds this particular
            patch/jump into the timing model (like a tempo2 offset), with a
            Gaussian prior with standard deviation offstd*P0.
            Ideally we would use a flat prior on [0, P0], but we cannot do that
            analytically.

        :param offstd:
            When using `fitpatch`, this parameter sets the width of the prior on
            the jump/offset of all patches
        """
        dd = dict()
        nobs = self.nobs

        newcand, cmask = self.make_pars_respect_constraints(cand,
                ass_cmin, ass_cmax)

        # Create the full design matrix
        # If we use 'fitpatch', we do not include a jump for patch 'fitpatch' in
        # Mj. Instead, we include an offset for the entire dataset (Mo), because
        # we need the uncertainty of the offset when doing prediction.
        Mj = self.get_jump_designmatrix(newcand, fitpatch=fitpatch)
        Mo = np.ones((nobs, 0 if fitpatch is None else 1))
        Mt = self.designmatrix(newcand)[:,cmask]
        Mtot = np.append(Mt, Mo, axis=1)
        Gj = self.get_jump_Gmatrix(newcand, fitpatch=fitpatch)
        
        # The parameter identifiers/keys
        parlabelst = list(np.array(self._psr.pars(which='fit'))[cmask])
        parlabelso = ['PatchOffset'] * Mo.shape[1]
        parlabels = parlabelst + parlabelso

        npart = Mtot.shape[1]
        nparj = Mj.shape[1]

        # Create the noise and prior matrices
        Nvec = self.toaerrs(newcand)**2
        Phivect = np.array([self._prerr[key]**2 for key in self._prerr])[cmask]
        Phiveco = np.array([offstd/self._psr['F0'].val]*Mo.shape[1])**2
        Phivec = np.append(Phivect, Phiveco)
        Phivec_inv = 1.0/Phivec
        prparst = np.array([self._prpars[key] for key in self._prpars])[cmask]
        prparso = np.array([0.0]*Mo.shape[1])
        prpars = np.append(prparst, prparso)
        prpars_delta = np.append(prparst-cand.pars, prparso-prparso)

        # phipar is the ML value of the TM parameters. But: we are calculating
        # it with respect to the current value of the TM parameters.
        phipar = prpars_delta * Phivec_inv  # Instead of: phipar = prpars * Phivec_inv

        # Set psr parameters, and remove the phase offsets of the patches
        dt_lt = self.residuals(newcand)
        dt, jvals = self.subtract_jumps(dt_lt, Mj, Nvec)

        if nparj == nobs:
            # We know nothing but the prior
            Sigma_inv = np.diag(Phivec_inv)
            Sigma = np.diag(Phivec)

            dpars = np.dot(Sigma, phipar)
            rp = 0.0 * dt               # Only true if pars chosen as ML prior
            rr = np.dot(Mtot, np.dot(Sigma, Mtot.T))
            rr = np.dot(Mtot, np.dot(np.diag(Phivec), Mtot.T))

            Np = np.zeros((0,0))
            MNM = np.zeros((0,0))
            MNt = np.zeros(0)
            Mp = np.zeros((0, Mtot.shape[1]))
            dtp = np.zeros(0)
            Np_cf = (np.zeros((0,0)), False)

            loglik = 0.0
            loglik_ml = 0.0
        elif nparj < nobs:
            # Transform M, N, dt
            Mp = np.dot(Gj.T, Mtot)
            dtp = np.dot(Gj.T, dt)
            Np = np.dot(Gj.T * Nvec, Gj)
            Np_cf = sl.cho_factor(Np)
            MNM = np.dot(Mp.T, sl.cho_solve(Np_cf, Mp))
            Sigma_inv = MNM + np.diag(Phivec_inv)

            # TODO: Use regularized Mtot??
            Sigma_inv_cf = sl.cho_factor(Sigma_inv)
            Sigma = sl.cho_solve(Sigma_inv_cf, np.eye(len(MNM)))

            # Calculate the prediction quantities
            MNt = np.dot(Mp.T, sl.cho_solve(Np_cf, dtp))
            dpars = np.dot(Sigma, MNt + phipar)
            rp = np.dot(Mtot, np.dot(Sigma, MNt))   # Should be approx~0.0
            rr = np.dot(Mtot, np.dot(Sigma, Mtot.T))

            # Calculate the log-likelihood
            logdetN2 = np.sum(np.log(np.diag(Np_cf[0])))
            logdetphi2 = 0.5*np.sum(np.log(Phivec))
            xi2dt = 0.5*np.dot(dtp, sl.cho_solve(Np_cf, dtp))
            xi2phi = 0.5*np.sum(phipar**2/Phivec)
            xi2phi1 = 0.5*np.dot(dpars, np.dot(Sigma_inv, dpars))
            xi2_active = 0.5*np.dot(dpars, np.dot(Sigma_inv, dpars))
            # NOTE: xi2_active is zero _if_ we move to ML solution. We are dpars
            #       away from there. That's why we subtract it from loglik
            loglik = -logdetN2-logdetphi2-xi2dt-xi2phi+xi2phi1-xi2_active
            loglik_ml = -logdetN2-logdetphi2-xi2dt-xi2phi+xi2phi1

            # We are not handeling the prior properly...
            # phipar is in xi2phi, but not in xi2_active
            #print("All: ", logdetN2, logdetphi2, xi2dt, xi2phi, xi2phi1, xi2_active)
            #print("Phivec: ", Phivec)
            #print("phipar: ", phipar)
        else:
            raise ValueError("# of patches is higher than # of observations")

        # Wrap up in a dictionary, and return
        dd['dt'] = dt                           # Timing residuals
        dd['dtp'] = dtp                         # Projected timing residuals
        dd['Mj'] = Mj                           # Jump design matrix
        dd['Mt'] = Mt                           # Timing model design matrix
        dd['Mtot'] = Mtot                       # Full design matrix
        dd['Mp'] = Mp                           # Projected design matrix
        dd['Gj'] = Gj                           # Jump design matrix G-matrix
        dd['Nvec'] = Nvec                       # Noise matrix diagonal
        dd['Np'] = Np                           # Projected noise matrix
        dd['Np_cf'] = Np_cf                     # Cholesky factorized Np
        dd['MNM'] = MNM                         # Data-only Sigma^-1
        dd['MNt'] = MNt                         # Data-only parameters
        dd['Phivec'] = Phivec                   # Prior diagnoal
        dd['Phivec_inv'] = Phivec_inv           # Inverse prior diagonal
        dd['Sigma_inv'] = Sigma_inv             # Parameter covariance (inv)
        dd['Sigma'] = Sigma                     # Inverse parameter covariance
        dd['dpars'] = dpars                     # Delta-parameters (the fit)
        dd['rp'] = rp                           # Residual projection
        dd['stdrp'] = np.sqrt(np.diag(rr))      # Residuals projection std
        dd['parlabels'] = parlabels             # Parameter labels
        dd['loglik'] = loglik                   # Log-likelihood
        dd['loglik_ml'] = loglik_ml             # Log-likelihood (ML)
        return dd


class CandidateSolution(object):
    """
    Class that represents a single candidate solution for a pulsar. Often this
    solution will need to be refined further.
    """

    def __init__(self, copyobject=None):
        """Initialize a new candidate. Either empty, or make a copy

        Initialize a new candidate. Either empty, or make a copy

        :param copyobject:
            If not none, the parameters from this object will be copied
        """
        self.init_empty()

        if copyobject is not None:
            self.copy_from_object(copyobject)

    def init_empty(self):
        """Initialize this object with empty attributes
        """
        self._pars = []
        self._patches = []
        self._rpn = []

        # TODO: implement this properly
        self._parent = None
        self._children = []

    def copy_from_object(self, copyobject):
        """Copy all the information from an object into this one.

        Copy all the information from an object into this one. This object will
        be an orphan: the parent/child information is not copied.
        """
        self._pars = copy.deepcopy(copyobject._pars)
        self._rpn = copy.deepcopy(copyobject._rpn)
        self._patches = copy.deepcopy(copyobject._patches)

    def set_start_solution(self, pars, nobs, patches=None, rpn=None):
        self._pars = pars

        if patches is None or rpn is None:
            self.init_separate_patches(nobs)
        else:
            self.set_patches(patches, rpn)

    def init_separate_patches(self, nobs):
        """Initialize the individual coherence patches
        """
        self._patches = [[ind] for ind in range(nobs)]
        self._rpn = [[0] for ind in range(nobs)]

    def set_patches(self, patches, rpn):
        """Set the patches to something specific
        """
        self._patches = copy.deepcopy(patches)
        self._rpn = copy.deepcopy(rpn)

    def join_patches(self, ind1, ind2, apn):
        """Join patches ind1, and ind2

        Given patch indices ind1 and ind2, join those two patches

        :param ind1:
            Index of patch one

        :param ind2:
            Index of patch two

        :param apn:
            Absolute pulse numbers (relative to PEPOCH) for all TOAs
        """
        ind1, ind2 = (ind1, ind2) if (ind1 < ind2) else (ind2, ind1)  # Sort
        patches1, rpns1 = self._patches[:ind1], self._rpn[:ind1]
        patches2, rpns2 = self._patches[ind1+1:ind2], self._rpn[ind1+1:ind2]
        patches3, rpns3 = self._patches[ind2+1:], self._rpn[ind2+1:]
        patch1, rpn1 = self._patches[ind1], self._rpn[ind1]
        patch2, rpn2 = self._patches[ind2], self._rpn[ind2]

        # Relative pulse number between patches
        iprpn = apn[patch2[0]]-apn[patch1[0]]
        patch2rpn = [rpn2[ii] + iprpn for ii in range(len(rpn2))]

        # Make new patch
        newpatch = patch1 + patch2
        newrpn = rpn1 + patch2rpn

        # Save all patches
        self._patches = patches1 + [newpatch] + patches2 + patches3
        self._rpn = rpns1 + [newrpn] + rpns2 + rpns3


    def get_patches(self, fitpatch=None):
        """Get the patches, minus element `fitpatch` if not None
        """
        if fitpatch is None:
            patches = self._patches
        else:
            patches = self._patches[:fitpatch] + self._patches[(fitpatch+1):]
        return patches

    def get_number_of_toas(self, exclude_nonconnected=False):
        """Get the number of toas, possibly excluding non-connected epochs

        :param exclude_nonconnected:
            If True, exclude coherent patches of length 1
        """
        ntoas = 0
        for pp, patch in enumerate(self._patches):
            ntoas += len(patch) if len(patch) > 1 else 0

        return ntoas


    def get_rpns(self, fitpatch=None):
        """Get the relative pulse numbers, minus element `fitpatch` if not None
        """
        if fitpatch is None:
            rpn = self._rpn
        else:
            rpn = self._rpn[:fitpatch] + self._rpn[(fitpatch+1):]
        return rpn

    @property
    def pars(self):
        return self._pars

    @property
    def npars(self):
        return len(self.pars)

    @pars.setter
    def pars(self, value):
        self._pars = value
