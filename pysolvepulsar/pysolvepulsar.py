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
import os, glob, sys
import copy
import bisect

from .candidate import CandidateSolution
from .rankreduced import get_rr_CiA
from .units import has_astropy_unit, un_unitize
from .linearfitter import LinearFitter

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

# Use the same interface for pint and libstempo
try:
    import pint.ltinterface as lti
    pintpulsar = lti.pintpulsar
    print("PINT available")
    have_pint = True
except ImportError:
    pintpulsar = None
    print("PINT not available")
    have_pint = False
try:
    import libstempo as lt
    tempopulsar = lt.tempopulsar
    print("Libstempo available")
    have_libstempo = True
except ImportError:
    tempopulsar = None
    print("Libstempo not available")
    have_libstempo = False


# Parameters that have tempo2 status as parameter, but are not model parameters
tempo2_excludepars = ['START', 'FINISH', 'PEPOCH', 'POSEPOCH', 'DMEPOCH',
                      'EPHVER', 'TZRMJD', 'TZRFRQ', 'TRES', 'PSR',
                      'PLANET_SHAPIRO']

# Parameters that are not always required to be fit for
tempo2_nonmandatory = {'DM':'DMX'}

# Parameter bounds (min, max):
tempo2_parbounds = {'F0': (0.0, np.inf),
                    'RAJ': (0.0, 2*np.pi),
                    'DECJ': (-0.5*np.pi, 0.5*np.pi),
                    'SINI': (0.0, 1.0),
                    'ECC': (0.0, 1.0),
                    'E': (0.0, 1.0)
                    }

tempo2_parbounds_front = {'JUMP': ('F0', -0.5, 0.5)}


def quantize(times, dt=86400, isort=None):
    """ Produce the quantisation matrix

    Given the toas, produce the quantization matrix
    
    Note: taken from libstempo, but now uses mergesort to maintain order for
          equal elements

    :param times:
        The observation times (TOAs)

    :param dt:
        Time-lag within which we consider TOAs within the same observing epoch

    :param isort:
        In case we already have a sorting map, pass here. Otherwise it will be
        set here using mergesort
    """
    N = np

    if isort == None:
        isort = N.argsort(times, kind='mergesort')
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    t = N.array([N.mean(times[l]) for l in bucket_ind],'d')
    
    U = N.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    return t, U

def quantize_cand(times, cand, dt=86400):
    """ Produce the quantisation matrix, based on a candidate solution

    Given the toas, produce the quantization matrix. This one also takes into
    account the candidate solution, and requires the input to be _sorted_.
    
    Note: taken from libstempo, but now uses mergesort to maintain order for
          equal elements

    :param times:
        The observation times (TOAs)

    :param cand:
        The candidate solution. NOTE: the candidate solution has indices that
        refer to _sorted_ toas. However, 

    :param dt:
        Time-lag within which we consider TOAs within the same observing epoch
    """
    N = np

    if cand.nobs <= 0:
        raise ValueError("Candidate solution does not contain observations")

    isort = np.arange(len(times))

    obs2patch = cand.get_obsind2patch_map()

    # Find first observation that is not deleted
    ind = isort[0]
    for jj in isort:
        if jj in obs2patch:
            ind = jj
            break

    # Do the same as quantize here, but also split if two observations are not
    # in the same coherence patch.
    
    bucket_ref = [(times[isort[0]], obs2patch[ind])]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if i in obs2patch:
            if times[i] - bucket_ref[-1][0] < dt and \
                    bucket_ref[-1][1] == obs2patch[i]:
                bucket_ind[-1].append(i)
            else:
                bucket_ref.append((times[i], obs2patch[i]))
                bucket_ind.append([i])
    
    t = N.array([N.mean(times[l]) for l in bucket_ind],'d')
    
    U = N.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    return t, U


class PulsarSolver(object):
    """ This class provides a user interface to algorithmic timing methods

    This is the base algorithmic timing class. An instance of this class
    represents a working environment in which a single pulsar can be solved. It
    has to be initialized with a Tempo2 parfile and timfile through libstempo.

    """

    def __init__(self, parfile, timfile, priors=None, logfile=None,
            loglevel=logging.DEBUG, delete_prob=0.01, mP0=1.0,
            backend='libstempo'):
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

        :param delete_prob:
            Proability that an observation needs to be deleted. Can be an array,
            with per-obs probabilities as well (default: 0.01)

        :param mP0:
            How many pulse periods fit within one epoch

        :param backend:
            What timing package to use ('libstempo'/'pint')
        """
        self.load_pulsar(parfile, timfile, backend=backend)
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

        # Set the deleted-point probabilities
        self.delete_prob = np.ones(self.nobs) * delete_prob

        # TODO: shift the PEPOCH
        self.set_pepoch_to_center()

        # Set the quantization matrix
        self.set_quantization(mP0=mP0)

    def load_pulsar(self, parfile, timfile, backend='libstempo'):
        """Read the pulsar object from par/tim file

        Use libstempo/pint to read in a tempo2 pulsar from a parfile and a
        timfile
        """
        if (backend == 'libstempo' and have_libstempo) or \
            (backend == 'pint' and have_libstempo and not have_pint):
            psrclass = tempopulsar
        elif have_pint:
            psrclass = pintpulsar
        else:
            raise ImportError("No pulsar backend available")
        self._psr = psrclass(parfile, timfile, dofit=False)
        self.psrclass = psrclass
        self.nobs = len(self._psr.toas())
        self.name = self._psr.name

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
        """
        self._isort = np.argsort(self._psr.toas(), kind='mergesort')
        self._iisort = np.zeros_like(self._isort, dtype=np.int)
        for ii, p in enumerate(self._isort):
            self._iisort[p] = ii

    def set_quantization(self, mP0=1.0):
        """Set the quantization matrix, and the epochs

        Set the quantization matrix, and the pochs

        :param mP0:
            How many pulse periods fit within one epoch (max)
        """
        P0 = 1.0/self._psr['F0'].val
        # Quantization (timing model is treated as if the TOAs came from the
        # same pulse) occurs when TOAs are separated by less than a pulse
        # period. This is for numerical precision, they can still be shifted in
        # relative pulse number.
        #self._tepoch, self._Umat = quantize(self._psr.toas(), dt=mP0*P0/86400.0)
        self._umat_dt = mP0*P0/86400.0
        self._tepoch, self._Umat = quantize(self._psr.toas(),
                dt=self._umat_dt, isort=self._isort)
        self.nepochs = len(self._tepoch)

    def savepar(self, parfilename, dd):
        """Given a fit dictionary and a filename, save the parfile

        Given a fit dictionary and a filename, save the parfile

        :param parfilename:
            Name of the parfile to save

        :param dd:
            Results dictionary
        """
        self._psr.vals(values=dd['newpars'], which='fit')

        newerrs = self._psr.errs(which='fit')
        newerrs[dd['cmask']] = np.sqrt(np.diag(dd['Sigma']))
        newerrs = self._psr.errs(values=newerrs, which='fit')

        self._psr.savepar(parfilename)

    def get_start_solution(self):
        """Given the pulsar, get the starting solution

        Given the pulsar, get the starting solution. This places every TOA in a
        separate coherence patch, except for those within the same epoch.
        """
        patches = []
        rpns = []
        residuals = self._psr.residuals(updatebats=True,
                formresiduals=True, removemean=False)[self._isort]
        epns = self._psr.pulsenumbers(updatebats=False,
                formresiduals=False, removemean=False)[self._isort]
        P0 = 1.0/self._psr['F0'].val
        for col in self._Umat[self._isort,:].T:
            # Observations in this epoch are all coherent. Obtain the relative
            # pulse numbers relative to the first observation by minimizing the
            # residual distance.
            msk = col.astype(bool)
            dt_1 = residuals[msk]           # Residuals of epoch
            dt_2 = dt_1 - dt_1[0]           # Residuals relative to dt_1[0]
            epn_1 = epns[msk]
            epn_2 = epn_1 - epn_1[0]        # T2 relative pulse number
            rpn = (dt_2 + 0.5*P0)/P0 + epn_2

            rpns.append(list(rpn.astype(int)))
            patches.append(list(np.where(msk)[0]))

        return patches, rpns

    def get_start_candidate(self):
        """Given the pulsar, get an initial candidate solution

        Given the pulsar, get the starting solution, and initialize a candidate
        with that solution. This places every TOA in a separate coherence patch,
        except for those within the same epoch.
        """
        patches, rpns = self.get_start_solution()
        start_pars = self.get_prior_values()

        cand = CandidateSolution()
        cand.set_solution(start_pars, patches, rpns)

        return cand

    def get_prior_values(self):
        """Get an array with the prior values"""
        return np.array([self._prpars[key] for key in self._prpars])

    def get_prior_uncertainties(self):
        """Get an array with the prior uncertainties"""
        return np.array([self._prerr[key] for key in self._prerr])

    def get_prior_min(self):
        """Get an array with the prior minimum bounds"""
        return np.array([self._prmin[key] for key in self._prmin])

    def get_prior_max(self):
        """Get an array with the prior maximum bounds"""
        return np.array([self._prmax[key] for key in self._prmax])

    def get_prior_bounds_from_psr(self, key):
        """Given the parameter key, obtain the prior bound

        Get the prior bound for the parameter key.
        """
        parmin, parmax = (-np.inf, np.inf) if not key in tempo2_parbounds \
                                            else tempo2_parbounds[key]
        if key[:4] in tempo2_parbounds_front:
            # We have a JUMP or something like that
            key_link = tempo2_parbounds_front[key[:4]][0]
            par_mult = 1.0 if key_link is None else 1.0/self._psr[key_link].val
            parmin = par_mult * tempo2_parbounds_front[key[:4]][1]
            parmax = par_mult * tempo2_parbounds_front[key[:4]][2]

        return parmin, parmax

    def create_prior_from_psr(self):
        """Create the prior dictionary from the pulsar object

        Create a prior dictionary from the information in the parfile/libstempo
        object
        """
        self._logger.warn("No proper tests for priors...")
        priors = OrderedDict()
        for key in self._psr.pars(which='set'):
            if not self._psr[key].fit and not key in tempo2_excludepars \
                    and not key in tempo2_nonmandatory:
                # Automatically turn on fitting for such parameters
                self._logger.info(
                    'Fitting for {0} automatically turned on.'.format(key))
                self._psr[key].fit = True
            elif self._psr[key].fit and key in tempo2_excludepars:
                # Automatically turn off fitting for these parameters
                self._psr[key].fit = False
                self._logger.info(
                    'Fitting for {0} automatically turned off.'.format(key))
            elif self._psr[key].fit and key in tempo2_nonmandatory:
                # Leave as it is in the parfile
                pass

            if self._psr[key].err <= 0.0 and self._psr[key].fit:
                self._logger.error('Prior for {0} cannot have 0 width'.
                        format(key))

                # If it is a JUMP, set it to the pulse width
                if key[:4] in ['JUMP']:
                    self._psr[key].err = 1.0/self._psr['F0'].val

            # If we are 'fitting' for this parameter, add it to the prior
            if self._psr[key].fit:
                val = un_unitize(key, self._psr[key].val)
                err = un_unitize(key, self._psr[key].err)

                parmin, parmax = self.get_prior_bounds_from_psr(key)
                priors[key] = (val, err, parmin, parmax)

        return priors

    def init_priors(self, priors):
        """Initialize the priors from a prior dictionary.

        Initialize the priors from a prior dictionary. This dictionary values should be
        organized as (parval, parsigma)

        :param priors:
            Dictionary with the priors for the paramters. Organized as
            priors[key] = (parval, parsigma, minbound, maxbound),
            with distribution as a normal Gaussian distribution N(parval,
            parsigma)

        This corresponds to a Gaussian prior, with mean parval, and standard
        deviation parsigma.

        We use the libstempo routines without the 'Offset' parameter. We are
        including jumps for each coherency patch, so the 'Offset' is
        unnecessary. The jumps have a prior sigma equal to twice the pulse
        period, and all other 'set' parameters are only fit for if the prior is
        set. Set parameters that do not have a prior are not fitted for, and a
        warning is displayed.
        """
        self._pars = OrderedDict()
        self._prpars = OrderedDict()
        self._prerr = OrderedDict()
        self._prmin = OrderedDict()
        self._prmax = OrderedDict()

        for key in priors:
            self._psr[key].set = True       # This _has_ to be that way
            # Do we set the initial value equal to the prior?
            #self._psr[key].val = priors[key][0]

            # We keep the value, so check whether it is somewhat consistent with
            # the prior. Give a warning otherwise
            val = un_unitize(key, self._psr[key].val)
            pr = un_unitize(key, priors[key][0])
            pre = un_unitize(key, priors[key][1])
            self.check_value_against_prior(val, pr, pre)

            self._pars[key] = self._psr[key].val
            self._prpars[key] = priors[key][0]
            self._prerr[key] = priors[key][1]
            self._prmin[key] = priors[key][2]
            self._prmax[key] = priors[key][3]

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

        #prparsmin = np.array([self._prmin[key] for key in self._prmin])
        #prparsmax = np.array([self._prmax[key] for key in self._prmax])
        prparsmin = self.get_prior_min()
        prparsmax = self.get_prior_max()

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

    def get_merge_pvals(self, cand, minpval=0.01, offstd=100.0):
        """For candidate solution cand, get the merge p-value tree.

        For candidate solution cand, get the merge p-value tree down to
        proposals with a p-value of minpval. We are only proposing to merge
        adjacent patches/observations.

        The merge_pvals tree consists of n-1 list elements (n = # patches) of
        p-value lists (list of lists). The i-th element describes the merge of
        patch i and i+1, where the various p-values at this location describe
        jumps of relative phase number

        :param cand:
            CandidateSolution candidate

        :param minpval:
            The minimum p-value to include in the proposals of the relative
            phase jumps
        """
        patches = cand.get_patches()
        oii = cand.obs_inds_inv
        np = len(patches)
        P0 = float(1.0 / self._psr['F0'].val)

        merge_pvals = [[] for ii in range(np-1)]        # List of lists
        for ii, mpv in enumerate(merge_pvals):
            # Merging patch ii and ii+1. Get the p-values
            dd = self.perform_linear_least_squares_fit(cand, fitpatch=ii,
                offstd=offstd)
            next_obs = patches[ii+1][0]                 # First obs next patch
            std = float(dd['stdrp'][oii[next_obs]])     # Prediction spread
            rpe = float(dd['rp'][oii[next_obs]])        # Prediction estimate
            res = float(dd['dtp'][oii[next_obs]])       # First observation
            pest = (rpe-res)/P0                         # Point estimate in P(x)
            rps = 0.0                                   # Relative phase shift
            pval = 1.0
            while pval > minpval:
                # Non-symmetric for positive and negative phase shifts
                left, right = pest+rps-0.5, pest+rps+0.5
                lpval = sst.norm.cdf(left, loc=0.0, scale=std/P0)
                rpval = sst.norm.cdf(right, loc=0.0, scale=std/P0)
                pval = rpval-lpval

                # When rps==0, we may need a log-pvalue to rank properly. Use
                # least-ranking value
                rank = max(left, 0.0-right)
                compl = None if rps!=0 else sst.norm.logcdf(rank, scale=std/P0)
                val = (pval, compl)

                if pval > minpval:
                    mpv.append(val)
                    if rps > 0:             # Add the negative p-val one as well
                        mpv.insert(0, val)

                rps += 1

        return merge_pvals

    def get_jump_designmatrix(self, cand, fitpatch=None):
        """Obtain the design matrix of inter-coherence-patch jumps

        Obtain the design matrix of jumps that disconnect all the coherence
        patches that are not phase connected.

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            If not None, exclude this patch from the designmatrix jumps

        """
        patches = cand.get_patches(fitpatch=fitpatch)

        npatches = len(patches)
        Mj = np.zeros((self.nobs, npatches))
        for pp, patch in enumerate(patches):
            Mj[patch,pp] = True

        return Mj[cand.obs_inds,:]

    def get_jump_designmatrix_onepatch(self, cand, fitpatch=None):
        """Obtain the design matrix of one inter-coherence-patch jump

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            Get the designmatrix of the jump for this patch number
        """
        if fitpatch is not None:
            patch = cand._patches[fitpatch]
            Mj = np.zeros((self.nobs, 1))
            Mj[patch,0] = True
        else:
            Mj = np.zeros((self.nobs, 0))
        return Mj[cand.obs_inds,:]

    def get_jump_Gmatrix(self, cand, fitpatch=None):
        """Obtain the complement/G-matrix of get_jump_designmatrix

        Obtain the G-matrix that is complementary to the design matrix of jumps
        that disconnect all the coherence patches that are not phase connected

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            If not None, exclude this patch from the designmatrix jumps

        """
        # If fitpatch is not None, we have one extra column. We therefore need
        # all patches (fitpatch = None in get_patches)
        patches = cand.get_patches(fitpatch=None)
        add_fitpatch_cols = 0 if fitpatch is None else 1

        npatches = len(patches)
        Gj = np.zeros((self.nobs, cand.nobs-npatches+add_fitpatch_cols))
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

        return Gj[cand.obs_inds,:]

    def get_jump_epoch_designmatrix(self, cand, fitpatch=None):
        """Obtain the epoch-domain design matrix of inter-coherence-patch jumps

        Obtain the design matrix of jumps that disconnect all the coherence
        patches that are not phase connected. This version is carried out in the
        epoch-domain (so it's Ui Mj)

        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            If not None, exclude this patch from the designmatrix jumps

        """
        #TODO: This can be constructed directly with a little bit of extra code
        Umat = self.Umat(cand)
        self._logger.info("Umat2 = {0}".format(Umat.shape))
        Uinv = self.Umat_i(Umat=Umat)
        Mj = self.get_jump_designmatrix(cand, fitpatch=fitpatch)

        #self._logger.info("Umat = {0}".format(Umat.shape))

        return np.dot(Uinv, Mj)

    def get_jump_epoch_Gmatrix(self, cand, fitpatch=None):
        """Obtain the complement/G-matrix of get_jump_epoch_designmatrix

        Obtain the G-matrix that is complementary to the design matrix of jumps
        that disconnect all the coherence patches that are not phase connected.
        This version is carried out in the epoch-domain (so it is the complement
        of the matrix Ui Mj)


        :param cand:
            CandidateSolution candidate

        :param fitpatch:
            If not None, exclude this patch from the designmatrix jumps

        """
        #TODO:This can be constructed analytically with a bit of extra code
        Mj = self.get_jump_epoch_designmatrix(cand, fitpatch=fitpatch)
        if Mj.shape[0] == Mj.shape[1]:
            Gj = np.zeros((Mj.shape[0], 0))
        else:
            U, s, Vt = sl.svd(Mj)
            Gj = U[:,Mj.shape[1]:]

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
        the patch jump designmatrix. Faster than analytical for n<15000 or so.
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
        print("This function is being deprecated: subtract_jumps")
        MNM = np.dot(Mj.T / Nvec, Mj)       # Diagonal matrix
        Sigma = np.diag(1.0/np.diag(MNM))

        #Sigma_inv = MNM
        #try:
        #    cf = sl.cho_factor(Sigma_inv)   # Cond. number of Mj = 1. Always works
        #except np.linalg.LinAlgError as err:
        #    print("ERROR!!!")
        #    raise
        #Sigma = sl.cho_solve(cf, np.eye(len(MNM)))

        jvals = np.dot(Sigma, np.dot(Mj.T, dt / Nvec))
        dtj = np.dot(Mj, jvals)

        return dt - dtj, jvals

    def map_patch2epatch(self, inds):
        """Given indices of a patch, return the corresponding epatch

        Given indices of a patch, return the corresponding epatch. Throw an
        exception if an epoch is not fully filled.
        """
        # TODO: this can be faster without use of Umat
        Umat = self.Umat()
        Uinv = self.Umat_i(Umat=Umat)
        
        # Vector with patch indices set to one
        allvals = np.zeros(self.nobs)
        allvals[inds] = 1.0
        evals = np.dot(Uinv, allvals)

        # Check for not-completely filled epochs
        msk = (evals != 0.0)
        comp = np.ones(np.sum(msk))
        if not np.allclose(comp, evals[msk]):
            raise ValueError("Not fully coherent epoch detected in candidate")

        return np.where(msk)[0]

    def map_epatch2patch(self, einds):
        """Given indices of an epatch, return the corresponding patch

        Given indices of an epatch, return the corresponding patch.
        """
        # TODO: this can be faster without use of Umat
        Umat = self.Umat()
        evals = np.zeros(Umat.shape[1])
        evals[einds] = 1.0
        allvals = np.dot(Umat, evals)
        msk = (allvals != 0.0)
        return np.where(msk)[0]

    def pars(self, which='fit'):
        """Return a list with parameter id's"""
        return self._psr.pars(which=which)

    def get_canonical_selection(self, cand, exclude_nonconnected=False):
        """Given a candidate solution, return the quantity selection
        
        Return the selector. If exclude_nonconnected is True, then only return
        the values for which the patches in cand contain more than one toa.
        Deleted points (points not in a patch) are just returned, unless
        exclude_nonconnected=True

        Note: exclude_nonconnected will/might mess up the ordering of the
              toas

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return uncertainties within coherent patches with more
            than one residual
        """
        selection = np.array([], dtype=np.int) if exclude_nonconnected \
                                               else np.arange(self.nobs)

        if exclude_nonconnected and cand is not None:
            patches = cand.get_patches()
            for pp, patch in enumerate(patches):
                # Only return if requested
                if len(patch) > 1:
                    selection = np.append(selection, patch)
        return selection

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
                formresiduals=True, removemean=False)[self._isort]
        pn_lt = self._psr.pulsenumbers(updatebats=False,
                formresiduals=False, removemean=False)[self._isort]
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

    def pulsenumbers(self, cand, exclude_nonconnected=False):
        """Return the pulse numbers relative to the PEPOCH

        Given a candidate solution, return the pulse numbers, relative to the
        PEPOCH, adjusted for the relative phase jumps in the candidate solution

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual
        """
        # Obtain the residuals as tempo2 would calculate them
        self._psr.vals(which='fit', values=cand.pars)
        pn_lt = self._psr.pulsenumbers(updatebats=True,
                formresiduals=True, removemean=False)[self._isort]
        pn = pn_lt.copy()

        selection = np.array([], dtype=np.int) if exclude_nonconnected \
                                               else np.arange(len(pn))
        rpns = cand.get_rpns()
        patches = cand.get_patches()
        for pp, patch in enumerate(patches):
            # Relative pulse numbers of patch, and from tempo2
            rpn = np.array(rpns[pp])
            rpn_lt = pn_lt[patch] - pn_lt[patch[0]]
            pn[patch] += (rpn - rpn_lt)

            if exclude_nonconnected and len(patch) > 1:
                selection = np.append(selection, patch)

        return pn[selection]

    def freqs(self, cand=None, exclude_nonconnected=False):
        """Return the freqs, possibly excluding non-connected toas

        Return the site freqs. If exclude_nonconnected is True, then only return
        the toas for which the patches in cand contain more than one toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              residuals

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual
        """
        if cand is not None:
            self._psr.vals(which='fit', values=cand.pars)

        freqs = self._psr.freqs[self._isort]
        selection = self.get_canonical_selection(cand, exclude_nonconnected)

        return freqs[selection]

    def stoas(self, cand=None, exclude_nonconnected=False):
        """Return the site toas, possibly excluding non-connected toas

        Return the site toas. If exclude_nonconnected is True, then only return
        the toas for which the patches in cand contain more than one toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              residuals

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual
        """
        if cand is not None:
            self._psr.vals(which='fit', values=cand.pars)

        stoas = self._psr.stoas[self._isort]
        selection = self.get_canonical_selection(cand, exclude_nonconnected)

        return stoas[selection]

    def toas(self, cand=None, exclude_nonconnected=False):
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
        if cand is not None:
            self._psr.vals(which='fit', values=cand.pars)

        toas = self._psr.toas(updatebats=True)[self._isort]
        selection = self.get_canonical_selection(cand, exclude_nonconnected)

        return toas[selection]

    def toaerrs(self, cand=None, exclude_nonconnected=False):
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
        toaerrs = self._psr.toaerrs[self._isort]*1e-6
        selection = self.get_canonical_selection(cand, exclude_nonconnected)
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
                fixsigns=True, incoffset=False)[self._isort,:]

        selection = self.get_canonical_selection(cand, exclude_nonconnected)

        return M[selection,:]

    def Umat(self, cand=None, exclude_nonconnected=False):
        """Return the quantization matrix, possibly excluding non-connected toas

        Return the quantization matrix. If exclude_nonconnected is True, then
        only return the toas for which the patches in cand contain more than one
        toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              residuals

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual
        """
        #Umat = self._Umat[self._isort,:]
        te, Umat = quantize_cand(self._psr.toas()[self._isort], cand,
                dt=self._umat_dt)
        selection = self.get_canonical_selection(cand, exclude_nonconnected)
        return Umat[selection,:]

    def Umat_i(self, cand=None, exclude_nonconnected=False, Umat=None,
            weighted=True):
        """Return the left-inverse quantization matrix

        Return the left-inverse quantization matrix. If exclude_nonconnected is
        True, then only return the toas for which the patches in cand contain
        more than one toa. This matrix is essentially the matrix that performs
        unweighted epoch-averaging

        Note: exclude_nonconnected will/might mess up the ordering of the
              residuals

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return resiudals within coherent patches with more
            than one residual

        :param Umat:
            If not None, this is the Umat matrix, so we do not have to
            re-calculate that quantity

        :param weighted:
            If True (default), weight the TOAs by their error bars. Due to the
            non-uniqueness of the Umat_i, this will still be a valid
            left-inverse of Umat. Not possible when Umat is given as an argument
        """
        if Umat is None:
            Umat = self.Umat(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)

        if weighted and not Umat is None:
            w = 1.0 / self.toaerrs(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)**2
        else:
            w = np.ones(Umat.shape[0])

        Uw = (Umat.T * w).T
        return ((1.0/np.sum(Uw, axis=0)) * Uw).T

    def get_chi2_pvalue(self, dd):
        """Return the chi2 p-value for this solve

        For the provided linear solution, calculate the chi2 and p-value,
        properly correcting for the number of degrees of freedom.  We are doing
        a one-sided test here.

        The chi2 we are calculating is from the likelihood:
        xi^2 =  (dt - M ksi)^T N^{-1} (dt - M ksi)

        Where dt, N, and M are projected with the G-matrix

        :param dd:
            The linear solution
        """
        dt = dd['dtp']
        M = dd['Mp']
        N_cf = dd['Np_cf']

        # We calculate the p-value from chi2 and the degrees of freedom. If
        # dof=0, the p-value equals 1 (no discrepancy).
        dof = max(len(dt)-M.shape[1], 0)
        pval = 1.0
        chi2 = 1.0

        if dof > 0:
            chi2 = float(np.dot(dt, sl.cho_solve(N_cf, dt)))
            pval = sst.chi2.sf(chi2, dof)

        return chi2, dof, pval

    def get_efac_distribution(self, efac, scale=5.0):
        """Given an efac, return the p-value"""
        if efac > 1.0:
            return 1.0 / (1.0 + (efac/scale)**2)
        else:
            return 1.0

    def get_efac_pval(self, dd):
        """Return the efac p-value for this solve

        For the provided linear solution, calculate the p-value of the efac.

        The efac we are calculating is the one that sets the xi^2 equal to one:
        xi^2 =  (dt - M ksi)^T N^{-1} (dt - M ksi)

        efac = sqrt(xi2/dof)

        Where dt, N, and M are projected with the G-matrix

        :param dd:
            The linear solution
        """
        dt = dd['dtp']
        M = dd['Mp']
        N_cf = dd['Np_cf']

        # We calculate the p-value from chi2 and the degrees of freedom. If
        # dof=0, the p-value equals 1 (no discrepancy).
        #dof = max(len(dt)-M.shape[1], 0)
        dof = len(dt)
        pval = 1.0
        efacpval = 1.0
        chi2 = 1.0
        efac = 1.0

        if dof > 0:
            chi2 = float(np.dot(dt, sl.cho_solve(N_cf, dt)))
            pval = sst.chi2.sf(chi2, dof)
            efac = np.sqrt(chi2 / dof)
            efacpval = self.get_efac_distribution(efac)

        return chi2, dof, pval, efacpval, efac

    def fit_constrained_iterative(self, cand, lltol=0.1, maxiter=10,
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
        newcand = CandidateSolution(cand)
        ass_cmin = np.zeros(newcand.npars, dtype=np.bool)
        ass_cmax = np.zeros(newcand.npars, dtype=np.bool)

        notdone = True
        loglik, prevloglik = np.inf, np.inf
        niter = 0

        while notdone and niter < maxiter:
            self._logger.info("Fitting iteration {0}".format(niter))
            dd = self.perform_linear_least_squares_fit(newcand, offstd=100.0)
            loglik = dd['loglik_ml']
            #newpars = dd['newpars']
            #newpars[dd['cmask']] += dd['dpars']
            prparsmin = self.get_prior_min()
            prparsmax = self.get_prior_max()
            newcand.pars[dd['cmask']] += dd['dpars']
            #newcand.pars = newpars
            niter += 1

            #ass_cmin = (newpars < prparsmin)
            #ass_cmax = (newpars > prparsmax)
            ass_cmin = (newcand.pars < prparsmin)
            ass_cmax = (newcand.pars > prparsmax)

            if np.sum(np.append(ass_cmin, ass_cmax)) == 0 and \
                    np.abs(loglik - prevloglik) <= lltol:
                notdone = False
            else:
                # LOG: CONSTRAINTS (CHECK IT)
                self._logger.info("Constraint violated: {0}".format(
                        np.logical_or(ass_cmin, ass_cmax)))
                self._logger.info("dd['cmask'] = {0}".format(dd['cmask']))
                prevloglik = loglik

        if niter == maxiter:
            self._logger.warn("Maximum number of iterations reached")

        return newcand, loglik

    def perform_linear_least_squares_fit_old(self, cand, ass_cmin=None,
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
        nobs = cand.nobs
        obs_inds = cand.obs_inds

        newcand, cmask = self.make_pars_respect_constraints(cand,
                ass_cmin, ass_cmax)

        # Create the full design matrix
        # If we use 'fitpatch', we do not include a jump for patch 'fitpatch' in
        # Mj. Instead, we include an offset for the entire dataset (Mo), because
        # we need the uncertainty of the offset when doing prediction.
        Mj = self.get_jump_designmatrix(newcand, fitpatch=fitpatch)
        Mo = np.ones((nobs, 0 if fitpatch is None else 1))
        Mt = self.designmatrix(newcand)[:,cmask][obs_inds]
        Mtot = np.append(Mt, Mo, axis=1)
        Gj = self.get_jump_Gmatrix(newcand, fitpatch=fitpatch)
        
        # The parameter identifiers/keys
        parlabelst = list(np.array(self._psr.pars(which='fit'))[cmask])
        parlabelso = ['PatchOffset'] * Mo.shape[1]
        parlabels = parlabelst + parlabelso

        npart = Mtot.shape[1]
        nparj = Mj.shape[1]

        # Create the noise and prior matrices
        Nvec = self.toaerrs(newcand)[obs_inds]**2
        Phivect = (self.get_prior_uncertainties()**2)[cmask]
        Phiveco = np.array([offstd/self._psr['F0'].val]*Mo.shape[1])**2
        Phivec = np.append(Phivect, Phiveco)
        Phivec_inv = 1.0/Phivec
        prparst = self.get_prior_values()[cmask]
        prparso = np.array([0.0]*Mo.shape[1])
        prpars = np.append(prparst, prparso)
        prpars_delta = np.append(prparst-cand.pars, prparso-prparso)

        # phipar is the ML value of the TM parameters. But: we are calculating
        # it with respect to the current value of the TM parameters.
        phipar = prpars_delta * Phivec_inv  # Instead of: phipar = prpars * Phivec_inv

        # Set psr parameters, and remove the phase offsets of the patches
        dt_lt = self.residuals(newcand)[obs_inds]
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

            if Mp.shape[0] < Mp.shape[1]:
                # Fewer effective observations than parameters.
                # Do Woodbury the other way around
                try:
                    Np_cf = sl.cho_factor(Np)
                    MNM = np.dot(Mp.T, sl.cho_solve(Np_cf, Mp))
                    Sigma_inv = MNM + np.diag(Phivec_inv)
                    # Use rank-reduced Cholesky code to invert Sigma_inv
                    Sigma = get_rr_CiA(Phivec_inv, Mp.T, 1.0/np.diag(Np), np.eye(len(Phivec_inv)))
                except np.linalg.LinAlgError as err:
                    print("Inverse Woodbury also has problems... :(")
                    raise
            else:
                try:
                    Np_cf = sl.cho_factor(Np)
                except np.linalg.LinAlgError as err:
                    print("Cho Np = ", Np)
                    np.savetxt('Np.txt', Np)
                    raise
                MNM = np.dot(Mp.T, sl.cho_solve(Np_cf, Mp))
                Sigma_inv = MNM + np.diag(Phivec_inv)

                # TODO: Use regularized Mtot??
                try:
                    Sigma_inv_cf = sl.cho_factor(Sigma_inv)
                    Sigma = sl.cho_solve(Sigma_inv_cf, np.eye(len(MNM)))
                except np.linalg.LinAlgError as err:
                    print("Cho Sigma_inv = ", Sigma_inv)
                    np.savetxt('Sigma_inv.txt', Sigma_inv)
                    raise

            # Calculate the prediction quantities
            MNt = np.dot(Mp.T, sl.cho_solve(Np_cf, dtp))
            dpars = np.dot(Sigma, MNt + phipar)
            rp = np.dot(Mtot, np.dot(Sigma, MNt))   # Should be approx~0.0
            rr = np.dot(Mtot, np.dot(Sigma, Mtot.T))

            # Calculate the log-likelihood
            logdetN2 = np.sum(np.log(np.diag(Np_cf[0])))
            logdetphi2 = 0.5*np.sum(np.log(Phivec))
            chi2dt = 0.5*np.dot(dtp, sl.cho_solve(Np_cf, dtp))
            chi2phi = 0.5*np.sum(prpars_delta**2/Phivec)
            chi2phi1 = 0.5*np.dot(dpars, np.dot(Sigma_inv, dpars))
            chi2_active = 0.5*np.dot(dpars, np.dot(Sigma_inv, dpars))
            # NOTE: chi2_active is zero _if_ we move to ML solution. We are dpars
            #       away from there. That's why we subtract it from loglik.
            #       Note also that, now, chi2phi1 and zi2_active are the same in
            #       this rescaling
            loglik = -logdetN2-logdetphi2-chi2dt-chi2phi+chi2phi1-chi2_active
            loglik_ml = -logdetN2-logdetphi2-chi2dt-chi2phi+chi2phi1
        else:
            raise ValueError("# of patches is higher than # of observations")

        # Wrap up in a dictionary, and return
        dd['dt'] = dt                           # Timing residuals
        dd['dtp'] = dtp                         # Projected timing residuals
        dd['Mj'] = Mj                           # Jump design matrix
        dd['Mt'] = Mt                           # Timing model design matrix
        dd['Mo'] = Mo                           # Offset design matrix
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
        dd['cmask'] = cmask                     # The non-constraints mask
        dd['phipar'] = phipar                   # TODO: unnecessary?
        dd['newpars'] = newcand.pars
        return dd

    def perform_linear_least_squares_fit_dep(self, cand, ass_cmin=None,
            ass_cmax=None, fitpatch=None, offstd=100.0):
        """Perform a constrained, linear, least-squares fit

        Perform a constrained, linear, least-squares fit. Internally, normalize
        the design matrix in order to regularize the linear algebra. Results
        should remain the same as before

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
        nobs = cand.nobs
        obs_inds = cand.obs_inds

        newcand, cmask = self.make_pars_respect_constraints(cand,
                ass_cmin, ass_cmax)

        # Create the full design matrix
        # If we use 'fitpatch', we do not include a jump for patch 'fitpatch' in
        # Mj. Instead, we include an offset for the entire dataset (Mo), because
        # we need the uncertainty of the offset when doing prediction.
        Mj = self.get_jump_designmatrix(newcand, fitpatch=fitpatch)
        Mo = np.ones((nobs, 0 if fitpatch is None else 1))
        Mt = self.designmatrix(newcand)[:,cmask][obs_inds]
        Mtot_orig = np.append(Mt, Mo, axis=1)
        Gj = self.get_jump_Gmatrix(newcand, fitpatch=fitpatch)
        
        # The parameter identifiers/keys
        parlabelst = list(np.array(self._psr.pars(which='fit'))[cmask])
        parlabelso = ['PatchOffset'] * Mo.shape[1]
        parlabels = parlabelst + parlabelso

        npart = Mtot_orig.shape[1]
        nparj = Mj.shape[1]

        # Create the noise and prior matrices
        Nvec = self.toaerrs(newcand)[obs_inds]**2
        Phivect = (self.get_prior_uncertainties()**2)[cmask]
        Phiveco = np.array([offstd/self._psr['F0'].val]*Mo.shape[1])**2
        Phivec_orig = np.append(Phivect, Phiveco)
        Phivec_inv_orig = 1.0/Phivec_orig

        # We have Mtot_orig and Phivec_orig. Do the normalization
        nv = np.mean(Nvec)
        mu = np.sum(Mtot_orig**2, axis=0)
        u = np.sqrt(mu/nv + Phivec_inv_orig)        # New units/normalization
        Mtot = Mtot_orig / u

        Phivec = np.append(Phivect, Phiveco) * u**2
        Phivec_inv = 1.0/Phivec

        prparst = self.get_prior_values()[cmask]
        prparso = np.array([0.0]*Mo.shape[1])
        prpars = np.append(prparst, prparso) * u
        prpars_delta = prpars - np.append(cand.pars, prparso) * u

        # phipar is the ML value of the TM parameters. But: we are calculating
        # it with respect to the current value of the TM parameters.
        phipar = prpars_delta * Phivec_inv  # Instead of: phipar = prpars * Phivec_inv

        # Set psr parameters, and remove the phase offsets of the patches
        dt_lt = self.residuals(newcand)[obs_inds]
        dt, jvals = self.subtract_jumps(dt_lt, Mj, Nvec)

        nepochs = len(self.map_patch2epatch(obs_inds))
        if nparj == nepochs: #nobs:
            # We know nothing but the prior. No fit for parameters
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
            Npi_dtp = np.zeros(0)

            loglik = 0.0
            loglik_ml = 0.0

            # Transform the units back
            dpars = dpars / u
            Sigma_inv = ((Sigma_inv*u).T*u).T
            Sigma = ((Sigma/u).T/u).T
        elif nparj < nepochs: # nobs:
            # We have to fit for the timing model
            # Transform M, N, dt
            Mp = np.dot(Gj.T, Mtot)
            dtp = np.dot(Gj.T, dt)
            Np = np.dot(Gj.T * Nvec, Gj)

            if nepochs-nparj < npart: # Mp.shape[0] < Mp.shape[1]:  #<= Mp.shape[1]+20:
                # Fewer effective observations than parameters
                # Do Woodbury the other way around
                print("Epochs", self.nobs, self.nepochs)
                if self.nobs != self.nepochs:
                    # Some observations are in the same epoch, & inverse Woodbury
                    Umat = self.Umat()[obs_inds,:]
                    eps_inds = np.sum(Umat, axis=0) > 0       # Epoch indices
                    Umat = Umat[:,eps_inds]
                    Uinv = self.Umat_i(Umat=Umat)
                    Nevec = 1.0/np.diag(np.dot(Umat.T / Nvec, Umat))
                    Mte = np.dot(Uinv, Mtot)

                    # Get the epoch_Gmatrix
                    Mje = self.get_jump_epoch_designmatrix(
                            newcand, fitpatch=fitpatch)[eps_inds,:]
                    U, s, Vt = sl.svd(Mje)
                    Gje = U[:,Mje.shape[1]:]

                    Npe = np.dot(Gje.T * Nevec, Gje)
                    Mpe = np.dot(Gje.T, Mte)
                    #dtpe = np.dot(Gje.T, # CONTINUE HERE

                    try:
                        Npe_cf = sl.cho_factor(Npe)
                        MNM = np.dot(Mpe.T, sl.cho_solve(Npe_cf, Mpe))
                        Sigma_inv = MNM + np.diag(Phivec_inv)

                        Ce = Npe + np.dot(Mpe * Phivec, Mpe.T)
                        Ce_cf = sl.cho_factor(Ce)
                        CeiMP = sl.cho_solve(Ce_cf, Mpe * Phivec)
                        Sigma = np.diag(Phivec) - \
                                np.dot((Mpe * Phivec).T, CeiMP)
                        #Npi_dtp = np.dot(Mpe, sl.cho_solve(Npe_df, d
                        print("Did the fit, biatch!")
                    except np.linalg.LinAlgError as err:
                        print("Inverse Woodbury also has problems... :(")
                        raise
                else:
                    # No observations in the same epoch. Inverse Woodbury
                    try:
                        # Get the quantization matrix
                        Np_cf = sl.cho_factor(Np)
                        MNM = np.dot(Mp.T, sl.cho_solve(Np_cf, Mp))
                        Sigma_inv = MNM + np.diag(Phivec_inv)
                        # Use rank-reduced Cholesky code to invert Sigma_inv
                        Sigma = get_rr_CiA(Phivec_inv, Mp.T, 1.0/np.diag(Np), np.eye(len(Phivec_inv)))
                    except np.linalg.LinAlgError as err:
                        print("Inverse Woodbury also has problems... :(")
                        raise

                #Phi = np.diag(Phivec)
                #S2 = np.dot(Mp, np.dot(Phi, Mp.T)) + Np
                #cf = sl.cho_factor(S2)
                #Sigma = Phi - np.dot(Phi, np.dot(Mp.T, sl.cho_solve(cf, np.dot(Mp, Phi))))
            else:
                # Regular Woodbury-type fit for the timing model
                try:
                    Np_cf = sl.cho_factor(Np)
                except np.linalg.LinAlgError as err:
                    print("ChoN Np = ", Np)
                    #np.savetxt('Np.txt', Np)
                    raise
                MNM = np.dot(Mp.T, sl.cho_solve(Np_cf, Mp))
                Sigma_inv = MNM + np.diag(Phivec_inv)

                # TODO: Use regularized Mtot??
                try:
                    Sigma_inv_cf = sl.cho_factor(Sigma_inv)
                    Sigma = sl.cho_solve(Sigma_inv_cf, np.eye(len(MNM)))
                    #Qs, Rs = sl.qr(Sigma_inv) 
                    #Sigma = sl.solve(Rs, Qs.T)
                except np.linalg.LinAlgError as err:
                    print("ChoN Sigma_inv = ", Sigma_inv)
                    #np.savetxt('Sigma_inv.txt', Sigma_inv)
                    #self.perform_linear_least_squares_fit(cand=cand,
                    #        ass_cmin=ass_cmin,
                    #        ass_cmax=ass_cmax, fitpatch=fitpatch, offstd=offstd)
                    raise

            # Calculate the prediction quantities
            #MNt = np.dot(Mp.T, sl.cho_solve(Np_cf, dtp))
            MNt = np.dot(Mp.T, Npi_dtp)
            dpars = np.dot(Sigma, MNt + phipar)
            # TODO: should use dpars, instead of MNt below here???
            rp = np.dot(Mtot, np.dot(Sigma, MNt))   # Should be approx~0.0
            rr = np.dot(Mtot, np.dot(Sigma, Mtot.T))

            # Calculate the log-likelihood
            logdetN2 = np.sum(np.log(np.diag(Np_cf[0])))
            logdetphi2 = 0.5*np.sum(np.log(Phivec))
            chi2dt = 0.5*np.dot(dtp, sl.cho_solve(Np_cf, dtp))
            chi2phi = 0.5*np.sum(prpars_delta**2/Phivec)
            chi2phi1 = 0.5*np.dot(dpars, np.dot(Sigma_inv, dpars))
            chi2_active = 0.5*np.dot(dpars, np.dot(Sigma_inv, dpars))
            # NOTE: chi2_active is zero _if_ we move to ML solution. We are dpars
            #       away from there. That's why we subtract it from loglik.
            #       Note also that, now, chi2phi1 and zi2_active are the same in
            #       this rescaling
            loglik = -logdetN2-logdetphi2-chi2dt-chi2phi+chi2phi1-chi2_active
            loglik_ml = -logdetN2-logdetphi2-chi2dt-chi2phi+chi2phi1

            # Transform the units back
            MNM = ((MNM/u).T/u).T
            MNt = MNt / u
            Phivec = Phivec / u**2
            Phivec_inv = Phivec_inv*u**2
            Sigma_inv = ((Sigma_inv*u).T*u).T
            Sigma = ((Sigma/u).T*u).T
            dpars = dpars / u
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
        # TODO: The abs here is totally incorrect!!!
        dd['stdrp'] = np.sqrt(np.diag(np.abs(rr)))      # Residuals projection std
        dd['parlabels'] = parlabels             # Parameter labels
        dd['loglik'] = loglik                   # Log-likelihood
        dd['loglik_ml'] = loglik_ml             # Log-likelihood (ML)
        dd['cmask'] = cmask                     # The non-constraints mask
        dd['newpars'] = newcand.pars
        return dd

    def perform_linear_least_squares_fit(self, cand, ass_cmin=None,
            ass_cmax=None, fitpatch=None, offstd=100.0, normalize=True):
        """Perform a constrained, linear, least-squares fit

        Perform a constrained, linear, least-squares fit. Internally, normalize
        the design matrix in order to regularize the linear algebra. Results
        should remain the same as before

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
        nobs = cand.nobs
        obs_inds = cand.obs_inds

        newcand, cmask = self.make_pars_respect_constraints(cand,
                ass_cmin, ass_cmax)

        # Use the U-matrix to do the epoch-averaging (use obs_inds)
        Umat = self.Umat(newcand)[obs_inds]
        self._logger.info("Umat1 = {0}".format(Umat.shape))
        toaerrs_f = self.toaerrs(newcand)[obs_inds]
        Uw = (Umat.T / toaerrs_f**2).T
        Umat_i = ((1.0/np.sum(Uw, axis=0)) * Uw).T  # A left-inverse of Umat

        # Get the epoch-ave data (again, use obs_inds), weighted by 1/toaerrs**2
        #print("newcand.pars = ", newcand.pars)
        #foobar = self.residuals(copy.deepcopy(newcand))
        #barfoo = self.get_jump_epoch_Gmatrix(newcand, fitpatch=fitpatch)  # NOT same as dot(Umat_i, Gj_f)
        residuals = np.dot(Umat_i, self.residuals(newcand)[obs_inds])
        toaerrs = 1.0 / np.sqrt( np.sum(Uw, axis=0) )

        # Create the full design matrix components
        # If we use 'fitpatch', we do not include a jump for patch 'fitpatch' in
        # Mj. Instead, we include an offset for the entire dataset (Mo), because
        # we need the uncertainty of the offset when doing prediction.
        Mj_f = self.get_jump_designmatrix(newcand, fitpatch=fitpatch)[obs_inds]
        Mo_f = np.ones((nobs, 0 if fitpatch is None else 1))
        Mt_f = self.designmatrix(newcand)[:,cmask][obs_inds]

        Mj = np.dot(Umat_i, Mj_f)       # Same as get_jump_epoch_designmatrix
        Mo = np.dot(Umat_i, Mo_f)
        Mt = np.dot(Umat_i, Mt_f)
        Gj = self.get_jump_epoch_Gmatrix(newcand, fitpatch=fitpatch)  # NOT same as dot(Umat_i, Gj_f)

        # The parameter identifiers/keys
        parlabelst = list(np.array(self._psr.pars(which='fit'))[cmask])
        parlabelso = ['PatchOffset'] * Mo.shape[1]

        # The prior information
        Phivect = (self.get_prior_uncertainties()**2)[cmask]
        Phiveco = np.array([offstd/self._psr['F0'].val]*Mo.shape[1])**2
        prparst = self.get_prior_values()[cmask]
        prparso = np.array([0.0]*Mo.shape[1])

        lf = LinearFitter(residuals=residuals, toaerrs=toaerrs, Mj=Mj, Mo=Mo,
                          Mt=Mt, Gj=Gj, parlabelst=parlabelst,
                          parlabelso=parlabelso, Phivect=Phivect,
                          Phiveco=Phiveco, prparst=prparst, prparso=prparso,
                          candpars=newcand.pars, normalize=normalize,
                          _logger=self._logger)

        # Return the fitting dictionary
        dd = lf.fit()
        dd['cmask'] = cmask
        return dd

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

        TODO: DEPRECATED. Just use perform_linear_least_squares_fit
        """
        self._logger.warn('DEPRECATED FUNCTION: get_linear_solution')
        dd = self.perform_linear_least_squares_fit(cand, fitpatch=fitpatch,
                offstd=offstd)
        return dd

    def solve(self, maxit=np.inf, verbose=False):
        """Solve the pulsar hierarchically.

        Solve the pulsar hierarchically using the candidate tree. Do not explore
        options with a prior probability of less than min_pval.
        """
        # Root_cand is the root of the entire candidate tree
        root_cand = CandidateSolution()
        start_pars = self.get_prior_values()

        # Obtain the start patches, taking into account simultaneous
        # observations
        patches, rpn = self.get_start_solution()
        root_cand.set_solution(start_pars, patches, rpn)

        root_cand.integrate_in_tree()

        cand_by_pval = [root_cand]      # This will be a sorted list (bisect)
        cur_cand = root_cand

        prop_hist = dict()      # Full proposal history to prevent duplicates
        prop_hist[cur_cand.get_history_hashable()] = True

        counter = 0
        while not cur_cand.is_coherent() and counter<maxit:
            # Optimize the next proposed candidate solution (constrained fit)
            cur_cand = cand_by_pval.pop()       # Largest p-value is popped
            new_cand, loglik = self.fit_constrained_iterative(cur_cand)
            cur_cand.pars = new_cand.pars.copy()
            if verbose:
                print("[{0}]: ".format(counter) +
                        ', '.join(cur_cand.get_history()))

            # Perform an unconstrained fit for p-values, and get stats
            dd = self.perform_linear_least_squares_fit(cur_cand)
            #chi2, dof, pval = self.get_chi2_pvalue(dd)
            chi2, dof, pval, efacpval, efac = self.get_efac_pval(dd)
            pulse_numbers = self.pulsenumbers(cur_cand)
            if verbose:
                print("   --- ", efac, efacpval, chi2, dof, pval)

            # Get the tree of merge p-values of children for this candidate
            merge_pvals = self.get_merge_pvals(cur_cand, minpval=0.001)

            # Register all this tree information to the current candidate
            cur_cand.register_optimized_results(efacpval, chi2, dof,
                    merge_pvals)

            # From here onwards, we are adding candidate solutions (proposals)
            # to the tree and the sorted list. Add any advanced proposals to
            # improve performance

            # TODO: rpn=0 is not always the most likely merger, so it should not
            #       always be first. What needs to be changed:
            #       1) get_merge_pvals needs to become asymmetric (done)
            #       2) get_merge_pvals should save information to further build
            #          the merge tree
            #       3) have_child_phase_shift_pval could be replaced by a
            #          function that automatically extends the merge tree?
            #       4) initialize_children_rpn0 should obtain the highest
            #          likelihood phase shift from somewhere, and register that
            #          value. Building the tree becomes asymmetric at that point
            #       5) is_parent_delete_trigger should use that registered value

            # If a child candidate was the first of a patch merger (highest
            # likelihood phase shift), we will add the option to delete the
            # single-point patches.
            # TODO: Should rpn shifts be added here as well, or done separately?
            additions = []
            if cur_cand.is_parent_rpn_trigger():
                rpn, p1, p2 = cur_cand.get_origin_patches()
                additions += cur_cand._parent.notify_patches_merged(
                        rpn, p1, p2, self.delete_prob, pulse_numbers)

            # Add candidates from cur_cand (only rpn = 0 initially)
            additions += cur_cand.initialize_children_rpn0(pulse_numbers)

            # Add all additions into the cand_by_pval sorted list
            for addition in additions:
                hist = addition.get_history_hashable()
                if not hist in prop_hist:
                    # Only add this proposal, if we have not used it before
                    ind = bisect.bisect_right(cand_by_pval, addition)
                    cand_by_pval.insert(ind, addition)
                    prop_hist[hist] = True

            counter += 1

        return root_cand, cand_by_pval

